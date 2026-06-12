"""GPU memory estimation for 3D FDTD adjoint runs.

Codifies the memory model of ``docs/research/08-toolchain.md`` (section 1)
for the actual `gradenna.fdtd3d.simulate_3d` state layout, so a run can be
sized against a given GPU *before* it is launched:

- forward state: the six Yee field components plus the twelve CPML psi
  accumulators (reported both for full-grid psi arrays, the conservative
  layout, and for PML-strip storage, where each psi array only spans the
  2 x thickness cell planes of its own axis);
- running-DFT monitors: ``n_dft_freqs`` complex copies of all six
  components, which travel inside the scan carry and are therefore
  duplicated by every checkpoint snapshot;
- reverse-mode AD: the naive all-steps bound and the sqrt-N checkpointing
  peak ``(K + n_steps/K) * state`` of ``checkpoint_segments=K``
  (research note 08, section 1.1).

All numbers are first-order array-size accounting: XLA temporaries can add
a factor ~1.5-2x on top, so keep headroom (the GPU-fit helper defaults to
85% usable VRAM).
"""

from __future__ import annotations

import numpy as np

from gradenna.grid import Grid3D

__all__ = [
    "fdtd3d_memory_estimate",
    "fits_gpu",
    "gpu_fit_report",
]

_GIB = float(1024**3)

#: Common consumer-GPU VRAM sizes [GB] used by :func:`gpu_fit_report`.
COMMON_VRAM_GB = (24.0, 16.0, 8.0)


def _bytes_per_value(dtype) -> int:
    return int(np.dtype(dtype).itemsize)


def fdtd3d_memory_estimate(
    grid: Grid3D,
    n_steps: int,
    *,
    checkpoint_segments: int | None = None,
    n_dft_freqs: int = 0,
    dtype="float32",
    cpml_thickness: int = 10,
) -> dict:
    """Estimate the peak memory [GB] of a differentiated `simulate_3d` run.

    Args:
        grid: the 3D Yee grid.
        n_steps: number of FDTD time steps.
        checkpoint_segments: K of ``simulate_3d(checkpoint_segments=K)``;
            None defaults to the sqrt-N choice ``round(sqrt(n_steps))`` for
            the checkpoint estimate.
        n_dft_freqs: number of full-grid running-DFT frequencies (0 = no
            DFT monitor). Each frequency stores six complex field copies
            inside the scan carry.
        dtype: real dtype of the simulation (float32 / float64). The DFT
            accumulators use the matching complex dtype (2x the real size).
        cpml_thickness: CPML thickness in cells (for the psi-strip variant).

    Returns:
        dict with (all sizes in GB unless suffixed ``_bytes``):
            n_cells, bytes_per_value, n_steps, checkpoint_segments,
            fields_gb           six Yee components (one time level),
            materials_gb        update-coefficient arrays (~4 cell fields),
            psi_full_gb         twelve full-grid psi arrays (current layout),
            psi_strip_gb        twelve PML-strip psi arrays,
            dft_monitor_gb      running-DFT accumulators (complex),
            state_full_gb       scan carry = fields + psi_full + dft,
            state_strip_gb      scan carry = fields + psi_strip + dft,
            forward_gb          forward peak (state_full + materials),
            adjoint_naive_gb    naive reverse-mode AD (state x n_steps),
            adjoint_checkpoint_gb        (K + n_steps/K) x state_full,
            adjoint_checkpoint_strip_gb  same with strip psi.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if n_dft_freqs < 0:
        raise ValueError("n_dft_freqs must be >= 0")
    k = (
        int(round(np.sqrt(n_steps)))
        if checkpoint_segments is None
        else int(checkpoint_segments)
    )
    if k < 1:
        raise ValueError("checkpoint_segments must be >= 1")

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    n_cells = nx * ny * nz
    b = _bytes_per_value(dtype)

    # Exact staggered shapes of the six Yee components (fdtd3d module
    # docstring); each is within one plane of n_cells.
    field_values = (
        (nx - 1) * ny * nz  # Ex
        + nx * (ny - 1) * nz  # Ey
        + nx * ny * (nz - 1)  # Ez
        + nx * (ny - 1) * (nz - 1)  # Hx
        + (nx - 1) * ny * (nz - 1)  # Hy
        + (nx - 1) * (ny - 1) * nz  # Hz
    )
    fields_bytes = field_values * b

    # Materials: eps_r and sigma are folded into ~4 coefficient arrays
    # (ca, cb at cell granularity plus the broadcast inputs); research
    # note 08 uses the same count.
    materials_bytes = 4 * n_cells * b

    # CPML psi: 12 arrays (two transverse derivative terms per component).
    psi_full_bytes = 12 * n_cells * b
    # Strip storage: a psi array for the derivative along axis a is nonzero
    # only in the two PML slabs of that axis (2 * thickness planes). Four
    # of the twelve arrays are associated with each axis.
    t = min(cpml_thickness, nx // 2, ny // 2, nz // 2)
    psi_strip_bytes = int(
        4 * n_cells * b * 2 * t * (1.0 / nx + 1.0 / ny + 1.0 / nz)
    )

    # Running DFT: six complex copies of the fields per frequency.
    dft_bytes = n_dft_freqs * field_values * 2 * b

    state_full_bytes = fields_bytes + psi_full_bytes + dft_bytes
    state_strip_bytes = fields_bytes + psi_strip_bytes + dft_bytes

    seg = float(n_steps) / k
    est = {
        "n_cells": n_cells,
        "bytes_per_value": b,
        "n_steps": int(n_steps),
        "checkpoint_segments": k,
        "fields_gb": fields_bytes / _GIB,
        "materials_gb": materials_bytes / _GIB,
        "psi_full_gb": psi_full_bytes / _GIB,
        "psi_strip_gb": psi_strip_bytes / _GIB,
        "dft_monitor_gb": dft_bytes / _GIB,
        "state_full_gb": state_full_bytes / _GIB,
        "state_strip_gb": state_strip_bytes / _GIB,
        "forward_gb": (state_full_bytes + materials_bytes) / _GIB,
        "adjoint_naive_gb": state_full_bytes * n_steps / _GIB,
        # sqrt-N checkpointing: K carry snapshots + the replayed segment of
        # seg steps, both at full carry size (note 08, section 1.1).
        "adjoint_checkpoint_gb": state_full_bytes * (k + seg) / _GIB,
        "adjoint_checkpoint_strip_gb": state_strip_bytes * (k + seg) / _GIB,
    }
    return est


def fits_gpu(required_gb: float, vram_gb: float, usable_fraction: float = 0.85) -> bool:
    """True if ``required_gb`` fits in ``vram_gb`` with XLA headroom.

    ``usable_fraction`` accounts for the CUDA context, XLA temporaries and
    fragmentation (0.85 is a conservative default for a dedicated GPU).
    """
    if not 0.0 < usable_fraction <= 1.0:
        raise ValueError("usable_fraction must be in (0, 1]")
    return required_gb <= usable_fraction * vram_gb


def gpu_fit_report(
    estimate: dict,
    *,
    key: str = "adjoint_checkpoint_gb",
    vram_sizes_gb=COMMON_VRAM_GB,
    usable_fraction: float = 0.85,
) -> dict[float, bool]:
    """Fit verdict of an estimate against common GPU VRAM sizes.

    Args:
        estimate: output of :func:`fdtd3d_memory_estimate`.
        key: which figure to judge (default: the checkpointed adjoint peak
            with the conservative full-grid psi layout).
        vram_sizes_gb: VRAM sizes to test (default 24 / 16 / 8 GB).
        usable_fraction: headroom factor passed to :func:`fits_gpu`.

    Returns:
        {vram_gb: fits} mapping, e.g. {24.0: True, 16.0: False, 8.0: False}.
    """
    required = float(estimate[key])
    return {
        float(v): fits_gpu(required, float(v), usable_fraction)
        for v in vram_sizes_gb
    }
