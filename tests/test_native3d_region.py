"""Acceptance tests for the native (Rust) design-region-limited 3D DFT.

The native kernel accumulates the running DFT only on per-component slabs
(:class:`gradenna.dft_region.DFTRegions`), returning a
:class:`gradenna.dft_region.RegionDFTMonitor`. Mirroring the XLA acceptance
suite (``tests/test_dft_region3d.py``) and the native suite
(``tests/test_native3d.py``), these tests assert:

1. native region-limited DFT vs native full-grid DFT on the covered cells
   (bit-identical / <= 1e-15 in f64; the native f32 threshold in f32);
2. native region-limited DFT vs the XLA region-limited DFT (f64 rel <= 1e-12);
3. ``freq_adjoint_gradient_3d(backend="native", objective_kind="port")``
   matches ``backend="xla", objective_kind="port"`` (cos >= 1 - 1e-10);
4. an off (``None``) component slab yields a ``None`` native spectrum.

Skipped wholesale when the native kernel cannot be built/loaded (no cargo on
the runner) so CI stays green.

Run::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/test_native3d_region.py -q
"""

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from gradenna import native3d  # noqa: E402
from gradenna.cpml import CPMLSpec  # noqa: E402
from gradenna.dft_region import (  # noqa: E402
    design_region_to_slabs,
    full_field_shapes,
    ntff_box_regions,
)
from gradenna.fdtd3d import simulate_3d  # noqa: E402
from gradenna.grid import Grid3D  # noqa: E402
from gradenna.native3d import simulate_3d_native  # noqa: E402

pytestmark = pytest.mark.skipif(
    not native3d.is_available(),
    reason="native FDTD kernel unavailable (cargo missing or build failed)",
)

# Same load-bearing thresholds as test_native3d.py for native-vs-XLA DFT.
RTOL = {np.float64: 1e-12, np.float32: 1e-5}

GRID = Grid3D(nx=22, ny=24, nz=20, dx=2e-3, dy=2.1e-3, dz=1.9e-3)
NPML = 6
F0 = 4e9
N_STEPS = 320
RAMP = 120
DR = (slice(8, 12), slice(9, 13), slice(7, 11))


def _cw(grid, n_steps=N_STEPS, ramp=RAMP, f0=F0):
    n = np.arange(n_steps)
    env = np.clip(n / float(ramp), 0.0, 1.0)
    return env * np.sin(2.0 * np.pi * f0 * (n + 0.5) * grid.dt)


def _rel(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(a)) + 1e-300))


# ---------------------------------------------------------------------------
# (1) native region-limited DFT == native full-grid DFT on the covered cells
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_native_region_matches_native_full(dtype):
    """Each native slab spectrum equals the corresponding full-grid slice."""
    cur = _cw(GRID).astype(dtype)
    common = dict(
        source_ijk=(9, 12, 10),
        source_current=cur,
        source_x_ijk=(7, 14, 12),
        source_x_current=(0.6 * cur).astype(dtype),
        dft_freqs=(F0,),
        cpml=CPMLSpec(thickness=NPML),
        dtype=dtype,
    )
    # Cover all six components so every slab branch is exercised.
    full_shapes = full_field_shapes(GRID.nx, GRID.ny, GRID.nz)
    regions = ntff_box_regions(GRID, NPML + 1, full_shapes)

    full = simulate_3d_native(GRID, **common)
    lim = simulate_3d_native(GRID, dft_regions=regions, **common)

    full_comps = (full.dft.ex, full.dft.ey, full.dft.ez,
                  full.dft.hx, full.dft.hy, full.dft.hz)
    lim_comps = (lim.dft.ex, lim.dft.ey, lim.dft.ez,
                 lim.dft.hx, lim.dft.hy, lim.dft.hz)
    for slab, full_arr, lim_arr in zip(regions, full_comps, lim_comps):
        assert slab is not None and lim_arr is not None
        sub = np.asarray(full_arr)[(slice(None),) + slab.slices()]
        if dtype == np.float64:
            # Same accumulation order on the same cells -> bit-identical.
            np.testing.assert_array_equal(np.asarray(lim_arr), sub)
        else:
            assert _rel(sub, lim_arr) <= 1e-15


# ---------------------------------------------------------------------------
# (2) native region-limited DFT == XLA region-limited DFT
# ---------------------------------------------------------------------------


def test_native_region_matches_xla_region():
    """native vs XLA region-limited slab spectra agree to f64 rel <= 1e-12."""
    dtype = np.float64
    cur = _cw(GRID)
    full_shapes = full_field_shapes(GRID.nx, GRID.ny, GRID.nz)
    regions = ntff_box_regions(GRID, NPML + 1, full_shapes)
    common = dict(
        source_ijk=(9, 12, 10),
        source_x_ijk=(7, 14, 12),
        source_x_current=0.6 * cur,
        dft_freqs=(F0,),
        cpml=CPMLSpec(thickness=NPML),
        dft_regions=regions,
    )
    r_xla = simulate_3d(
        GRID, source_current=jnp.asarray(cur, dtype),
        dft_dtype=jnp.complex128, **common,
    )
    r_nat = simulate_3d_native(GRID, source_current=cur, dtype=dtype, **common)

    for nm in ("ex", "ey", "ez", "hx", "hy", "hz"):
        a = getattr(r_xla.dft, nm)
        b = getattr(r_nat.dft, nm)
        assert (a is None) == (b is None), nm
        if a is not None:
            assert _rel(a, b) <= RTOL[dtype], "dft_" + nm


# ---------------------------------------------------------------------------
# (3) freq_adjoint port-objective gradient: native == XLA
# ---------------------------------------------------------------------------


def test_freq_adjoint_native_region_matches_xla():
    """objective_kind='port' gradient agrees across backends (cos >= 1-1e-10)."""
    from gradenna.freq_adjoint import freq_adjoint_gradient_3d
    from gradenna.sparams import s11_power_wave

    grid = Grid3D(24, 24, 24, 2e-3, 2e-3, 2e-3)
    f0 = 3e9
    n_steps = 2400
    ramp = 250
    n = jnp.arange(n_steps)
    env = jnp.clip(n / float(ramp), 0.0, 1.0)
    vs = env * jnp.sin(2.0 * jnp.pi * f0 * (n + 0.5) * grid.dt)
    region = (slice(10, 14), slice(10, 14), slice(10, 14))
    rs = 50.0

    def obj(ph):
        s = s11_power_wave(ph.port_v[0, 0], ph.port_i[0, 0], rs)
        return s.real**2 + s.imag**2

    sig0 = jnp.full((4, 4, 4), 0.05)
    common = dict(
        design_sigma=sig0, design_region=region, dft_freqs=(f0,), env=env,
        port_ijk=(12, 12, 12), port_voltage=vs, port_resistance=rs,
        objective_kind="port", cpml=CPMLSpec(thickness=6),
    )
    g_xla = freq_adjoint_gradient_3d(grid, obj, backend="xla", **common)["sigma"]
    g_nat = freq_adjoint_gradient_3d(grid, obj, backend="native", **common)["sigma"]
    a = np.asarray(g_xla).ravel()
    b = np.asarray(g_nat).ravel()
    assert np.all(np.isfinite(b))
    rel = np.max(np.abs(a - b)) / (np.max(np.abs(a)) + 1e-300)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos >= 1 - 1e-10, f"cos {cos:.15f}"
    assert rel <= 1e-6, f"rel {rel:.2e}"


# ---------------------------------------------------------------------------
# (4) None component slabs -> None native spectra
# ---------------------------------------------------------------------------


def test_native_none_components_are_none():
    """Design-region E3-only slabs leave Hx/Hy/Hz None in the native result."""
    full_shapes = full_field_shapes(GRID.nx, GRID.ny, GRID.nz)
    regions = design_region_to_slabs(DR, full_shapes)
    assert regions.hx is None and regions.hy is None and regions.hz is None
    res = simulate_3d_native(
        GRID,
        source_ijk=(9, 12, 10),
        source_current=_cw(GRID),
        dft_freqs=(F0,),
        dft_regions=regions,
        cpml=CPMLSpec(thickness=NPML),
        dtype=np.float64,
    )
    assert res.dft.ex is not None
    assert res.dft.ey is not None
    assert res.dft.ez is not None
    assert res.dft.hx is None
    assert res.dft.hy is None
    assert res.dft.hz is None
