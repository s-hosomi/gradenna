"""Differentiable 3D FDTD with CPML, a lumped RVS port and running DFT monitors.

Yee staggering on a uniform grid (PEC outer shell — tangential E on the six
boundary faces is held at zero):

    Ex(i+1/2, j,     k    )  shape (nx-1, ny,   nz  )
    Ey(i,     j+1/2, k    )  shape (nx,   ny-1, nz  )
    Ez(i,     j,     k+1/2)  shape (nx,   ny,   nz-1)
    Hx(i,     j+1/2, k+1/2)  shape (nx,   ny-1, nz-1)
    Hy(i+1/2, j,     k+1/2)  shape (nx-1, ny,   nz-1)
    Hz(i+1/2, j+1/2, k    )  shape (nx-1, ny-1, nz  )

The whole time loop is one `jax.lax.scan` (or, with
``checkpoint_segments=K``, an outer scan over K segments whose inner scan is
wrapped in `jax.checkpoint`, giving O(sqrt(N))-style memory at the cost of
one extra forward sweep), so any scalar loss of the outputs can be
differentiated with `jax.grad` with respect to the material arrays
(eps_r, sigma) and the source waveforms.

Materials are defined per cell on an (nx, ny, nz) lattice; the value at
node (i, j, k) is applied unchanged to the three E edges emanating from it
(Ex(i+1/2,j,k), Ey(i,j+1/2,k), Ez(i,j,k+1/2)). A consequence: a 1-cell
conductor sheet also shorts the vertical Ez edges directly above it (a pin
layer); see tests/test_patch_antenna.py for the details and the
compensation.

Lumped resistive voltage source (RVS; Piket-May, Taflove & Baron 1994,
research note 12) on a single z-directed Ez edge, semi-implicit in Ez:

    Ez^{n+1} = (1 - h - beta)/(1 + h + beta) Ez^n
               + (dt/eps)/(1 + h + beta) (curl H)_z^{n+1/2}
               - dt Vs^{n+1/2} / (Rs eps dx dy (1 + h + beta))

    beta = dt dz / (2 Rs eps dx dy),   h = sigma dt / (2 eps)

with port voltage (the same semi-implicit average as the 2D solver, so V
and I are time-aligned at t = (n+1/2) dt)

    V^{n+1/2} = -dz (Ez^n + Ez^{n+1}) / 2

and Ampere-loop current

    I^{n+1/2} = (Hy|i+1/2 - Hy|i-1/2) dy + (Hx|j-1/2 - Hx|j+1/2) dx

(the smallest Yee loop around the port edge; equals (curl H)_z dx dy, i.e.
the total — conduction plus displacement — current through the port face).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gradenna.constants import EPS0, MU0
from gradenna.cpml import (
    CPMLSpec,
    PsiSlabs,
    axis_coefficients,
    psi_step,
    slab_coefficients,
    slab_slices,
)
from gradenna.grid import Grid3D  # noqa: F401  (re-exported for backward compat)


class DFTMonitor(NamedTuple):
    """Running DFT of all six field components over the whole grid.

    freqs: (n_freqs,) evaluation frequencies [Hz].
    ex..hz: complex (n_freqs, *field_shape) spectra,
        X̂(f) = Δt Σ_n x^n exp(-i 2π f t_n), with the exact sample times
        t_n = (n+1) Δt for E components and (n+1/2) Δt for H components.
    """

    freqs: jnp.ndarray
    ex: jnp.ndarray
    ey: jnp.ndarray
    ez: jnp.ndarray
    hx: jnp.ndarray
    hy: jnp.ndarray
    hz: jnp.ndarray


class SimResult3D(NamedTuple):
    """Time series, monitors and final fields of a 3D simulation.

    probe_ez: (n_steps, n_probes) Ez at the probe edges; row n is time (n+1) dt.
    port_v:   (n_steps,) port voltage V = -dz (Ez^n + Ez^{n+1})/2, time-aligned
        at t = (n+1/2) dt (same convention as the 2D solver), or None.
    port_i:   (n_steps,) Ampere-loop port current at t = (n+1/2) dt, or None.
    energy:   (n_steps,) total field energy [J], or None unless requested.
    dft:      DFTMonitor with full-grid spectra, or None unless requested.
    ex..hz:   final field snapshots.
    """

    probe_ez: jnp.ndarray
    port_v: jnp.ndarray | None
    port_i: jnp.ndarray | None
    energy: jnp.ndarray | None
    dft: DFTMonitor | None
    ex: jnp.ndarray
    ey: jnp.ndarray
    ez: jnp.ndarray
    hx: jnp.ndarray
    hy: jnp.ndarray
    hz: jnp.ndarray


class _Psi3D(NamedTuple):
    """The 12 CPML psi variables, each as a low/high PML slab pair.

    Strip (slab) storage of note 14 Sec. 5.3: psi_{F,w} is non-zero only
    where the stretched axis w lies inside the PML, so each variable is two
    slabs of `npml` samples along w, spanning the full transverse extent
    (corners included). E-type psi live on the PEC interior, H-type on the
    half grid. With full arrays the psi set adds ~200% to the 3D scan carry
    — and reverse-mode AD puts the whole carry on the tape every step.
    """

    exy: PsiSlabs  # (nx-1, npml, nz-2) x2
    exz: PsiSlabs  # (nx-1, ny-2, npml) x2
    eyx: PsiSlabs  # (npml, ny-1, nz-2) x2
    eyz: PsiSlabs  # (nx-2, ny-1, npml) x2
    ezx: PsiSlabs  # (npml, ny-2, nz-1) x2
    ezy: PsiSlabs  # (nx-2, npml, nz-1) x2
    hxy: PsiSlabs  # (nx, npml, nz-1) x2
    hxz: PsiSlabs  # (nx, ny-1, npml) x2
    hyx: PsiSlabs  # (npml, ny, nz-1) x2
    hyz: PsiSlabs  # (nx-1, ny, npml) x2
    hzx: PsiSlabs  # (npml, ny-1, nz) x2
    hzy: PsiSlabs  # (nx-1, npml, nz) x2


def _init_psi(nx: int, ny: int, nz: int, npml: int, dtype) -> _Psi3D:
    """Zero-initialized slab-stored psi state (note 14 Sec. 5.3)."""
    pair = lambda shape: PsiSlabs(jnp.zeros(shape, dtype), jnp.zeros(shape, dtype))  # noqa: E731
    return _Psi3D(
        exy=pair((nx - 1, npml, nz - 2)),
        exz=pair((nx - 1, ny - 2, npml)),
        eyx=pair((npml, ny - 1, nz - 2)),
        eyz=pair((nx - 2, ny - 1, npml)),
        ezx=pair((npml, ny - 2, nz - 1)),
        ezy=pair((nx - 2, npml, nz - 1)),
        hxy=pair((nx, npml, nz - 1)),
        hxz=pair((nx, ny - 1, npml)),
        hyx=pair((npml, ny, nz - 1)),
        hyz=pair((nx - 1, ny, npml)),
        hzx=pair((npml, ny - 1, nz)),
        hzy=pair((nx - 1, npml, nz)),
    )


class _State3D(NamedTuple):
    ex: jnp.ndarray
    ey: jnp.ndarray
    ez: jnp.ndarray
    hx: jnp.ndarray
    hy: jnp.ndarray
    hz: jnp.ndarray
    # CPML psi variables in slab storage (note 14 Sec. 5.3).
    psi: _Psi3D
    # Running DFT accumulators (tuple of six complex arrays) or None.
    dft: tuple | None


def field_energy_3d(ex, ey, ez, hx, hy, hz, eps, grid: Grid3D):
    """Total electromagnetic energy, 1/2 (eps E^2 + mu H^2) dV.

    `eps` is the absolute permittivity on the (nx, ny, nz) cell lattice.
    Diagnostic only: E and H live at staggered times/locations, so this is
    conserved up to a bounded O((w dt)^2) oscillation in a lossless box.
    """
    cell = grid.dx * grid.dy * grid.dz
    ue = 0.5 * cell * (
        jnp.sum(eps[:-1, :, :] * ex**2)
        + jnp.sum(eps[:, :-1, :] * ey**2)
        + jnp.sum(eps[:, :, :-1] * ez**2)
    )
    uh = 0.5 * MU0 * cell * (jnp.sum(hx**2) + jnp.sum(hy**2) + jnp.sum(hz**2))
    return ue + uh


def time_series_dft(x, dt: float, freqs, *, t0: float = 0.0):
    """Exact-phase DFT of a sampled time series: X̂(f) = Δt Σ_n x_n e^{-i2πf(t0+nΔt)}.

    Use t0 = dt for E-type records (probe_ez) and t0 = dt/2 for half-step
    records (port_v, port_i, source/port waveforms), matching the sample
    times in SimResult3D.
    """
    x = jnp.asarray(x)
    freqs = jnp.atleast_1d(jnp.asarray(freqs))
    t = t0 + dt * jnp.arange(x.shape[0])
    phase = jnp.exp(-2j * jnp.pi * freqs[:, None] * t[None, :])
    return dt * jnp.tensordot(phase, x, axes=1)


def port_impedance(
    result: SimResult3D,
    grid: Grid3D,
    freqs,
    *,
    deembed_gap: bool = True,
    eps_r_port: float = 1.0,
):
    """Port input impedance Z(f) from the recorded V/I time series.

    V and I are both recorded at t = (n+1/2) dt (V is the semi-implicit
    average of Ez^n and Ez^{n+1}), so a single exact-phase DFT kernel
    applies to both. The Ampere loop measures the total current through
    the port face, which includes the displacement current of the 1-cell
    gap itself (research note 12, sections 2.1 and 6.1) — a shunt
    susceptance across the port. DFT'ing the discrete Maxwell-Ampere
    update of the port edge term by term gives the exact identity (lossless
    port cell, fields zero at the start and decayed at the end of the run)

        Î + j ω̄ C_gap V̂ = (V̂s - V̂) / Rs,
        C_gap = eps dx dy / dz,   ω̄ = 2 tan(ω dt/2) / dt,

    i.e. the loop current is the Thevenin branch current minus
    j ω̄ C_gap V̂ (the gap displacement current; with V = -Ez dz it enters
    with a minus sign), so the branch current is restored by *adding*
    + j ω̄ C_gap to the measured admittance. With ``deembed_gap=True``
    (default) this removes the exact discrete gap susceptance in parallel:

        Z = 1 / ( Î/V̂ + j ω̄ C_gap ).

    With ``deembed_gap=False`` the raw V̂/Î is returned; its real part is
    distorted by the gap shunt (severely so for electrically small loads).
    """
    if result.port_v is None or result.port_i is None:
        raise ValueError("result has no port records")
    dt = grid.dt
    freqs = jnp.atleast_1d(jnp.asarray(freqs))
    v_hat = time_series_dft(result.port_v, dt, freqs, t0=0.5 * dt)
    i_hat = time_series_dft(result.port_i, dt, freqs, t0=0.5 * dt)
    if not deembed_gap:
        return v_hat / i_hat
    omega_d = 2.0 * jnp.tan(jnp.pi * freqs * dt) / dt
    c_gap = EPS0 * eps_r_port * grid.dx * grid.dy / grid.dz
    return 1.0 / (i_hat / v_hat + 1j * omega_d * c_gap)


def _as_index_array_3d(ijk, name: str, grid: Grid3D, margin: int) -> np.ndarray:
    """Validate Ez-edge indices (i, j, k+1/2): i,j on nodes, k in [0, nz-2]."""
    idx = np.asarray(ijk, dtype=np.int32).reshape(-1, 3)
    lo = np.array([margin, margin, 0])
    hi = np.array([grid.nx - 1 - margin, grid.ny - 1 - margin, grid.nz - 2])
    if idx.size and (np.any(idx < lo) or np.any(idx > hi)):
        raise ValueError(
            f"{name} {idx.tolist()} outside the valid Ez range (margin {margin})"
        )
    return idx


def simulate_3d(
    grid: Grid3D,
    *,
    eps_r=1.0,
    sigma=0.0,
    source_ijk=None,
    source_current=None,
    source_x_ijk=None,
    source_x_current=None,
    source_y_ijk=None,
    source_y_current=None,
    mx_ijk=None,
    mx_current=None,
    my_ijk=None,
    my_current=None,
    mz_ijk=None,
    mz_current=None,
    port_ijk=None,
    port_voltage=None,
    port_resistance: float = 50.0,
    probe_ijk=(),
    cpml: CPMLSpec = CPMLSpec(),
    dft_freqs=None,
    dft_dtype=None,
    dft_regions=None,
    record_energy: bool = False,
    checkpoint_segments: int | None = None,
) -> SimResult3D:
    """Run a 3D FDTD simulation.

    Args:
        grid: the Yee grid.
        eps_r: relative permittivity per cell — scalar or (nx, ny, nz).
        sigma: electric conductivity [S/m] per cell — scalar or (nx, ny, nz).
        source_ijk: z-directed point-current Ez edge(s) — (i, j, k) or
            (n_sources, 3). The current I(t) is injected as Jz = I/(dx dy)
            over one cell (an infinitesimal dipole of length dz).
        source_current: currents I(t) [A] sampled at t = (n+1/2) dt; shape
            (n_steps,) or (n_steps, n_sources).
        source_x_ijk, source_x_current: x-directed point currents on the
            Ex(i+1/2, j, k) edges, injected as Jx = I/(dy dz). The transverse
            area is dy*dz (not dx*dy as for Jz) — the cross-section of the
            Ex curl loop (note 16 Sec. 2.3, pitfall 1). Backward compatible
            (no effect when omitted).
        source_y_ijk, source_y_current: y-directed point currents on the
            Ey(i, j+1/2, k) edges, injected as Jy = I/(dx dz).
        mx_ijk, mx_current: magnetic currents on the Hx(i, j+1/2, k+1/2)
            edges, sampled at t = (n+1/2) dt, injected as Hx -= (dt/mu) Mx
            (the mu dH/dt = -curl E - M term). Used by the 3D frequency-domain
            adjoint (:mod:`gradenna.freq_adjoint`); backward compatible.
        my_ijk, my_current: same for Hy(i+1/2, j, k+1/2), Hy -= (dt/mu) My.
        mz_ijk, mz_current: same for Hz(i+1/2, j+1/2, k), Hz -= (dt/mu) Mz.
        port_ijk: single z-directed lumped RVS port Ez edge (i, j, k), 1-cell
            gap, or None.
        port_voltage: Thevenin source voltage Vs(t) [V] sampled at
            t = (n+1/2) dt, shape (n_steps,). Positive Vs drives positive
            port voltage V = -Ez dz.
        port_resistance: Thevenin source resistance Rs [ohm].
        probe_ijk: Ez edges to record — sequence of (i, j, k) or (n, 3).
        cpml: CPML parameters; thickness 0 gives a plain PEC box.
        dft_freqs: frequencies [Hz] for a full-grid running DFT of all six
            components (memory: n_freqs x grid complex per component).
        dft_dtype: complex dtype of the running-DFT accumulators. ``None``
            (default) follows the field dtype (complex64 in a float32 run).
            Pass an explicit complex type (e.g. ``jnp.complex128``) to keep the
            DFT phasors and exact-phase kernel in higher precision while the
            fields stay float32 — the float32 underflow rescue for power
            monitors through strongly attenuating media (see the 2D solver's
            note). Only the DFT carry and the returned spectra change dtype;
            the fields, CPML psi slabs and checkpointing are untouched.
        dft_regions: optional :class:`gradenna.dft_region.DFTRegions` limiting
            the running DFT to a static per-component slab (``None`` entries
            are not accumulated at all, so they never land on the scan carry
            or the reverse tape). ``None`` (default) records the full grid and
            returns a :class:`DFTMonitor`; when given, ``SimResult3D.dft`` is a
            :class:`gradenna.dft_region.RegionDFTMonitor` with the same
            dt-scale and phase conventions. Memory-bounding hook for the
            frequency-domain adjoint (:mod:`gradenna.freq_adjoint`).
        record_energy: also record the total field energy at every step.
        checkpoint_segments: split the time loop into K segments and wrap the
            inner scan in `jax.checkpoint` (sqrt-N checkpointing). Must divide
            n_steps. None runs a single flat scan.

    Differentiable in eps_r, sigma, source_current and port_voltage.
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    if min(nx, ny, nz) <= 2 * cpml.thickness + 2:
        raise ValueError(
            f"grid {nx}x{ny}x{nz} is too small for CPML thickness {cpml.thickness}"
        )
    dt = grid.dt
    dx, dy, dz = grid.dx, grid.dy, grid.dz

    has_src = source_ijk is not None
    has_port = port_ijk is not None
    if has_src != (source_current is not None):
        raise ValueError("source_ijk and source_current must be given together")
    if has_port != (port_voltage is not None):
        raise ValueError("port_ijk and port_voltage must be given together")

    # Additional electric- and magnetic-current channels (Jx/Jy and Mx/My/Mz),
    # all optional and backward compatible. Each is an (index array, waveform)
    # pair; an absent channel resolves to ((0, 3) index, None waveform).
    def _check_pair(idx, cur, name):
        if (idx is None) != (cur is None):
            raise ValueError(f"{name}_ijk and {name}_current must be given together")
        return idx is not None

    has_jx = _check_pair(source_x_ijk, source_x_current, "source_x")
    has_jy = _check_pair(source_y_ijk, source_y_current, "source_y")
    has_mx = _check_pair(mx_ijk, mx_current, "mx")
    has_my = _check_pair(my_ijk, my_current, "my")
    has_mz = _check_pair(mz_ijk, mz_current, "mz")
    has_extra = has_jx or has_jy or has_mx or has_my or has_mz

    if not (has_src or has_port or has_extra):
        raise ValueError("at least one current/voltage source is required")

    probe_idx = _as_index_array_3d(probe_ijk, "probe_ijk", grid, margin=0)

    n_steps = None
    if has_src:
        source_current = jnp.asarray(source_current)
        if source_current.ndim == 1:
            source_current = source_current[:, None]
        n_steps = source_current.shape[0]
        src_idx = _as_index_array_3d(source_ijk, "source_ijk", grid, margin=1)
        if source_current.shape[1] != src_idx.shape[0]:
            raise ValueError(
                f"source_current has {source_current.shape[1]} columns "
                f"for {src_idx.shape[0]} sources"
            )
    if has_port:
        port_voltage = jnp.asarray(port_voltage).reshape(-1)
        if n_steps is not None and port_voltage.shape[0] != n_steps:
            raise ValueError("source_current and port_voltage lengths differ")
        n_steps = port_voltage.shape[0]
        (pi, pj, pk), = _as_index_array_3d(port_ijk, "port_ijk", grid, margin=1)
        if port_resistance <= 0.0:
            raise ValueError("port_resistance must be positive")

    # Resolve the extra current/magnetic-current channels to (idx, waveform).
    def _setup_channel(idx_in, cur_in, present, shape, name):
        nonlocal n_steps
        if not present:
            return np.zeros((0, 3), np.int32), None
        idx = np.asarray(idx_in, np.int32).reshape(-1, 3)
        hi = np.array([shape[0] - 1, shape[1] - 1, shape[2] - 1])
        if idx.size and (np.any(idx < 0) or np.any(idx > hi)):
            raise ValueError(
                f"{name} {idx.tolist()} outside the {name} grid {tuple(shape)}"
            )
        cur = jnp.asarray(cur_in)
        if cur.ndim == 1:
            cur = cur[:, None]
        if n_steps is None:
            n_steps = cur.shape[0]
        elif cur.shape[0] != n_steps:
            raise ValueError(
                f"channel waveform length {cur.shape[0]} does not match n_steps {n_steps}"
            )
        if cur.shape[1] != idx.shape[0]:
            raise ValueError(
                f"channel waveform has {cur.shape[1]} columns for {idx.shape[0]} sources"
            )
        return idx, cur

    # Per-component field shapes the channel indices address: Jx->Ex, Jy->Ey,
    # Mx->Hx, My->Hy, Mz->Hz (Yee staggering).
    jx_idx, jx_cur = _setup_channel(
        source_x_ijk, source_x_current, has_jx, (nx - 1, ny, nz), "source_x_ijk"
    )
    jy_idx, jy_cur = _setup_channel(
        source_y_ijk, source_y_current, has_jy, (nx, ny - 1, nz), "source_y_ijk"
    )
    mx_idx, mx_cur = _setup_channel(
        mx_ijk, mx_current, has_mx, (nx, ny - 1, nz - 1), "mx_ijk"
    )
    my_idx, my_cur = _setup_channel(
        my_ijk, my_current, has_my, (nx - 1, ny, nz - 1), "my_ijk"
    )
    mz_idx, mz_cur = _setup_channel(
        mz_ijk, mz_current, has_mz, (nx - 1, ny - 1, nz), "mz_ijk"
    )

    parts = [eps_r, sigma]
    if has_src:
        parts.append(source_current)
    if has_port:
        parts.append(port_voltage)
    for c in (jx_cur, jy_cur, mx_cur, my_cur, mz_cur):
        if c is not None:
            parts.append(c)
    dtype = jnp.result_type(*parts)
    if has_src:
        source_current = source_current.astype(dtype)
    if has_port:
        port_voltage = port_voltage.astype(dtype)
    jx_cur = None if jx_cur is None else jx_cur.astype(dtype)
    jy_cur = None if jy_cur is None else jy_cur.astype(dtype)
    mx_cur = None if mx_cur is None else mx_cur.astype(dtype)
    my_cur = None if my_cur is None else my_cur.astype(dtype)
    mz_cur = None if mz_cur is None else mz_cur.astype(dtype)

    eps = EPS0 * jnp.broadcast_to(jnp.asarray(eps_r, dtype), (nx, ny, nz))
    sig = jnp.broadcast_to(jnp.asarray(sigma, dtype), (nx, ny, nz))

    half_loss = sig * dt / (2.0 * eps)
    ca = (1.0 - half_loss) / (1.0 + half_loss)
    cb = (dt / eps) / (1.0 + half_loss)
    # Interior slices at the three E-component positions (cell (i,j,k) maps
    # to the E edges emanating from node (i,j,k)).
    ca_ex, cb_ex = ca[:-1, 1:-1, 1:-1], cb[:-1, 1:-1, 1:-1]
    ca_ey, cb_ey = ca[1:-1, :-1, 1:-1], cb[1:-1, :-1, 1:-1]
    ca_ez, cb_ez = ca[1:-1, 1:-1, :-1], cb[1:-1, 1:-1, :-1]

    # CPML tables: integer positions for the E updates (sliced to the PEC
    # interior), half positions for the H updates. psi is stored on the two
    # PML slabs per stretched axis (note 14 Sec. 5.3), so the b/c tables are
    # restricted to the slabs; 1/kappa stays full-size (== 1 outside the PML).
    ax_e = axis_coefficients(nx, dx, dt, cpml, half=False, dtype=dtype)
    ay_e = axis_coefficients(ny, dy, dt, cpml, half=False, dtype=dtype)
    az_e = axis_coefficients(nz, dz, dt, cpml, half=False, dtype=dtype)
    ax_h = axis_coefficients(nx, dx, dt, cpml, half=True, dtype=dtype)
    ay_h = axis_coefficients(ny, dy, dt, cpml, half=True, dtype=dtype)
    az_h = axis_coefficients(nz, dz, dt, cpml, half=True, dtype=dtype)
    kx_e = ax_e.inv_kappa[1:-1, None, None]
    ky_e = ay_e.inv_kappa[None, 1:-1, None]
    kz_e = az_e.inv_kappa[None, None, 1:-1]
    kx_h = ax_h.inv_kappa[:, None, None]
    ky_h = ay_h.inv_kappa[None, :, None]
    kz_h = az_h.inv_kappa[None, None, :]

    npml = cpml.thickness
    sx_e = slab_slices(nx - 2, npml)
    sy_e = slab_slices(ny - 2, npml)
    sz_e = slab_slices(nz - 2, npml)
    sx_h = slab_slices(nx - 1, npml)
    sy_h = slab_slices(ny - 1, npml)
    sz_h = slab_slices(nz - 1, npml)
    bcx_e = slab_coefficients(ax_e.b[1:-1], ax_e.c[1:-1], sx_e, axis=0, ndim=3)
    bcy_e = slab_coefficients(ay_e.b[1:-1], ay_e.c[1:-1], sy_e, axis=1, ndim=3)
    bcz_e = slab_coefficients(az_e.b[1:-1], az_e.c[1:-1], sz_e, axis=2, ndim=3)
    bcx_h = slab_coefficients(ax_h.b, ax_h.c, sx_h, axis=0, ndim=3)
    bcy_h = slab_coefficients(ay_h.b, ay_h.c, sy_h, axis=1, ndim=3)
    bcz_h = slab_coefficients(az_h.b, az_h.c, sz_h, axis=2, ndim=3)

    inv_dx, inv_dy, inv_dz = 1.0 / dx, 1.0 / dy, 1.0 / dz
    dt_mu = dt / MU0

    if has_src:
        # Discretized point current: Jz = I / (dx dy) over one cell.
        cb_src = cb[src_idx[:, 0], src_idx[:, 1], src_idx[:, 2]] * (inv_dx * inv_dy)
    # Jx/Jy point currents: each is divided by the cross-section of *its* curl
    # loop (note 16 Sec. 2.3, pitfall 1): Jx area dy*dz, Jy area dx*dz, Jz dx*dy.
    if has_jx:
        cb_jx = cb[jx_idx[:, 0], jx_idx[:, 1], jx_idx[:, 2]] * (inv_dy * inv_dz)
    if has_jy:
        cb_jy = cb[jy_idx[:, 0], jy_idx[:, 1], jy_idx[:, 2]] * (inv_dx * inv_dz)
    if has_port:
        # Semi-implicit RVS coefficients (research note 12, with conductivity).
        eps_p = eps[pi, pj, pk]
        h_p = sig[pi, pj, pk] * dt / (2.0 * eps_p)
        beta = dt * dz / (2.0 * port_resistance * eps_p * dx * dy)
        denom = 1.0 + h_p + beta
        a_port = (1.0 - h_p - beta) / denom
        b_port = (dt / eps_p) / denom
        c_port = -dt / (port_resistance * eps_p * dx * dy * denom)

    has_dft = dft_freqs is not None
    # Region-limited DFT: each component is accumulated only on a static slab
    # (or not at all when its slab is None), cutting the carry/tape footprint.
    has_regions = has_dft and dft_regions is not None
    region_slabs = None  # six entries: tuple(lo, hi) of Python ints, or None.
    if has_regions:
        region_slabs = tuple(
            None if slab is None else (tuple(slab.lo), tuple(slab.hi))
            for slab in dft_regions
        )
    if has_dft:
        # Stored as float64 regardless of the (possibly float32) field dtype,
        # so phase-sensitive consumers never inherit a low-precision frequency.
        freqs = jnp.atleast_1d(jnp.asarray(dft_freqs, jnp.float64))
        if dft_dtype is None:
            cdtype = jnp.result_type(dtype, jnp.complex64)
        else:
            cdtype = jnp.dtype(dft_dtype)
            if not jnp.issubdtype(cdtype, jnp.complexfloating):
                raise ValueError(f"dft_dtype must be a complex dtype, got {dft_dtype}")
        # Exact-phase tables, generated in float64 (note 12 Sec. 5.2: never
        # build the phasor recursively in low precision).
        f_np = np.atleast_1d(np.asarray(dft_freqs, np.float64))
        n_np = np.arange(n_steps, dtype=np.float64)
        # (n_steps, n_freqs), already scaled by dt.
        e_phase = jnp.asarray(
            dt * np.exp(-2j * np.pi * np.outer(n_np + 1.0, f_np) * dt), cdtype
        )
        h_phase = jnp.asarray(
            dt * np.exp(-2j * np.pi * np.outer(n_np + 0.5, f_np) * dt), cdtype
        )

    def step(state: _State3D, xs):
        ex, ey, ez = state.ex, state.ey, state.ez
        hx, hy, hz = state.hx, state.hy, state.hz
        psi = state.psi

        # --- H update: E^n -> H^{n+1/2} ---------------------------------
        dey_dz = (ey[:, :, 1:] - ey[:, :, :-1]) * inv_dz  # (nx, ny-1, nz-1)
        dez_dy = (ez[:, 1:, :] - ez[:, :-1, :]) * inv_dy
        p_hxz, t_hxz = psi_step(psi.hxz, dey_dz, bcz_h, sz_h, 2, kz_h)
        p_hxy, t_hxy = psi_step(psi.hxy, dez_dy, bcy_h, sy_h, 1, ky_h)
        hx = hx + dt_mu * (t_hxz - t_hxy)
        if has_mx:
            # mu dHx/dt = -curl E - Mx  ->  Hx -= (dt/mu) Mx.
            hx = hx.at[mx_idx[:, 0], mx_idx[:, 1], mx_idx[:, 2]].add(-dt_mu * xs["mx"])

        dez_dx = (ez[1:, :, :] - ez[:-1, :, :]) * inv_dx  # (nx-1, ny, nz-1)
        dex_dz = (ex[:, :, 1:] - ex[:, :, :-1]) * inv_dz
        p_hyx, t_hyx = psi_step(psi.hyx, dez_dx, bcx_h, sx_h, 0, kx_h)
        p_hyz, t_hyz = psi_step(psi.hyz, dex_dz, bcz_h, sz_h, 2, kz_h)
        hy = hy + dt_mu * (t_hyx - t_hyz)
        if has_my:
            hy = hy.at[my_idx[:, 0], my_idx[:, 1], my_idx[:, 2]].add(-dt_mu * xs["my"])

        dex_dy = (ex[:, 1:, :] - ex[:, :-1, :]) * inv_dy  # (nx-1, ny-1, nz)
        dey_dx = (ey[1:, :, :] - ey[:-1, :, :]) * inv_dx
        p_hzy, t_hzy = psi_step(psi.hzy, dex_dy, bcy_h, sy_h, 1, ky_h)
        p_hzx, t_hzx = psi_step(psi.hzx, dey_dx, bcx_h, sx_h, 0, kx_h)
        hz = hz + dt_mu * (t_hzy - t_hzx)
        if has_mz:
            hz = hz.at[mz_idx[:, 0], mz_idx[:, 1], mz_idx[:, 2]].add(-dt_mu * xs["mz"])

        if has_port:
            # Ampere loop around the port edge at t = (n+1/2) dt.
            i_loop = (hy[pi, pj, pk] - hy[pi - 1, pj, pk]) * dy + (
                hx[pi, pj - 1, pk] - hx[pi, pj, pk]
            ) * dx

        # --- E update: H^{n+1/2} -> E^{n+1} (interior; PEC shell fixed) --
        dhz_dy = (hz[:, 1:, :] - hz[:, :-1, :])[:, :, 1:-1] * inv_dy  # (nx-1, ny-2, nz-2)
        dhy_dz = (hy[:, :, 1:] - hy[:, :, :-1])[:, 1:-1, :] * inv_dz
        p_exy, t_exy = psi_step(psi.exy, dhz_dy, bcy_e, sy_e, 1, ky_e)
        p_exz, t_exz = psi_step(psi.exz, dhy_dz, bcz_e, sz_e, 2, kz_e)
        curl_x = t_exy - t_exz
        ex = ex.at[:, 1:-1, 1:-1].set(ca_ex * ex[:, 1:-1, 1:-1] + cb_ex * curl_x)
        if has_jx:
            ex = ex.at[jx_idx[:, 0], jx_idx[:, 1], jx_idx[:, 2]].add(-cb_jx * xs["jx"])

        dhx_dz = (hx[:, :, 1:] - hx[:, :, :-1])[1:-1, :, :] * inv_dz  # (nx-2, ny-1, nz-2)
        dhz_dx = (hz[1:, :, :] - hz[:-1, :, :])[:, :, 1:-1] * inv_dx
        p_eyz, t_eyz = psi_step(psi.eyz, dhx_dz, bcz_e, sz_e, 2, kz_e)
        p_eyx, t_eyx = psi_step(psi.eyx, dhz_dx, bcx_e, sx_e, 0, kx_e)
        curl_y = t_eyz - t_eyx
        ey = ey.at[1:-1, :, 1:-1].set(ca_ey * ey[1:-1, :, 1:-1] + cb_ey * curl_y)
        if has_jy:
            ey = ey.at[jy_idx[:, 0], jy_idx[:, 1], jy_idx[:, 2]].add(-cb_jy * xs["jy"])

        dhy_dx = (hy[1:, :, :] - hy[:-1, :, :])[:, 1:-1, :] * inv_dx  # (nx-2, ny-2, nz-1)
        dhx_dy = (hx[:, 1:, :] - hx[:, :-1, :])[1:-1, :, :] * inv_dy
        p_ezx, t_ezx = psi_step(psi.ezx, dhy_dx, bcx_e, sx_e, 0, kx_e)
        p_ezy, t_ezy = psi_step(psi.ezy, dhx_dy, bcy_e, sy_e, 1, ky_e)
        curl_z = t_ezx - t_ezy
        ez_prev = ez
        ez = ez.at[1:-1, 1:-1, :].set(ca_ez * ez[1:-1, 1:-1, :] + cb_ez * curl_z)

        if has_src:
            ez = ez.at[src_idx[:, 0], src_idx[:, 1], src_idx[:, 2]].add(
                -cb_src * xs["i_src"]
            )
        if has_port:
            # Overwrite the port edge with the semi-implicit RVS update.
            ez = ez.at[pi, pj, pk].set(
                a_port * ez_prev[pi, pj, pk]
                + b_port * curl_z[pi - 1, pj - 1, pk]
                + c_port * xs["vs"]
            )

        dft_acc = state.dft
        if has_dft:
            eph = xs["e_phase"][:, None, None, None]
            hph = xs["h_phase"][:, None, None, None]
            if has_regions:
                # Accumulate each component only on its static slab (None
                # components carry no array, keeping them off carry/tape).
                fields = (ex, ey, ez, hx, hy, hz)
                phases = (eph, eph, eph, hph, hph, hph)
                new = []
                for acc, field, ph, sl in zip(dft_acc, fields, phases, region_slabs):
                    if acc is None:
                        new.append(None)
                        continue
                    (lo0, lo1, lo2), (hi0, hi1, hi2) = sl
                    patch = field[lo0:hi0, lo1:hi1, lo2:hi2]
                    new.append(acc + ph * patch[None])
                dft_acc = tuple(new)
            else:
                d_ex, d_ey, d_ez, d_hx, d_hy, d_hz = dft_acc
                dft_acc = (
                    d_ex + eph * ex[None],
                    d_ey + eph * ey[None],
                    d_ez + eph * ez[None],
                    d_hx + hph * hx[None],
                    d_hy + hph * hy[None],
                    d_hz + hph * hz[None],
                )

        state = _State3D(
            ex, ey, ez, hx, hy, hz,
            _Psi3D(
                p_exy, p_exz, p_eyx, p_eyz, p_ezx, p_ezy,
                p_hxy, p_hxz, p_hyx, p_hyz, p_hzx, p_hzy,
            ),
            dft_acc,
        )
        out = {"probe": ez[probe_idx[:, 0], probe_idx[:, 1], probe_idx[:, 2]]}
        if has_port:
            # V^{n+1/2} = -dz (Ez^n + Ez^{n+1})/2: time-aligned with I.
            out["v"] = -0.5 * dz * (ez_prev[pi, pj, pk] + ez[pi, pj, pk])
            out["i"] = i_loop
        if record_energy:
            out["energy"] = field_energy_3d(ex, ey, ez, hx, hy, hz, eps, grid)
        return state, out

    zeros = lambda shape: jnp.zeros(shape, dtype)  # noqa: E731
    dft0 = None
    if has_dft:
        n_freq = freqs.shape[0]
        if has_regions:
            # One zero array per accumulated slab; None components stay None
            # (no pytree leaf -> never on the scan carry / reverse tape).
            slab_shapes = []
            for sl in region_slabs:
                if sl is None:
                    slab_shapes.append(None)
                else:
                    (lo0, lo1, lo2), (hi0, hi1, hi2) = sl
                    slab_shapes.append((hi0 - lo0, hi1 - lo1, hi2 - lo2))
            dft0 = tuple(
                None if s is None else jnp.zeros((n_freq,) + s, cdtype)
                for s in slab_shapes
            )
        else:
            shapes = (
                (nx - 1, ny, nz), (nx, ny - 1, nz), (nx, ny, nz - 1),
                (nx, ny - 1, nz - 1), (nx - 1, ny, nz - 1), (nx - 1, ny - 1, nz),
            )
            dft0 = tuple(jnp.zeros((n_freq,) + s, cdtype) for s in shapes)
    state0 = _State3D(
        ex=zeros((nx - 1, ny, nz)),
        ey=zeros((nx, ny - 1, nz)),
        ez=zeros((nx, ny, nz - 1)),
        hx=zeros((nx, ny - 1, nz - 1)),
        hy=zeros((nx - 1, ny, nz - 1)),
        hz=zeros((nx - 1, ny - 1, nz)),
        psi=_init_psi(nx, ny, nz, npml, dtype),
        dft=dft0,
    )

    xs = {}
    if has_src:
        xs["i_src"] = source_current
    if has_port:
        xs["vs"] = port_voltage
    if has_jx:
        xs["jx"] = jx_cur
    if has_jy:
        xs["jy"] = jy_cur
    if has_mx:
        xs["mx"] = mx_cur
    if has_my:
        xs["my"] = my_cur
    if has_mz:
        xs["mz"] = mz_cur
    if has_dft:
        xs["e_phase"] = e_phase
        xs["h_phase"] = h_phase

    if checkpoint_segments is None:
        final, outputs = jax.lax.scan(step, state0, xs)
    else:
        k = int(checkpoint_segments)
        if k <= 0 or n_steps % k != 0:
            raise ValueError(
                f"checkpoint_segments={checkpoint_segments} must divide n_steps={n_steps}"
            )
        seg = n_steps // k
        xs = jax.tree.map(lambda a: a.reshape((k, seg) + a.shape[1:]), xs)
        inner = jax.checkpoint(lambda c, x: jax.lax.scan(step, c, x))
        final, outputs = jax.lax.scan(inner, state0, xs)
        outputs = jax.tree.map(
            lambda a: a.reshape((a.shape[0] * a.shape[1],) + a.shape[2:]), outputs
        )

    dft_out = None
    if has_regions:
        from gradenna.dft_region import RegionDFTMonitor

        dft_out = RegionDFTMonitor(freqs, dft_regions, *final.dft)
    elif has_dft:
        dft_out = DFTMonitor(freqs, *final.dft)
    return SimResult3D(
        probe_ez=outputs["probe"],
        port_v=outputs.get("v"),
        port_i=outputs.get("i"),
        energy=outputs.get("energy"),
        dft=dft_out,
        ex=final.ex,
        ey=final.ey,
        ez=final.ez,
        hx=final.hx,
        hy=final.hy,
        hz=final.hz,
    )
