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
(Ex(i+1/2,j,k), Ey(i,j+1/2,k), Ez(i,j,k+1/2)).

Lumped resistive voltage source (RVS; Piket-May, Taflove & Baron 1994,
research note 12) on a single z-directed Ez edge, semi-implicit in Ez:

    Ez^{n+1} = (1 - h - beta)/(1 + h + beta) Ez^n
               + (dt/eps)/(1 + h + beta) (curl H)_z^{n+1/2}
               - dt Vs^{n+1/2} / (Rs eps dx dy (1 + h + beta))

    beta = dt dz / (2 Rs eps dx dy),   h = sigma dt / (2 eps)

with port voltage V^n = -Ez^n dz and Ampere-loop current

    I^{n+1/2} = (Hy|i+1/2 - Hy|i-1/2) dy + (Hx|j-1/2 - Hx|j+1/2) dx

(the smallest Yee loop around the port edge; equals (curl H)_z dx dy, i.e.
the total — conduction plus displacement — current through the port face).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gradenna.constants import C0, EPS0, MU0
from gradenna.cpml import CPMLSpec, axis_coefficients


@dataclass(frozen=True)
class Grid3D:
    """Uniform 3D Yee grid (see module docstring for field locations)."""

    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    courant: float = 0.99  # fraction of the 3D stability limit

    def __post_init__(self) -> None:
        if min(self.nx, self.ny, self.nz) < 3:
            raise ValueError(
                f"grid must be at least 3x3x3, got {self.nx}x{self.ny}x{self.nz}"
            )
        if min(self.dx, self.dy, self.dz) <= 0.0 or self.courant <= 0.0:
            raise ValueError("dx, dy, dz and courant must be positive")

    @property
    def dt(self) -> float:
        """Time step Δt = S / (c √(1/Δx² + 1/Δy² + 1/Δz²)) with S = courant."""
        return self.courant / (
            C0 * math.sqrt(1.0 / self.dx**2 + 1.0 / self.dy**2 + 1.0 / self.dz**2)
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)


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
    port_v:   (n_steps,) port voltage V = -Ez dz at t = (n+1) dt, or None.
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


class _State3D(NamedTuple):
    ex: jnp.ndarray
    ey: jnp.ndarray
    ez: jnp.ndarray
    hx: jnp.ndarray
    hy: jnp.ndarray
    hz: jnp.ndarray
    # CPML psi for E updates (interior-sized, two transverse axes each).
    p_exy: jnp.ndarray
    p_exz: jnp.ndarray
    p_eyx: jnp.ndarray
    p_eyz: jnp.ndarray
    p_ezx: jnp.ndarray
    p_ezy: jnp.ndarray
    # CPML psi for H updates (full-sized, two transverse axes each).
    p_hxy: jnp.ndarray
    p_hxz: jnp.ndarray
    p_hyx: jnp.ndarray
    p_hyz: jnp.ndarray
    p_hzx: jnp.ndarray
    p_hzy: jnp.ndarray
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

    Use t0 = dt for E-type records (probe_ez, port_v) and t0 = dt/2 for
    H-type records (port_i), matching the sample times in SimResult3D.
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

    V and I are DFT'ed at their exact sample times ((n+1) dt and
    (n+1/2) dt), so the half-step phase correction is exact. The Ampere
    loop measures the total current through the port face, which includes
    the displacement current of the 1-cell gap itself (research note 12,
    sections 2.1 and 6.1) — a shunt susceptance across the port. With
    ``deembed_gap=True`` (default) the exact discrete gap susceptance is
    removed in parallel:

        Z = 1 / ( Î/V̂ + j ω̃ C_gap ),
        C_gap = eps dx dy / dz,   ω̃ = 2 sin(ω dt/2) / dt

    With ``deembed_gap=False`` the raw V̂/Î is returned; its real part is
    distorted by the gap shunt (severely so for electrically small loads).
    """
    if result.port_v is None or result.port_i is None:
        raise ValueError("result has no port records")
    dt = grid.dt
    freqs = jnp.atleast_1d(jnp.asarray(freqs))
    v_hat = time_series_dft(result.port_v, dt, freqs, t0=dt)
    i_hat = time_series_dft(result.port_i, dt, freqs, t0=0.5 * dt)
    if not deembed_gap:
        return v_hat / i_hat
    omega_d = 2.0 * jnp.sin(jnp.pi * freqs * dt) / dt
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
    port_ijk=None,
    port_voltage=None,
    port_resistance: float = 50.0,
    probe_ijk=(),
    cpml: CPMLSpec = CPMLSpec(),
    dft_freqs=None,
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
    if not (has_src or has_port):
        raise ValueError("at least one of source_current / port_voltage is required")

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

    parts = [eps_r, sigma]
    if has_src:
        parts.append(source_current)
    if has_port:
        parts.append(port_voltage)
    dtype = jnp.result_type(*parts)
    if has_src:
        source_current = source_current.astype(dtype)
    if has_port:
        port_voltage = port_voltage.astype(dtype)

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
    # interior), half positions for the H updates.
    bx_e, cx_e, kx_e = (
        a[1:-1, None, None] for a in axis_coefficients(nx, dx, dt, cpml, half=False, dtype=dtype)
    )
    by_e, cy_e, ky_e = (
        a[None, 1:-1, None] for a in axis_coefficients(ny, dy, dt, cpml, half=False, dtype=dtype)
    )
    bz_e, cz_e, kz_e = (
        a[None, None, 1:-1] for a in axis_coefficients(nz, dz, dt, cpml, half=False, dtype=dtype)
    )
    bx_h, cx_h, kx_h = (
        a[:, None, None] for a in axis_coefficients(nx, dx, dt, cpml, half=True, dtype=dtype)
    )
    by_h, cy_h, ky_h = (
        a[None, :, None] for a in axis_coefficients(ny, dy, dt, cpml, half=True, dtype=dtype)
    )
    bz_h, cz_h, kz_h = (
        a[None, None, :] for a in axis_coefficients(nz, dz, dt, cpml, half=True, dtype=dtype)
    )

    inv_dx, inv_dy, inv_dz = 1.0 / dx, 1.0 / dy, 1.0 / dz
    dt_mu = dt / MU0

    if has_src:
        # Discretized point current: Jz = I / (dx dy) over one cell.
        cb_src = cb[src_idx[:, 0], src_idx[:, 1], src_idx[:, 2]] * (inv_dx * inv_dy)
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
    if has_dft:
        freqs = jnp.atleast_1d(jnp.asarray(dft_freqs, dtype))
        cdtype = jnp.result_type(dtype, jnp.complex64)
        n = jnp.arange(n_steps, dtype=dtype)
        omega_t_e = 2.0 * jnp.pi * freqs[None, :] * ((n[:, None] + 1.0) * dt)
        omega_t_h = 2.0 * jnp.pi * freqs[None, :] * ((n[:, None] + 0.5) * dt)
        e_phase = (dt * jnp.exp(-1j * omega_t_e)).astype(cdtype)  # (n_steps, n_freqs)
        h_phase = (dt * jnp.exp(-1j * omega_t_h)).astype(cdtype)

    def step(state: _State3D, xs):
        ex, ey, ez = state.ex, state.ey, state.ez
        hx, hy, hz = state.hx, state.hy, state.hz

        # --- H update: E^n -> H^{n+1/2} ---------------------------------
        dey_dz = (ey[:, :, 1:] - ey[:, :, :-1]) * inv_dz  # (nx, ny-1, nz-1)
        dez_dy = (ez[:, 1:, :] - ez[:, :-1, :]) * inv_dy
        p_hxz = bz_h * state.p_hxz + cz_h * dey_dz
        p_hxy = by_h * state.p_hxy + cy_h * dez_dy
        hx = hx + dt_mu * (dey_dz * kz_h + p_hxz - dez_dy * ky_h - p_hxy)

        dez_dx = (ez[1:, :, :] - ez[:-1, :, :]) * inv_dx  # (nx-1, ny, nz-1)
        dex_dz = (ex[:, :, 1:] - ex[:, :, :-1]) * inv_dz
        p_hyx = bx_h * state.p_hyx + cx_h * dez_dx
        p_hyz = bz_h * state.p_hyz + cz_h * dex_dz
        hy = hy + dt_mu * (dez_dx * kx_h + p_hyx - dex_dz * kz_h - p_hyz)

        dex_dy = (ex[:, 1:, :] - ex[:, :-1, :]) * inv_dy  # (nx-1, ny-1, nz)
        dey_dx = (ey[1:, :, :] - ey[:-1, :, :]) * inv_dx
        p_hzy = by_h * state.p_hzy + cy_h * dex_dy
        p_hzx = bx_h * state.p_hzx + cx_h * dey_dx
        hz = hz + dt_mu * (dex_dy * ky_h + p_hzy - dey_dx * kx_h - p_hzx)

        if has_port:
            # Ampere loop around the port edge at t = (n+1/2) dt.
            i_loop = (hy[pi, pj, pk] - hy[pi - 1, pj, pk]) * dy + (
                hx[pi, pj - 1, pk] - hx[pi, pj, pk]
            ) * dx

        # --- E update: H^{n+1/2} -> E^{n+1} (interior; PEC shell fixed) --
        dhz_dy = (hz[:, 1:, :] - hz[:, :-1, :])[:, :, 1:-1] * inv_dy  # (nx-1, ny-2, nz-2)
        dhy_dz = (hy[:, :, 1:] - hy[:, :, :-1])[:, 1:-1, :] * inv_dz
        p_exy = by_e * state.p_exy + cy_e * dhz_dy
        p_exz = bz_e * state.p_exz + cz_e * dhy_dz
        curl_x = dhz_dy * ky_e + p_exy - dhy_dz * kz_e - p_exz
        ex = ex.at[:, 1:-1, 1:-1].set(ca_ex * ex[:, 1:-1, 1:-1] + cb_ex * curl_x)

        dhx_dz = (hx[:, :, 1:] - hx[:, :, :-1])[1:-1, :, :] * inv_dz  # (nx-2, ny-1, nz-2)
        dhz_dx = (hz[1:, :, :] - hz[:-1, :, :])[:, :, 1:-1] * inv_dx
        p_eyz = bz_e * state.p_eyz + cz_e * dhx_dz
        p_eyx = bx_e * state.p_eyx + cx_e * dhz_dx
        curl_y = dhx_dz * kz_e + p_eyz - dhz_dx * kx_e - p_eyx
        ey = ey.at[1:-1, :, 1:-1].set(ca_ey * ey[1:-1, :, 1:-1] + cb_ey * curl_y)

        dhy_dx = (hy[1:, :, :] - hy[:-1, :, :])[:, 1:-1, :] * inv_dx  # (nx-2, ny-2, nz-1)
        dhx_dy = (hx[:, 1:, :] - hx[:, :-1, :])[1:-1, :, :] * inv_dy
        p_ezx = bx_e * state.p_ezx + cx_e * dhy_dx
        p_ezy = by_e * state.p_ezy + cy_e * dhx_dy
        curl_z = dhy_dx * kx_e + p_ezx - dhx_dy * ky_e - p_ezy
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
            d_ex, d_ey, d_ez, d_hx, d_hy, d_hz = dft_acc
            eph = xs["e_phase"][:, None, None, None]
            hph = xs["h_phase"][:, None, None, None]
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
            p_exy, p_exz, p_eyx, p_eyz, p_ezx, p_ezy,
            p_hxy, p_hxz, p_hyx, p_hyz, p_hzx, p_hzy,
            dft_acc,
        )
        out = {"probe": ez[probe_idx[:, 0], probe_idx[:, 1], probe_idx[:, 2]]}
        if has_port:
            out["v"] = -ez[pi, pj, pk] * dz
            out["i"] = i_loop
        if record_energy:
            out["energy"] = field_energy_3d(ex, ey, ez, hx, hy, hz, eps, grid)
        return state, out

    zeros = lambda shape: jnp.zeros(shape, dtype)  # noqa: E731
    dft0 = None
    if has_dft:
        shapes = (
            (nx - 1, ny, nz), (nx, ny - 1, nz), (nx, ny, nz - 1),
            (nx, ny - 1, nz - 1), (nx - 1, ny, nz - 1), (nx - 1, ny - 1, nz),
        )
        dft0 = tuple(jnp.zeros((freqs.shape[0],) + s, cdtype) for s in shapes)
    state0 = _State3D(
        ex=zeros((nx - 1, ny, nz)),
        ey=zeros((nx, ny - 1, nz)),
        ez=zeros((nx, ny, nz - 1)),
        hx=zeros((nx, ny - 1, nz - 1)),
        hy=zeros((nx - 1, ny, nz - 1)),
        hz=zeros((nx - 1, ny - 1, nz)),
        p_exy=zeros((nx - 1, ny - 2, nz - 2)),
        p_exz=zeros((nx - 1, ny - 2, nz - 2)),
        p_eyx=zeros((nx - 2, ny - 1, nz - 2)),
        p_eyz=zeros((nx - 2, ny - 1, nz - 2)),
        p_ezx=zeros((nx - 2, ny - 2, nz - 1)),
        p_ezy=zeros((nx - 2, ny - 2, nz - 1)),
        p_hxy=zeros((nx, ny - 1, nz - 1)),
        p_hxz=zeros((nx, ny - 1, nz - 1)),
        p_hyx=zeros((nx - 1, ny, nz - 1)),
        p_hyz=zeros((nx - 1, ny, nz - 1)),
        p_hzx=zeros((nx - 1, ny - 1, nz)),
        p_hzy=zeros((nx - 1, ny - 1, nz)),
        dft=dft0,
    )

    xs = {}
    if has_src:
        xs["i_src"] = source_current
    if has_port:
        xs["vs"] = port_voltage
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
    if has_dft:
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
