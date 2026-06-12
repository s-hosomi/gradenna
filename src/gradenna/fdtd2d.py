"""Differentiable 2D TM-mode FDTD with CPML absorbing boundaries.

Non-zero field components: Ez(i, j), Hx(i, j+1/2), Hy(i+1/2, j).
The whole time loop is a single `jax.lax.scan`, so any scalar loss of the
outputs can be differentiated with `jax.grad` with respect to the material
arrays (eps_r, sigma), the source currents and the port voltage waveforms.

Update scheme (Schneider, *Understanding the FDTD Method*, Ch. 8/11):

    Hx^{n+1/2} = Hx - dt/mu [ dEz/dy / kappa_y + psi_Hx,y ]
    Hy^{n+1/2} = Hy + dt/mu [ dEz/dx / kappa_x + psi_Hy,x ]
    Ez^{n+1}   = Ca Ez + Cb [ dHy/dx / kappa_x + psi_Ez,x
                              - dHx/dy / kappa_y - psi_Ez,y - Jz ]

with Ca = (1 - sigma dt / 2 eps) / (1 + sigma dt / 2 eps) and
Cb = (dt/eps) / (1 + sigma dt / 2 eps). The outermost Ez ring is PEC.

Lumped RVS ports (docs/research/12-port-s11-theory.md, Sec. 1.3, 2D form)
-------------------------------------------------------------------------

The 2D TM system is the dz = DZ = 1 m unit-length slice of a z-invariant 3D
problem, so the gap voltage is V = -Ez * DZ and a Thevenin branch (source
Vs(t) in series with Rs) across the gap carries

    I_L^{n+1/2} = (Vs^{n+1/2} - V^{n+1/2}) / Rs
                = [Vs^{n+1/2} + DZ (Ez^{n+1} + Ez^n)/2] / Rs

(semi-implicit average of Piket-May/Taflove/Baron 1994). Substituting
J_L = I_L / (dx dy) into the Maxwell-Ampere update and solving for Ez^{n+1}
gives exactly the boxed RVS update of note 12 Sec. 1.3,

    Ez^{n+1} = (1-beta)/(1+beta) Ez^n
               + (dt/eps)/(1+beta) (curl H)_z
               - dt / (Rs eps dx dy (1+beta)) Vs^{n+1/2},
    beta = dt DZ / (2 Rs eps dx dy),

which is implemented here by (a) adding the equivalent cell conductivity
sigma_port = DZ / (Rs dx dy) to `sigma` before building Ca/Cb (the beta term
then coincides with the standard semi-implicit loss term), and (b) injecting
the source term Cb * (-Vs / (Rs dx dy)) at the port cell.

Port recordings (note 12 Sec. 2.1/2.2), both time-aligned at t=(n+1/2) dt:

    V^{n+1/2} = -DZ (Ez^n + Ez^{n+1}) / 2
    I^{n+1/2} = Ampere loop around the port cell
              = (Hy|i+1/2,j - Hy|i-1/2,j) dy - (Hx|i,j+1/2 - Hx|i,j-1/2) dx

I is the *total* (conduction + displacement) current through the loop, so
the gap capacitance eps dx dy / DZ appears as a parallel shunt (note 12
Sec. 6.1) -- a known systematic of lumped FDTD ports.

DFT field monitor (note 12 Sec. 5.1): for each requested frequency the scan
carry accumulates X(f) = dt sum_n x^n exp(-i 2 pi f t_n) with the exact
sample times t_n = (n+1) dt for Ez and t_n = (n+1/2) dt for Hx/Hy, which is
mathematically identical to a post-hoc DFT of the full time series.
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
from gradenna.grid import Grid2D

#: Unit length [m] of the 2D TM slice; port voltage is V = -Ez * DZ.
DZ = 1.0


class Port(NamedTuple):
    """Lumped resistive voltage source (RVS) port on a single Ez cell.

    A Thevenin branch (ideal source `voltage` in series with `resistance`)
    connected across the unit-length gap of the Ez edge at `ij`, following
    note 12 Sec. 1.3 (see the module docstring for the 2D specialization).

    Attributes:
        ij: (i, j) index of the Ez cell carrying the port.
        resistance: internal resistance Rs per unit length [ohm m]
            (numerically equal to ohms for the DZ = 1 m slice). Must be > 0.
        voltage: source waveform Vs(t) sampled at t = (n+1/2) dt, shape
            (n_steps,); None makes the port a passive resistive load (Vs=0).
    """

    ij: tuple[int, int]
    resistance: float = 50.0
    voltage: jnp.ndarray | None = None


class SimResult(NamedTuple):
    """Time series, frequency-domain monitors and final fields.

    probe_ez: (n_steps, n_probes) Ez at the probe points; row n is time (n+1) dt.
    energy:   (n_steps,) total field energy [J/m], or None unless requested.
    ez, hx, hy: final field snapshots.
    port_v: (n_steps, n_ports) port voltage V = -DZ (Ez^n + Ez^{n+1})/2,
        time-aligned at t = (n+1/2) dt; None if no ports were given.
    port_i: (n_steps, n_ports) Ampere-loop port current at t = (n+1/2) dt;
        None if no ports were given.
    dft_ez: (n_freq, nx, ny) running-DFT phasor of Ez with exact (n+1) dt
        phases (already scaled by dt); None unless dft_freqs was given.
    dft_hx: (n_freq, nx, ny-1) DFT of Hx with exact (n+1/2) dt phases.
    dft_hy: (n_freq, nx-1, ny) DFT of Hy with exact (n+1/2) dt phases.
    """

    probe_ez: jnp.ndarray
    energy: jnp.ndarray | None
    ez: jnp.ndarray
    hx: jnp.ndarray
    hy: jnp.ndarray
    port_v: jnp.ndarray | None = None
    port_i: jnp.ndarray | None = None
    dft_ez: jnp.ndarray | None = None
    dft_hx: jnp.ndarray | None = None
    dft_hy: jnp.ndarray | None = None


class _Psi2D(NamedTuple):
    """CPML psi variables, each as a low/high PML slab pair (note 14 Sec. 5.3).

    Only the slabs along the stretched axis are stored (the slabs span the
    full transverse extent, corners included); outside them psi == 0.
    """

    ezx: PsiSlabs  # (npml, ny-2) x2 — x-stretched, interior Ez rows
    ezy: PsiSlabs  # (nx-2, npml) x2 — y-stretched, interior Ez columns
    hyx: PsiSlabs  # (npml, ny)   x2 — x-stretched, Hy points
    hxy: PsiSlabs  # (nx, npml)   x2 — y-stretched, Hx points


def _init_psi(nx: int, ny: int, npml: int, dtype) -> _Psi2D:
    """Zero-initialized slab-stored psi state (note 14 Sec. 5.3)."""
    pair = lambda shape: PsiSlabs(jnp.zeros(shape, dtype), jnp.zeros(shape, dtype))  # noqa: E731
    return _Psi2D(
        ezx=pair((npml, ny - 2)),
        ezy=pair((nx - 2, npml)),
        hyx=pair((npml, ny)),
        hxy=pair((nx, npml)),
    )


class _State(NamedTuple):
    ez: jnp.ndarray
    hx: jnp.ndarray
    hy: jnp.ndarray
    psi: _Psi2D


def field_energy(ez, hx, hy, eps, grid: Grid2D):
    """Total electromagnetic energy per unit length, 1/2 (eps E^2 + mu H^2) dA.

    Diagnostic only: E and H live at staggered times/locations, so this is
    conserved up to a bounded O((w dt)^2) oscillation in a lossless box.
    """
    cell = grid.dx * grid.dy
    ue = 0.5 * jnp.sum(eps * ez**2) * cell
    uh = 0.5 * MU0 * (jnp.sum(hx**2) + jnp.sum(hy**2)) * cell
    return ue + uh


def _as_index_array(ij, name: str, grid: Grid2D, margin: int) -> np.ndarray:
    idx = np.asarray(ij, dtype=np.int32).reshape(-1, 2)
    lo = np.array([margin, margin])
    hi = np.array([grid.nx - 1 - margin, grid.ny - 1 - margin])
    if idx.size and (np.any(idx < lo) or np.any(idx > hi)):
        raise ValueError(f"{name} {idx.tolist()} outside the interior (PML margin {margin})")
    return idx


def simulate_tm(
    grid: Grid2D,
    *,
    source_ij=None,
    source_current=None,
    eps_r=1.0,
    sigma=0.0,
    probe_ij=(),
    ports=(),
    dft_freqs=(),
    cpml: CPMLSpec = CPMLSpec(),
    record_energy: bool = False,
) -> SimResult:
    """Run a 2D TM FDTD simulation.

    Args:
        grid: the Yee grid.
        source_ij: line-current Ez cell(s) — a single (i, j) pair or an
            (n_sources, 2) array. Optional if at least one active port is
            given.
        source_current: currents I(t) [A] sampled at t = (n+1/2) dt; shape
            (n_steps,) for a single source or (n_steps, n_sources).
        eps_r: relative permittivity at Ez points — scalar or (nx, ny).
        sigma: electric conductivity [S/m] at Ez points — scalar or (nx, ny).
        probe_ij: Ez points to record — sequence of (i, j) or (n_probes, 2).
        ports: sequence of `Port` specs (or (ij, resistance, voltage) tuples).
            Each adds a lumped RVS branch (note 12 Sec. 1.3) at its Ez cell
            and records V(t)/I(t) at t = (n+1/2) dt into the result. Two
            ports on the same cell combine as parallel branches.
        dft_freqs: frequencies [Hz] for the running DFT field monitor.
            When non-empty, the result carries complex Ez/Hx/Hy phasors
            accumulated with exact sample-time phases (note 12 Sec. 5.1).
        cpml: CPML parameters; thickness 0 gives a plain PEC box.
        record_energy: also record the total field energy at every step
            (adds full-grid reductions; off by default).

    The number of time steps is taken from `source_current` and/or the port
    voltage waveforms (which must agree); at least one of them is required.

    Differentiable in eps_r, sigma, source_current and the port voltages.
    """
    nx, ny = grid.nx, grid.ny
    if min(nx, ny) <= 2 * cpml.thickness + 2:
        raise ValueError(f"grid {nx}x{ny} is too small for CPML thickness {cpml.thickness}")
    dt = grid.dt

    if (source_ij is None) != (source_current is None):
        raise ValueError("source_ij and source_current must be given together")

    ports = tuple(p if isinstance(p, Port) else Port(*p) for p in ports)
    n_ports = len(ports)
    port_voltages = [None if p.voltage is None else jnp.asarray(p.voltage) for p in ports]

    # Determine n_steps from the source and/or port waveforms.
    n_steps = None
    if source_current is not None:
        source_current = jnp.asarray(source_current)
        n_steps = source_current.shape[0]
    for v in port_voltages:
        if v is None:
            continue
        if v.ndim != 1:
            raise ValueError(f"port voltage waveform must be 1D, got shape {v.shape}")
        if n_steps is None:
            n_steps = v.shape[0]
        elif v.shape[0] != n_steps:
            raise ValueError(
                f"port voltage length {v.shape[0]} does not match n_steps {n_steps}"
            )
    if n_steps is None:
        raise ValueError(
            "need source_current and/or at least one port with a voltage waveform"
        )

    operands = [eps_r, sigma]
    if source_current is not None:
        operands.append(source_current)
    operands.extend(v for v in port_voltages if v is not None)
    dtype = jnp.result_type(*operands)

    probe_idx = _as_index_array(probe_ij, "probe_ij", grid, margin=0)

    if source_current is None:
        src_idx = np.zeros((0, 2), dtype=np.int32)
        source_current = jnp.zeros((n_steps, 0), dtype)
    else:
        src_idx = _as_index_array(source_ij, "source_ij", grid, margin=1)
        if source_current.ndim == 1:
            source_current = source_current[:, None]
        if source_current.shape[1] != src_idx.shape[0]:
            raise ValueError(
                f"source_current has {source_current.shape[1]} columns "
                f"for {src_idx.shape[0]} sources"
            )
        source_current = source_current.astype(dtype)

    if n_ports:
        port_idx = _as_index_array([p.ij for p in ports], "port ij", grid, margin=1)
        rs_np = np.asarray([float(p.resistance) for p in ports])
        if np.any(rs_np <= 0.0):
            raise ValueError("port resistance must be positive")
        port_rs = jnp.asarray(rs_np, dtype)
        port_vs = jnp.stack(
            [jnp.zeros((n_steps,), dtype) if v is None else v.astype(dtype) for v in port_voltages],
            axis=1,
        )
        pi, pj = port_idx[:, 0], port_idx[:, 1]

    eps = EPS0 * jnp.broadcast_to(jnp.asarray(eps_r, dtype), (nx, ny))
    sig = jnp.broadcast_to(jnp.asarray(sigma, dtype), (nx, ny))
    if n_ports:
        # RVS resistor == extra cell conductivity sigma_port = DZ/(Rs dx dy):
        # the semi-implicit loss term then reproduces beta of note 12 Sec. 1.3.
        sig = sig.at[pi, pj].add(DZ / (port_rs * grid.dx * grid.dy))

    half_loss = sig * dt / (2.0 * eps)
    ca = (1.0 - half_loss) / (1.0 + half_loss)
    cb = (dt / eps) / (1.0 + half_loss)

    cx_e = axis_coefficients(nx, grid.dx, dt, cpml, half=False, dtype=dtype)
    cx_h = axis_coefficients(nx, grid.dx, dt, cpml, half=True, dtype=dtype)
    cy_e = axis_coefficients(ny, grid.dy, dt, cpml, half=False, dtype=dtype)
    cy_h = axis_coefficients(ny, grid.dy, dt, cpml, half=True, dtype=dtype)

    # Slab (strip) storage of psi (note 14 Sec. 5.3): static slices of the
    # two PML slabs along each stretched axis, plus the b/c tables restricted
    # to them. 1/kappa stays full-size (== 1 outside the PML).
    npml = cpml.thickness
    sx_e = slab_slices(nx - 2, npml)  # E-type psi live on the PEC interior
    sy_e = slab_slices(ny - 2, npml)
    sx_h = slab_slices(nx - 1, npml)  # H-type psi live on the half grid
    sy_h = slab_slices(ny - 1, npml)
    bc_ezx = slab_coefficients(cx_e.b[1:-1], cx_e.c[1:-1], sx_e, axis=0, ndim=2)
    bc_ezy = slab_coefficients(cy_e.b[1:-1], cy_e.c[1:-1], sy_e, axis=1, ndim=2)
    bc_hyx = slab_coefficients(cx_h.b, cx_h.c, sx_h, axis=0, ndim=2)
    bc_hxy = slab_coefficients(cy_h.b, cy_h.c, sy_h, axis=1, ndim=2)
    kx_e = cx_e.inv_kappa[1:-1, None]
    ky_e = cy_e.inv_kappa[None, 1:-1]
    kx_h = cx_h.inv_kappa[:, None]
    ky_h = cy_h.inv_kappa[None, :]

    inv_dx, inv_dy = 1.0 / grid.dx, 1.0 / grid.dy
    dt_mu = dt / MU0
    # Discretized line current: Jz = I / (dx dy) over one cell.
    cb_src = cb[src_idx[:, 0], src_idx[:, 1]] * (inv_dx * inv_dy)
    if n_ports:
        # Source term of the RVS update: Ez += Cb * (-Vs / (Rs dx dy)),
        # i.e. -dt Vs / (Rs eps dx dy (1+beta)) of note 12 Sec. 1.3.
        cb_vs = cb[pi, pj] / (port_rs * grid.dx * grid.dy)

    dft_freqs = tuple(float(f) for f in dft_freqs)
    n_freq = len(dft_freqs)
    if n_freq:
        cdtype = jnp.result_type(dtype, np.complex64)
        # Exact-phase tables, generated in float64 (note 12 Sec. 5.2: never
        # build the phasor recursively in low precision).
        f_np = np.asarray(dft_freqs, np.float64)
        n_np = np.arange(n_steps, dtype=np.float64)
        ph_e = jnp.asarray(np.exp(-2j * np.pi * np.outer(n_np + 1.0, f_np) * dt), cdtype)
        ph_h = jnp.asarray(np.exp(-2j * np.pi * np.outer(n_np + 0.5, f_np) * dt), cdtype)

    xs = {"j": source_current}
    if n_ports:
        xs["vs"] = port_vs
    if n_freq:
        xs["ph_e"] = ph_e
        xs["ph_h"] = ph_h

    def step(carry, x):
        state, acc = carry
        ez, hx, hy, psi = state

        dez_dy = (ez[:, 1:] - ez[:, :-1]) * inv_dy  # (nx, ny-1) at Hx points
        p_hxy, term_hx = psi_step(psi.hxy, dez_dy, bc_hxy, sy_h, 1, ky_h)
        hx = hx - dt_mu * term_hx

        dez_dx = (ez[1:, :] - ez[:-1, :]) * inv_dx  # (nx-1, ny) at Hy points
        p_hyx, term_hy = psi_step(psi.hyx, dez_dx, bc_hyx, sx_h, 0, kx_h)
        hy = hy + dt_mu * term_hy

        if n_ports:
            # Ampere loop around the port cell with H at (n+1/2) dt
            # (note 12 Sec. 2.1): total +z current through the loop.
            i_port = (hy[pi, pj] - hy[pi - 1, pj]) * grid.dy - (
                hx[pi, pj] - hx[pi, pj - 1]
            ) * grid.dx
            ez_before = ez[pi, pj]

        dhy_dx = (hy[1:, 1:-1] - hy[:-1, 1:-1]) * inv_dx  # (nx-2, ny-2)
        dhx_dy = (hx[1:-1, 1:] - hx[1:-1, :-1]) * inv_dy
        p_ezx, term_x = psi_step(psi.ezx, dhy_dx, bc_ezx, sx_e, 0, kx_e)
        p_ezy, term_y = psi_step(psi.ezy, dhx_dy, bc_ezy, sy_e, 1, ky_e)
        curl = term_x - term_y
        ez = ez.at[1:-1, 1:-1].set(ca[1:-1, 1:-1] * ez[1:-1, 1:-1] + cb[1:-1, 1:-1] * curl)
        ez = ez.at[src_idx[:, 0], src_idx[:, 1]].add(-cb_src * x["j"])

        out = {}
        if n_ports:
            ez = ez.at[pi, pj].add(-cb_vs * x["vs"])
            # V^{n+1/2} = -DZ (Ez^n + Ez^{n+1})/2: time-aligned with I.
            out["v"] = -0.5 * DZ * (ez_before + ez[pi, pj])
            out["i"] = i_port

        if n_freq:
            acc = (
                acc[0] + x["ph_e"][:, None, None] * ez,
                acc[1] + x["ph_h"][:, None, None] * hx,
                acc[2] + x["ph_h"][:, None, None] * hy,
            )

        state = _State(ez, hx, hy, _Psi2D(p_ezx, p_ezy, p_hyx, p_hxy))
        out["probe"] = ez[probe_idx[:, 0], probe_idx[:, 1]]
        if record_energy:
            out["energy"] = field_energy(ez, hx, hy, eps, grid)
        return (state, acc), out

    state0 = _State(
        ez=jnp.zeros((nx, ny), dtype),
        hx=jnp.zeros((nx, ny - 1), dtype),
        hy=jnp.zeros((nx - 1, ny), dtype),
        psi=_init_psi(nx, ny, npml, dtype),
    )
    acc0 = None
    if n_freq:
        acc0 = (
            jnp.zeros((n_freq, nx, ny), cdtype),
            jnp.zeros((n_freq, nx, ny - 1), cdtype),
            jnp.zeros((n_freq, nx - 1, ny), cdtype),
        )

    (final, acc_final), outputs = jax.lax.scan(step, (state0, acc0), xs)

    dft_ez = dft_hx = dft_hy = None
    if n_freq:
        # X(f) = dt sum_n x^n exp(-i 2 pi f t_n)  (note 12 Sec. 5.1).
        dft_ez = acc_final[0] * dt
        dft_hx = acc_final[1] * dt
        dft_hy = acc_final[2] * dt

    return SimResult(
        probe_ez=outputs["probe"],
        energy=outputs.get("energy"),
        ez=final.ez,
        hx=final.hx,
        hy=final.hy,
        port_v=outputs.get("v"),
        port_i=outputs.get("i"),
        dft_ez=dft_ez,
        dft_hx=dft_hx,
        dft_hy=dft_hy,
    )
