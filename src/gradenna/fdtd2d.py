"""Differentiable 2D TM-mode FDTD with CPML absorbing boundaries.

Non-zero field components: Ez(i, j), Hx(i, j+1/2), Hy(i+1/2, j).
The whole time loop is a single `jax.lax.scan`, so any scalar loss of the
outputs can be differentiated with `jax.grad` with respect to the material
arrays (eps_r, sigma) and the source currents.

Update scheme (Schneider, *Understanding the FDTD Method*, Ch. 8/11):

    Hx^{n+1/2} = Hx - dt/mu [ dEz/dy / kappa_y + psi_Hx,y ]
    Hy^{n+1/2} = Hy + dt/mu [ dEz/dx / kappa_x + psi_Hy,x ]
    Ez^{n+1}   = Ca Ez + Cb [ dHy/dx / kappa_x + psi_Ez,x
                              - dHx/dy / kappa_y - psi_Ez,y - Jz ]

with Ca = (1 - sigma dt / 2 eps) / (1 + sigma dt / 2 eps) and
Cb = (dt/eps) / (1 + sigma dt / 2 eps). The outermost Ez ring is PEC.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gradenna.constants import EPS0, MU0
from gradenna.cpml import CPMLSpec, axis_coefficients
from gradenna.grid import Grid2D


class SimResult(NamedTuple):
    """Time series and final fields of a simulation.

    probe_ez: (n_steps, n_probes) Ez at the probe points; row n is time (n+1) dt.
    energy:   (n_steps,) total field energy [J/m], or None unless requested.
    ez, hx, hy: final field snapshots.
    """

    probe_ez: jnp.ndarray
    energy: jnp.ndarray | None
    ez: jnp.ndarray
    hx: jnp.ndarray
    hy: jnp.ndarray


class _State(NamedTuple):
    ez: jnp.ndarray
    hx: jnp.ndarray
    hy: jnp.ndarray
    p_ezx: jnp.ndarray
    p_ezy: jnp.ndarray
    p_hyx: jnp.ndarray
    p_hxy: jnp.ndarray


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
    source_ij,
    source_current,
    eps_r=1.0,
    sigma=0.0,
    probe_ij=(),
    cpml: CPMLSpec = CPMLSpec(),
    record_energy: bool = False,
) -> SimResult:
    """Run a 2D TM FDTD simulation.

    Args:
        grid: the Yee grid.
        source_ij: line-current Ez cell(s) — a single (i, j) pair or an
            (n_sources, 2) array.
        source_current: currents I(t) [A] sampled at t = (n+1/2) dt; shape
            (n_steps,) for a single source or (n_steps, n_sources).
        eps_r: relative permittivity at Ez points — scalar or (nx, ny).
        sigma: electric conductivity [S/m] at Ez points — scalar or (nx, ny).
        probe_ij: Ez points to record — sequence of (i, j) or (n_probes, 2).
        cpml: CPML parameters; thickness 0 gives a plain PEC box.
        record_energy: also record the total field energy at every step
            (adds full-grid reductions; off by default).

    Differentiable in eps_r, sigma and source_current.
    """
    nx, ny = grid.nx, grid.ny
    if min(nx, ny) <= 2 * cpml.thickness + 2:
        raise ValueError(f"grid {nx}x{ny} is too small for CPML thickness {cpml.thickness}")
    dt = grid.dt

    src_idx = _as_index_array(source_ij, "source_ij", grid, margin=1)
    probe_idx = _as_index_array(probe_ij, "probe_ij", grid, margin=0)

    source_current = jnp.asarray(source_current)
    if source_current.ndim == 1:
        source_current = source_current[:, None]
    if source_current.shape[1] != src_idx.shape[0]:
        raise ValueError(
            f"source_current has {source_current.shape[1]} columns "
            f"for {src_idx.shape[0]} sources"
        )

    dtype = jnp.result_type(source_current, eps_r, sigma)
    source_current = source_current.astype(dtype)

    eps = EPS0 * jnp.broadcast_to(jnp.asarray(eps_r, dtype), (nx, ny))
    sig = jnp.broadcast_to(jnp.asarray(sigma, dtype), (nx, ny))

    half_loss = sig * dt / (2.0 * eps)
    ca = (1.0 - half_loss) / (1.0 + half_loss)
    cb = (dt / eps) / (1.0 + half_loss)

    cx_e = axis_coefficients(nx, grid.dx, dt, cpml, half=False, dtype=dtype)
    cx_h = axis_coefficients(nx, grid.dx, dt, cpml, half=True, dtype=dtype)
    cy_e = axis_coefficients(ny, grid.dy, dt, cpml, half=False, dtype=dtype)
    cy_h = axis_coefficients(ny, grid.dy, dt, cpml, half=True, dtype=dtype)

    inv_dx, inv_dy = 1.0 / grid.dx, 1.0 / grid.dy
    dt_mu = dt / MU0
    # Discretized line current: Jz = I / (dx dy) over one cell.
    cb_src = cb[src_idx[:, 0], src_idx[:, 1]] * (inv_dx * inv_dy)

    def step(state: _State, i_n):
        ez, hx, hy, p_ezx, p_ezy, p_hyx, p_hxy = state

        dez_dy = (ez[:, 1:] - ez[:, :-1]) * inv_dy  # (nx, ny-1) at Hx points
        p_hxy = cy_h.b[None, :] * p_hxy + cy_h.c[None, :] * dez_dy
        hx = hx - dt_mu * (dez_dy * cy_h.inv_kappa[None, :] + p_hxy)

        dez_dx = (ez[1:, :] - ez[:-1, :]) * inv_dx  # (nx-1, ny) at Hy points
        p_hyx = cx_h.b[:, None] * p_hyx + cx_h.c[:, None] * dez_dx
        hy = hy + dt_mu * (dez_dx * cx_h.inv_kappa[:, None] + p_hyx)

        dhy_dx = (hy[1:, 1:-1] - hy[:-1, 1:-1]) * inv_dx  # (nx-2, ny-2)
        dhx_dy = (hx[1:-1, 1:] - hx[1:-1, :-1]) * inv_dy
        p_ezx = cx_e.b[1:-1, None] * p_ezx + cx_e.c[1:-1, None] * dhy_dx
        p_ezy = cy_e.b[None, 1:-1] * p_ezy + cy_e.c[None, 1:-1] * dhx_dy
        curl = (
            dhy_dx * cx_e.inv_kappa[1:-1, None]
            + p_ezx
            - dhx_dy * cy_e.inv_kappa[None, 1:-1]
            - p_ezy
        )
        ez = ez.at[1:-1, 1:-1].set(ca[1:-1, 1:-1] * ez[1:-1, 1:-1] + cb[1:-1, 1:-1] * curl)
        ez = ez.at[src_idx[:, 0], src_idx[:, 1]].add(-cb_src * i_n)

        state = _State(ez, hx, hy, p_ezx, p_ezy, p_hyx, p_hxy)
        probe_vals = ez[probe_idx[:, 0], probe_idx[:, 1]]
        if record_energy:
            return state, (probe_vals, field_energy(ez, hx, hy, eps, grid))
        return state, probe_vals

    state0 = _State(
        ez=jnp.zeros((nx, ny), dtype),
        hx=jnp.zeros((nx, ny - 1), dtype),
        hy=jnp.zeros((nx - 1, ny), dtype),
        p_ezx=jnp.zeros((nx - 2, ny - 2), dtype),
        p_ezy=jnp.zeros((nx - 2, ny - 2), dtype),
        p_hyx=jnp.zeros((nx - 1, ny), dtype),
        p_hxy=jnp.zeros((nx, ny - 1), dtype),
    )
    final, outputs = jax.lax.scan(step, state0, source_current)
    probe_ez, energy = outputs if record_energy else (outputs, None)
    return SimResult(probe_ez=probe_ez, energy=energy, ez=final.ez, hx=final.hx, hy=final.hy)
