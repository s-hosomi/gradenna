"""Differentiable 2D TM-mode FDTD with CPML absorbing boundaries.

Non-zero field components: Ez(i, j), Hx(i, j+1/2), Hy(i+1/2, j).
The whole time loop is a single `jax.lax.scan`, so any scalar loss of the
outputs can be differentiated with `jax.grad` with respect to the material
arrays (eps_r, sigma) and the source current.

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

from gradenna.constants import EPS0, MU0
from gradenna.cpml import CPMLSpec, axis_coefficients
from gradenna.grid import Grid2D


class SimResult(NamedTuple):
    """Time series recorded during a simulation.

    probe_ez: (n_steps, n_probes) Ez at the probe points; row n is time (n+1) dt.
    energy:   (n_steps,) total field energy [J/m] (per unit z length).
    ez_final: (nx, ny) final Ez snapshot.
    """

    probe_ez: jnp.ndarray
    energy: jnp.ndarray
    ez_final: jnp.ndarray


def field_energy(ez, hx, hy, eps, grid: Grid2D):
    """Total electromagnetic energy per unit length, 1/2 (eps E^2 + mu H^2) dA.

    Diagnostic only: E and H live at staggered times/locations, so this is
    conserved up to a bounded O((w dt)^2) oscillation in a lossless box.
    """
    cell = grid.dx * grid.dy
    ue = 0.5 * jnp.sum(eps * ez**2) * cell
    uh = 0.5 * MU0 * (jnp.sum(hx**2) + jnp.sum(hy**2)) * cell
    return ue + uh


def simulate_tm(
    grid: Grid2D,
    *,
    source_ij: tuple[int, int],
    source_current,
    eps_r=1.0,
    sigma=0.0,
    probe_ij: tuple = (),
    cpml: CPMLSpec = CPMLSpec(),
) -> SimResult:
    """Run a 2D TM FDTD simulation.

    Args:
        grid: the Yee grid.
        source_ij: (i, j) Ez cell of the line-current source.
        source_current: (n_steps,) current I(t) [A] sampled at t = (n+1/2) dt.
        eps_r: relative permittivity at Ez points — scalar or (nx, ny).
        sigma: electric conductivity [S/m] at Ez points — scalar or (nx, ny).
        probe_ij: tuple of (i, j) Ez points to record.
        cpml: CPML parameters; thickness 0 gives a plain PEC box.

    Differentiable in eps_r, sigma and source_current.
    """
    nx, ny = grid.nx, grid.ny
    dt = grid.dt
    source_current = jnp.asarray(source_current)
    dtype = jnp.result_type(source_current.dtype, jnp.zeros(0).dtype)

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
    si, sj = source_ij
    # Discretized line current: Jz = I / (dx dy) over one cell.
    j_scale = inv_dx * inv_dy
    probes = tuple(probe_ij)

    def step(state, i_n):
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
        ez = ez.at[si, sj].add(-cb[si, sj] * i_n * j_scale)

        state = (ez, hx, hy, p_ezx, p_ezy, p_hyx, p_hxy)
        probe_vals = jnp.stack([ez[i, j] for (i, j) in probes]) if probes else jnp.zeros((0,), dtype)
        return state, (probe_vals, field_energy(ez, hx, hy, eps, grid))

    state0 = (
        jnp.zeros((nx, ny), dtype),
        jnp.zeros((nx, ny - 1), dtype),
        jnp.zeros((nx - 1, ny), dtype),
        jnp.zeros((nx - 2, ny - 2), dtype),
        jnp.zeros((nx - 2, ny - 2), dtype),
        jnp.zeros((nx - 1, ny), dtype),
        jnp.zeros((nx, ny - 1), dtype),
    )
    final_state, (probe_ez, energy) = jax.lax.scan(step, state0, source_current)
    return SimResult(probe_ez=probe_ez, energy=energy, ez_final=final_state[0])
