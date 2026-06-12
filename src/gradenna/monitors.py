"""Differentiable frequency-domain power monitors.

Currently a single monitor: the closed-contour spectral Poynting flux of
the 2D TM solver, the radiated-power figure of merit used by the Phase 3/5
optimization demos and regression tests.
"""

from __future__ import annotations

import jax.numpy as jnp

from gradenna.grid import Grid2D

__all__ = [
    "log_radiated_fraction",
    "poynting_flux_box_2d",
]


def poynting_flux_box_2d(dft_ez, dft_hx, dft_hy, grid: Grid2D, box):
    """Outward spectral Poynting flux through a closed dual-grid rectangle.

    The contour is the rectangle x in [il+1/2, ir+1/2], y in [jb+1/2, jt+1/2]
    through the H sample points (half a cell outside the Ez nodes il/ir/jb/jt),
    with ``box = (il, ir, jb, jt)``. Choose it outside the design region and
    inside the CPML interface.

    Conventions:
    - Time-average Poynting vector of phasors, S = 1/2 Re(E x H*); for the TM
      mode Sx = -1/2 Re(Ez Hy*) and Sy = +1/2 Re(Ez Hx*).
    - Yee staggering: Ez is averaged onto the H positions of each face (e.g.
      on the right face Ez(ir, j) and Ez(ir+1, j) -> Ez(ir+1/2, j)); the E/H
      half-time-step offset is already compensated by the solver's
      exact-sample-time running DFT ((n+1) dt for Ez, (n+1/2) dt for Hx/Hy).
    - Units: [W/m] per unit z for absolute phasors; for the dt-scaled DFT
      phasors of `simulate_tm` the value is directly comparable with source
      and port powers computed from spectra using the same dt-scaled DFT
      (e.g. ``half_step_dft``), exactly as in `gradenna.ntff`.

    All operations are jnp, so the flux is differentiable with `jax.grad`.

    Args:
        dft_ez: (n_freq, nx, ny) Ez phasors (`SimResult.dft_ez`).
        dft_hx: (n_freq, nx, ny-1) Hx phasors.
        dft_hy: (n_freq, nx-1, ny) Hy phasors.
        grid: the 2D Yee grid of the simulation.
        box: (il, ir, jb, jt) Ez-node indices of the contour rectangle.

    Returns:
        (n_freq,) outward flux per DFT frequency.
    """
    il, ir, jb, jt = (int(v) for v in box)
    if not (0 <= il < ir < grid.nx - 1 and 0 <= jb < jt < grid.ny - 1):
        raise ValueError(f"box {box!r} is not a valid contour on a {grid.nx}x{grid.ny} grid")
    dx, dy = grid.dx, grid.dy
    js = slice(jb + 1, jt + 1)
    isl = slice(il + 1, ir + 1)
    # Right face (outward +x): Sx = -1/2 Re(Ez Hy*).
    ez_r = 0.5 * (dft_ez[:, ir, js] + dft_ez[:, ir + 1, js])
    p = -0.5 * jnp.real(ez_r * jnp.conj(dft_hy[:, ir, js])).sum(-1) * dy
    # Left face (outward -x).
    ez_l = 0.5 * (dft_ez[:, il, js] + dft_ez[:, il + 1, js])
    p += 0.5 * jnp.real(ez_l * jnp.conj(dft_hy[:, il, js])).sum(-1) * dy
    # Top face (outward +y): Sy = +1/2 Re(Ez Hx*).
    ez_t = 0.5 * (dft_ez[:, isl, jt] + dft_ez[:, isl, jt + 1])
    p += 0.5 * jnp.real(ez_t * jnp.conj(dft_hx[:, isl, jt])).sum(-1) * dx
    # Bottom face (outward -y).
    ez_b = 0.5 * (dft_ez[:, isl, jb] + dft_ez[:, isl, jb + 1])
    p -= 0.5 * jnp.real(ez_b * jnp.conj(dft_hx[:, isl, jb])).sum(-1) * dx
    return p


def log_radiated_fraction(p_rad, p_avail):
    """Scale-invariant log radiated-power objective ``log P_rad - log P_avail``.

    The topology-optimization figure of merit is the radiated-power fraction
    ``P_rad / P_avail`` (`poynting_flux_box_2d` flux normalized by the available
    source power). For maximization, ``log P_rad - log P_avail`` is monotone in
    that ratio but **scale invariant**: its gradient is ``(1/P_rad) dP_rad``,
    which does not multiply the (possibly tiny) absolute flux back in. This is
    the float32-robust form when both powers sit far below 1 — the linear ratio
    differentiates ``P_rad/P_avail`` and the backward pass carries the small
    ``1/P_avail`` factor through the flux product, whereas the log form keeps
    the relative sensitivity at order 1 regardless of the absolute scale.

    Note this rescues only *finite, positive* fluxes: if ``P_rad`` has already
    underflowed the field dtype to exactly 0 (extreme attenuation, see
    ``simulate_tm``'s ``dft_dtype`` argument), the log is -inf and no loss
    reformulation can recover a gradient — keep the DFT accumulator in higher
    precision and/or the fields out of the underflow regime instead.

    Args:
        p_rad: radiated power (e.g. a `poynting_flux_box_2d` entry), > 0.
        p_avail: available source power |Vs_hat|^2 / (8 Rs), > 0.

    Returns:
        ``log(P_rad) - log(P_avail)``, the same dtype-promoted shape as the
        inputs. Differentiable with `jax.grad`.
    """
    return jnp.log(p_rad) - jnp.log(p_avail)
