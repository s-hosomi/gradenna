r"""Differentiable thin-wire Method of Moments (MoM) backend.

A fully ``jax``-differentiable Method of Moments solver for a straight,
center-fed, z-directed thin wire (linear dipole).  This is the minimal
skeleton requested as a Phase 5 extension (see ``docs/research/06`` on the
feasibility of a differentiable MoM route).

Formulation
-----------
We solve the electric-field integral equation (EFIE) for a straight wire of
length ``L`` and radius ``a`` lying on the z-axis, using

  * **piecewise-sinusoidal (PWS) basis functions** with **Galerkin testing**
    (the *reaction* / induced-EMF method), and
  * the standard **thin-wire kernel**: the source current is treated as a
    filament on the wire axis while the observation point is placed on the
    wire surface (``rho = a``), which regularizes the ``1/R`` self-term
    singularity without an explicit singularity extraction.

Each PWS mode ``n`` spans two adjacent segments and carries the sinusoidal
current ``I_n(z) = sin(k (d - |z - z_n|)) / sin(k d)`` on ``[z_n - d, z_n + d]``
with half-width ``d`` (the node spacing) and peak value 1 at its node ``z_n``.
The overlapping triangular-support modes enforce current continuity and a
zero end current automatically.  The Galerkin EFIE matrix element is the
standard weak (reaction) form

.. math::

    Z_{mn} = \frac{j \eta_0}{k}\!\iint
        \big[k^2 I_m(z) I_n(z') - I_m'(z) I_n'(z')\big]\,
        \frac{e^{-jkR}}{4\pi R}\,dz\,dz',
    \qquad R = \sqrt{(z - z')^2 + a^2},

evaluated by tensor-product Gauss-Legendre quadrature over the two-segment
supports.  This is the textbook PWS / EMF formulation; see Balanis,
*Antenna Theory: Analysis and Design*, 3rd ed., Sec. 8.5-8.6 (integral
equations, the moment method, and the piecewise-sinusoidal Galerkin solution
of the dipole), and Harrington, *Field Computation by Moment Methods*,
Ch. 4 (thin-wire EFIE, point matching vs. Galerkin).  PWS-Galerkin is
preferred here over pulse / point-matching because the latter converges
poorly (the dipole input resistance fails to settle) for the thin-wire
dipole, whereas PWS-Galerkin reproduces the resonance near ``0.47-0.48 lambda``
and a stable input impedance with only a few tens of modes.

Excitation and input impedance
------------------------------
A **delta-gap** feed is applied at the center mode: the excitation vector has
a unit tangential voltage on the feed mode and zero elsewhere.  Solving
``Z I = V`` for the modal current amplitudes ``I`` gives the input current
``I_feed``, and the input impedance is ``Zin = V_feed / I_feed = 1 / I_feed``.

Differentiability
-----------------
The whole chain (geometry -> quadrature -> ``jnp.linalg.solve`` -> ``Zin``)
is written in ``jnp`` and is differentiable end-to-end.  ``jax.grad`` flows
through ``length`` and ``radius``; the linear solve uses the implicit VJP of
``jnp.linalg.solve`` (the adjoint is a transpose-solve of the same matrix), so
no solver internals are unrolled.  ``float64`` is assumed (run with
``JAX_ENABLE_X64=1``).

Relation to the FDTD backend
----------------------------
This MoM backend models perfectly conducting wires in free space.  It is a
fast surrogate for PEC structures (no substrate); extending it to planar
antennas on a dielectric (e.g. FR-4) requires layered-media Green's functions
(MPIE + Sommerfeld integrals / DCIM), which is left as future work
(``docs/research/06``, theme A).  For substrate-bearing structures the FDTD
backend (:mod:`gradenna.fdtd2d`, :mod:`gradenna.fdtd3d`) remains the route.

References
----------
* C. A. Balanis, *Antenna Theory: Analysis and Design*, 3rd ed., Wiley 2005,
  Ch. 8 (integral equations and the moment method).
* R. F. Harrington, *Field Computation by Moment Methods*, IEEE Press 1993,
  Ch. 4 (thin-wire EFIE).
* J. H. Richmond, "Digital computer solutions of the rigorous equations for
  scattering problems," Proc. IEEE, 1965 (PWS reaction integrals).
"""

from __future__ import annotations

import functools

import jax.numpy as jnp
import numpy as np

from .constants import C0, ETA0

__all__ = [
    "wire_impedance_matrix",
    "wire_dipole_impedance",
    "wire_dipole_input_current",
]

# Default number of Gauss-Legendre nodes per segment support for the
# tensor-product reaction quadrature.  80 is comfortably converged for the
# thin-wire dipole over the radii of interest (a/lambda ~ 1e-3 ... 1e-2).
_DEFAULT_NQ = 80


@functools.lru_cache(maxsize=8)
def _gauss_legendre(nq: int):
    """Cached Gauss-Legendre nodes/weights on [-1, 1] as jnp arrays."""
    x, w = np.polynomial.legendre.leggauss(int(nq))
    return jnp.asarray(x), jnp.asarray(w)


def _pws_current(z, z_node, half_width, k):
    """Piecewise-sinusoidal mode current at ``z`` for a node at ``z_node``.

    ``I(z) = sin(k (d - |z - z_node|)) / sin(k d)`` on the two-segment support
    of half-width ``d = half_width``; peak value 1 at ``z_node``.
    """
    return jnp.sin(k * (half_width - jnp.abs(z - z_node))) / jnp.sin(k * half_width)


def _pws_current_deriv(z, z_node, half_width, k):
    """d/dz of :func:`_pws_current` (used in the weak-form reaction integral)."""
    s = jnp.sign(z - z_node)
    return -s * k * jnp.cos(k * (half_width - jnp.abs(z - z_node))) / jnp.sin(
        k * half_width
    )


def wire_impedance_matrix(length, radius, freq, n_modes, nq: int = _DEFAULT_NQ):
    """Galerkin PWS thin-wire EFIE impedance matrix for a z-directed wire.

    Parameters
    ----------
    length : float
        Wire length ``L`` [m].
    radius : float
        Wire radius ``a`` [m] (used in the thin-wire kernel ``R = sqrt(dz^2 + a^2)``).
    freq : float
        Frequency [Hz].
    n_modes : int
        Number of overlapping piecewise-sinusoidal modes (unknowns).  Nodes are
        equispaced at ``z_n = -L/2 + n d``, ``d = L / (n_modes + 1)``,
        ``n = 1 .. n_modes``; the end nodes at +-L/2 carry zero current.
    nq : int, optional
        Gauss-Legendre nodes per segment support for the reaction quadrature.

    Returns
    -------
    Z : complex jnp.ndarray, shape (n_modes, n_modes)
        Symmetric (reciprocal) impedance matrix [ohm].
    """
    k = 2.0 * jnp.pi * freq / C0
    d = length / (n_modes + 1)
    nodes = -length / 2.0 + (jnp.arange(n_modes) + 1) * d  # (N,)

    gx, gw = _gauss_legendre(nq)

    # Node positions broadcast over all (m, n) mode pairs.
    zi = nodes[:, None]  # observation modes  (N, N)
    zj = nodes[None, :]  # source modes       (N, N)

    # Quadrature points on each mode's [-d, d] support, shape (N, N, NQ_obs, NQ_src).
    oz = zi[:, :, None, None] + gx[None, None, :, None] * d
    sz = zj[:, :, None, None] + gx[None, None, None, :] * d

    zin_node = zi[:, :, None, None]
    zjn_node = zj[:, :, None, None]

    im = _pws_current(oz, zin_node, d, k)
    is_ = _pws_current(sz, zjn_node, d, k)
    dim = _pws_current_deriv(oz, zin_node, d, k)
    dis = _pws_current_deriv(sz, zjn_node, d, k)

    # Thin-wire kernel: source on axis, observation on surface (rho = a).
    R = jnp.sqrt((oz - sz) ** 2 + radius**2)
    green = jnp.exp(-1j * k * R) / (4.0 * jnp.pi * R)

    integrand = (k * k * im * is_ - dim * dis) * green
    weights = gw[None, None, :, None] * gw[None, None, None, :] * d * d

    z_mat = 1j * ETA0 / k * jnp.sum(weights * integrand, axis=(2, 3))
    return z_mat


def wire_dipole_input_current(
    length, radius, freq, n_modes: int = 39, nq: int = _DEFAULT_NQ
):
    """Modal input current of a center-fed delta-gap thin-wire dipole.

    Returns the complex feed-mode current amplitude for a unit delta-gap
    voltage.  ``n_modes`` must be odd so that a mode sits exactly at the
    feed point (the wire center).
    """
    if n_modes % 2 == 0:
        raise ValueError("n_modes must be odd so a mode is centered at the feed.")
    z_mat = wire_impedance_matrix(length, radius, freq, n_modes, nq=nq)
    feed = n_modes // 2
    v = jnp.zeros(n_modes, dtype=z_mat.dtype).at[feed].set(1.0 + 0.0j)
    current = jnp.linalg.solve(z_mat, v)
    return current[feed]


def wire_dipole_impedance(length, radius, freqs, n_segments: int = 39, nq: int = _DEFAULT_NQ):
    """Input impedance ``Zin(f)`` of a center-fed straight thin-wire dipole.

    Delta-gap excitation; ``Zin = V_feed / I_feed = 1 / I_feed`` for unit feed
    voltage.  Fully differentiable in ``length`` and ``radius`` via ``jax.grad``.

    Parameters
    ----------
    length : float
        Dipole total length ``L`` [m].
    radius : float
        Wire radius ``a`` [m].
    freqs : float or array_like
        Frequency or frequencies [Hz].
    n_segments : int, optional
        Number of PWS modes (unknowns).  Must be odd (a mode is placed at the
        feed).  Default 39 gives ~1% input-impedance convergence for the
        half-wave dipole.
    nq : int, optional
        Gauss-Legendre nodes per segment support.

    Returns
    -------
    Zin : complex jnp.ndarray
        Input impedance [ohm]; scalar for scalar ``freqs``, else one per
        frequency.
    """
    freqs_arr = jnp.atleast_1d(jnp.asarray(freqs, dtype=jnp.float64))

    def _one(f):
        return 1.0 / wire_dipole_input_current(length, radius, f, n_modes=n_segments, nq=nq)

    # Loop in Python over the (typically few) frequencies; each is an
    # independent, fully differentiable solve.
    zins = jnp.stack([_one(f) for f in freqs_arr])
    if jnp.ndim(jnp.asarray(freqs)) == 0:
        return zins[0]
    return zins
