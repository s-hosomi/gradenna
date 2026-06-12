"""Convolutional PML (CPML, Roden & Gedney 2000) coefficient tables.

The stretched-coordinate factor is the CFS form

    s_w = kappa_w + sigma_w / (alpha_w + j w eps0),   w in {x, y}

implemented in the time domain as a recursive convolution on auxiliary
psi variables:

    psi^n = b psi^{n-1} + c (spatial difference)^n
    b = exp(-(sigma/kappa + alpha) dt/eps0)
    c = sigma (b - 1) / (kappa (sigma + kappa alpha))

Grading (graded into the layer of `thickness` cells; rho = 0 at the
interface, rho = 1 at the outer edge):

    sigma(rho) = sigma_max rho^m
    kappa(rho) = 1 + (kappa_max - 1) rho^m
    alpha(rho) = alpha_max (1 - rho)^ma   (max at the interface — reversed)

with sigma_max = sigma_factor * 0.8 (m+1) / (eta0 dx sqrt(eps_r_bg)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp

from gradenna.constants import EPS0, ETA0


@dataclass(frozen=True)
class CPMLSpec:
    """CPML parameters (defaults follow the Taflove & Hagness reference code)."""

    thickness: int = 10  # cells per side; 0 disables the PML (PEC box)
    m: float = 3.0  # polynomial grading order for sigma and kappa
    ma: float = 1.0  # grading order for alpha
    kappa_max: float = 5.0
    alpha_max: float = 0.0  # CFS term [S/m]; see alpha_max_for_fmin
    sigma_factor: float = 0.75  # sigma_max as a fraction of sigma_opt


def alpha_max_for_fmin(f_min: float) -> float:
    """CFS alpha_max so the transition frequency sits at the band lower edge."""
    return 2.0 * math.pi * f_min * EPS0


class AxisCoefficients(NamedTuple):
    """Per-position CPML tables along one axis: b, c, and 1/kappa."""

    b: jnp.ndarray
    c: jnp.ndarray
    inv_kappa: jnp.ndarray


class PsiSlabs(NamedTuple):
    """One CPML psi variable stored as a low/high PML slab pair.

    psi_w (the auxiliary variable of the stretched axis w) is identically
    zero outside the PML because c_w = 0 there, so only the two slabs of
    `thickness` samples at each end of axis w need storage. With full
    arrays the psi set adds ~133% to the 2D scan carry (and ~200% in 3D);
    storing only the slabs costs a few percent instead — which is what
    keeps the reverse-mode AD tape affordable on a GPU.
    """

    lo: jnp.ndarray
    hi: jnp.ndarray


class SlabCoefficients(NamedTuple):
    """b/c tables restricted to the two PML slabs of one stretched axis,
    reshaped for broadcasting against the field arrays."""

    b_lo: jnp.ndarray
    c_lo: jnp.ndarray
    b_hi: jnp.ndarray
    c_hi: jnp.ndarray


def slab_slices(n: int, thickness: int) -> tuple[slice, slice]:
    """Static low/high PML slab slices along an axis with `n` sample positions.

    Outside these two slabs c_w = 0 and psi_w stays identically zero, so
    psi values and their b/c tables are only kept on `thickness` samples
    per side. With
    ``thickness=0`` (PEC box) both slices are empty and every slab
    operation degenerates to a no-op on zero-size arrays.
    """
    if not 0 <= 2 * thickness <= n:
        raise ValueError(f"PML slab thickness {thickness} too large for axis length {n}")
    return slice(0, thickness), slice(n - thickness, n)


def slab_coefficients(
    b: jnp.ndarray,
    c: jnp.ndarray,
    slabs: tuple[slice, slice],
    axis: int,
    ndim: int,
) -> SlabCoefficients:
    """Slice 1D b/c tables to the two PML slabs and broadcast to `ndim` dims.

    `b`/`c` must already be sampled at the positions of the psi array along
    its stretched axis (e.g. pre-sliced to the PEC interior for E-type psi).
    """
    def expand(a: jnp.ndarray) -> jnp.ndarray:
        shape = [1] * ndim
        shape[axis] = a.shape[0]
        return a.reshape(shape)

    lo, hi = slabs
    return SlabCoefficients(expand(b[lo]), expand(c[lo]), expand(b[hi]), expand(c[hi]))


def psi_step(
    psi: PsiSlabs,
    diff: jnp.ndarray,
    coeffs: SlabCoefficients,
    slabs: tuple[slice, slice],
    axis: int,
    inv_kappa: jnp.ndarray,
) -> tuple[PsiSlabs, jnp.ndarray]:
    """One CPML recursion on slab-stored psi (note 14 Sec. 5.3 / 7.2).

    Updates psi^n = b psi^{n-1} + c diff^n on the two PML slabs of the
    stretched axis and returns ``(psi_new, diff / kappa + psi_new)``. The
    full-size term is assembled as concatenate(lo-slab, middle, hi-slab)
    along the stretched axis; outside the slabs psi = 0 and kappa = 1
    *exactly* (rho = 0 in the grading), so the middle block is the plain
    spatial difference, untouched. Benchmarked on CPU against (a)
    scatter-adding the slabs into ``diff * inv_kappa`` and (b) in-place
    static-slice writes into `diff`: the concatenate fuses with the
    producers of all three regions and was the fastest (and the only
    variant at parity with the old full-array psi arithmetic).
    """
    lo, hi = slabs
    if lo.stop == lo.start:  # thickness=0 (PEC box): kappa = 1 everywhere
        return psi, diff
    idx = lambda s: (slice(None),) * axis + (s,)  # noqa: E731
    mid = slice(lo.stop, hi.start)
    psi_new = PsiSlabs(
        lo=coeffs.b_lo * psi.lo + coeffs.c_lo * diff[idx(lo)],
        hi=coeffs.b_hi * psi.hi + coeffs.c_hi * diff[idx(hi)],
    )
    term = jnp.concatenate(
        [
            diff[idx(lo)] * inv_kappa[idx(lo)] + psi_new.lo,
            diff[idx(mid)],
            diff[idx(hi)] * inv_kappa[idx(hi)] + psi_new.hi,
        ],
        axis=axis,
    )
    return psi_new, term


def axis_coefficients(
    n: int,
    delta: float,
    dt: float,
    spec: CPMLSpec,
    *,
    half: bool,
    eps_r_bg: float = 1.0,
    dtype=None,
) -> AxisCoefficients:
    """CPML tables for one axis.

    `n` is the number of integer (Ez) points along the axis. With
    ``half=False`` the tables are evaluated at integer positions (length n,
    for the E-field psi); with ``half=True`` at i+1/2 (length n-1, for the
    H-field psi). Outside the layer sigma = 0 and c = 0, so psi stays
    identically zero and the update reduces exactly to the plain Yee scheme.

    ``eps_r_bg`` is a manual override of the background relative permittivity
    used to scale ``sigma_max`` when the medium adjacent to the PML is a
    dielectric rather than vacuum; no in-tree benchmark exercises it.
    """
    npml = spec.thickness
    float_dtype = dtype if dtype is not None else jnp.result_type(float)
    if half:
        pos = jnp.arange(n - 1, dtype=float_dtype) + 0.5
    else:
        pos = jnp.arange(n, dtype=float_dtype)

    if npml == 0:
        zeros = jnp.zeros_like(pos)
        return AxisCoefficients(b=zeros, c=zeros, inv_kappa=jnp.ones_like(pos))

    depth_left = (npml - pos) / npml
    depth_right = (pos - (n - 1 - npml)) / npml
    rho = jnp.clip(jnp.maximum(depth_left, depth_right), 0.0, 1.0)

    sigma_max = spec.sigma_factor * 0.8 * (spec.m + 1.0) / (ETA0 * delta * math.sqrt(eps_r_bg))
    sigma = sigma_max * rho**spec.m
    kappa = 1.0 + (spec.kappa_max - 1.0) * rho**spec.m
    alpha = jnp.where(rho > 0.0, spec.alpha_max * (1.0 - rho) ** spec.ma, 0.0)

    b = jnp.exp(-(sigma / kappa + alpha) * dt / EPS0)
    denom = sigma + kappa * alpha
    safe_denom = jnp.where(denom > 0.0, denom, 1.0)
    c = jnp.where(denom > 0.0, sigma * (b - 1.0) / (kappa * safe_denom), 0.0)

    return AxisCoefficients(b=b, c=c, inv_kappa=1.0 / kappa)
