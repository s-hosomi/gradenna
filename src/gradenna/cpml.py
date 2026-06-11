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
