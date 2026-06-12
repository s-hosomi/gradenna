"""Metal/material interpolations for density-based topology optimization.

Implements the metal representation fixed in docs/research/00-summary.md
(design decision 2) and 05-metal-representation.md:

- :func:`sigma_from_density`: differentiable log-sigma interpolation between
  an air-like floor and a response-saturating cap (~1e5 S/m), the standard
  conductivity parameterization of the Hassan/Wadbro/Berggren line of FDTD
  antenna topology optimization.
- :func:`sheet_conductivity`: 1-cell-thick conductive-sheet surrogate for
  copper foil in the 3D solver (Phase 4), sigma_eff = sigma * t / dz.

After the final binarization, designs should be re-verified with the real
copper conductivity (or PEC).
"""

from __future__ import annotations

import math

import jax.numpy as jnp

__all__ = [
    "sheet_conductivity",
    "sigma_from_density",
]


def sigma_from_density(rho, sigma_min: float = 1e-4, sigma_max: float = 1e5):
    """Log-sigma metal interpolation (note 00 decision 2 / note 05).

        sigma(rho) = 10 ** ( log10(sigma_min)
                             + rho * (log10(sigma_max) - log10(sigma_min)) )

    so rho = 0 gives the air-like floor ``sigma_min`` and rho = 1 the
    saturated metal ``sigma_max`` (beyond ~1e5 S/m the FDTD response no
    longer changes, so a higher cap only flattens the gradient).
    Differentiable in ``rho`` (smooth and strictly positive).

    Args:
        rho: design density in [0, 1] (any shape).
        sigma_min: conductivity at rho = 0 [S/m].
        sigma_max: conductivity at rho = 1 [S/m].

    Returns:
        Conductivity [S/m] with the shape of ``rho``.
    """
    if not 0.0 < sigma_min < sigma_max:
        raise ValueError(f"need 0 < sigma_min < sigma_max, got [{sigma_min}, {sigma_max}]")
    lo = math.log10(sigma_min)
    hi = math.log10(sigma_max)
    return 10.0 ** ((hi - lo) * jnp.asarray(rho) + lo)


def sheet_conductivity(sigma_bulk: float, thickness: float, dz: float) -> float:
    """Equivalent conductivity of a thin sheet smeared over one cell.

    A conductor of bulk conductivity ``sigma_bulk`` and physical thickness
    ``thickness`` (< dz) represented as a 1-cell-thick sheet must preserve
    the sheet conductance sigma * t, hence (Phase 4 conductive sheet,
    note 00 decision 2)

        sigma_eff = sigma_bulk * thickness / dz.

    Args:
        sigma_bulk: bulk conductivity [S/m] (e.g. copper, 5.8e7).
        thickness: physical foil thickness [m] (e.g. 35e-6 for 1 oz copper).
        dz: vertical cell size of the grid [m].

    Returns:
        Effective cell conductivity [S/m].
    """
    if thickness <= 0.0 or dz <= 0.0:
        raise ValueError("thickness and dz must be positive")
    return sigma_bulk * thickness / dz
