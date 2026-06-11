"""gradenna — differentiable FDTD antenna inverse design in JAX."""

from gradenna.constants import C0, EPS0, ETA0, MU0
from gradenna.cpml import CPMLSpec, alpha_max_for_fmin
from gradenna.fdtd2d import SimResult, field_energy, simulate_tm
from gradenna.grid import Grid2D
from gradenna.sources import gaussian, gaussian_derivative, modulated_gaussian

__all__ = [
    "C0",
    "EPS0",
    "ETA0",
    "MU0",
    "CPMLSpec",
    "Grid2D",
    "SimResult",
    "alpha_max_for_fmin",
    "field_energy",
    "gaussian",
    "gaussian_derivative",
    "modulated_gaussian",
    "simulate_tm",
]
