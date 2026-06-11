"""Excitation waveforms.

All waveforms take a time array [s] and return dimensionless amplitudes
of order one; scale by the desired source current outside.
"""

from __future__ import annotations

import jax.numpy as jnp


def gaussian(t, t0: float, tau: float):
    """Gaussian pulse exp(-(t-t0)^2 / (2 tau^2)). Contains DC."""
    u = (t - t0) / tau
    return jnp.exp(-0.5 * u**2)


def gaussian_derivative(t, t0: float, tau: float):
    """Differentiated Gaussian (no DC), peak amplitude ~1.

    Spectrum peaks at f_p = 1/(2 pi tau).
    """
    u = (t - t0) / tau
    return -u * jnp.exp(0.5 * (1.0 - u**2))


def modulated_gaussian(t, f0: float, t0: float, tau: float):
    """Sine carrier at f0 under a Gaussian envelope (bandwidth sigma_f = 1/(2 pi tau))."""
    u = (t - t0) / tau
    return jnp.sin(2.0 * jnp.pi * f0 * (t - t0)) * jnp.exp(-0.5 * u**2)
