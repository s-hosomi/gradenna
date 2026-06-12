"""S-parameter extraction for lumped RVS ports.

The recipe is:

- exact-phase DFT of the (n+1/2) dt port samples — the half-step
  sample-time offset is folded into the DFT kernel, so no time-domain
  averaging error is introduced;
- Kurokawa power-wave S11 (Kurokawa 1965), identical to the openEMS
  ``calcLumpedPort`` reflection ``uf_ref/uf_inc`` and to
  (Zin - Z0)/(Zin + Z0);
- the analytic incident wave V_inc = Vs/2 of a matched RVS port, the
  zero-cost reference for cross-checks;
- the openEMS-compatible modulated-Gaussian band excitation.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp


def half_step_dft(x_t, dt: float, freqs):
    """DFT of a series sampled at t_n = (n + 1/2) dt with exact phases.

        X(f) = dt * sum_n x[n] exp(-i 2 pi f (n + 1/2) dt)

    This is note 12 Sec. 2.2 method 2: the half-step offset of the sample
    times (port V/I and source waveforms in `simulate_tm` are all recorded
    at (n+1/2) dt) enters as the exact phase factor, with no approximation.

    Args:
        x_t: real series, shape (n_steps,) or (n_steps, k).
        dt: time step [s].
        freqs: scalar or (n_freq,) frequencies [Hz].

    Returns:
        Complex spectrum, shape (n_freq,) or (n_freq, k).
    """
    x_t = jnp.asarray(x_t)
    f = jnp.atleast_1d(jnp.asarray(freqs))
    n = jnp.arange(x_t.shape[0])
    t = (n + 0.5) * dt
    kernel = jnp.exp(-2j * jnp.pi * f[:, None] * t[None, :]) * dt  # (n_freq, n_steps)
    return jnp.tensordot(kernel, x_t, axes=(1, 0))


def port_dft(v_t, i_t, dt: float, freqs):
    """Spectra (V_hat, I_hat) of port voltage/current time series.

    Both series produced by `simulate_tm` are time-aligned at t = (n+1/2) dt
    (V is the average of Ez^n and Ez^{n+1}, I is the Ampere loop of
    H^{n+1/2}), so a single exact-phase kernel applies to both
    (note 12 Sec. 2.2).

    Args:
        v_t, i_t: shape (n_steps,) or (n_steps, n_ports).
        dt: time step [s].
        freqs: scalar or (n_freq,) frequencies [Hz].

    Returns:
        (v_hat, i_hat), each (n_freq,) or (n_freq, n_ports), complex.
    """
    return half_step_dft(v_t, dt, freqs), half_step_dft(i_t, dt, freqs)


def s11_power_wave(v_hat, i_hat, z0=50.0):
    """Power-wave reflection coefficient (Kurokawa 1965; note 12 Sec. 3.1c).

        S11 = b/a = (V_hat - Z0 I_hat) / (V_hat + Z0 I_hat)

    Algebraically identical to (Zin - Z0)/(Zin + Z0) and to the openEMS
    lumped-port ``uf_ref/uf_inc`` (note 12 Sec. 3.2/3.4), but with a single
    division whose denominator tracks the source spectrum, so it is the
    numerically safest form (and the recommended one for AD losses; use
    ``s.real**2 + s.imag**2`` rather than ``abs(s)`` near zero).
    """
    return (v_hat - z0 * i_hat) / (v_hat + z0 * i_hat)


def incident_voltage(vs_hat):
    """Analytic incident wave of a matched RVS port (note 12 Sec. 3.1b).

    For a lumped Thevenin source Vs with internal resistance Rs = Z0, the
    voltage delivered to a matched load is V_inc = Vs/2; no reference
    simulation is needed. ``vs_hat`` is the spectrum of the source waveform
    (e.g. ``half_step_dft(vs_t, dt, freqs)``, since Vs is sampled at
    (n+1/2) dt in `simulate_tm`).
    """
    return vs_hat / 2.0


class BandPulse(NamedTuple):
    """openEMS-compatible modulated Gaussian excitation (note 12 Sec. 4.1).

        s(t) = cos(2 pi f0 (t - t0)) * exp(-(t - t0)^2 / (2 sigma^2))

    with sigma = 3/(2 sqrt(2) pi fc) and t0 = 9/(2 pi fc) = 3 sqrt(2) sigma,
    the standard-form rewrite of openEMS ``SetupGaussianPulse(f0, fc)``.
    The band edges f0 -/+ fc sit at amplitude exp(-9/4) ~ -20 dB, and the
    turn-on truncation at t = 0 is exp(-9) ~ -78 dB.
    """

    f0: float
    fc: float
    sigma: float
    t0: float

    @property
    def duration(self) -> float:
        """Full pulse length 2 t0 (openEMS allocates ceil(2 t0 / dt) steps)."""
        return 2.0 * self.t0

    def __call__(self, t):
        """Evaluate the waveform at time(s) t [s] (dimensionless, order 1)."""
        u = t - self.t0
        return jnp.cos(2.0 * jnp.pi * self.f0 * u) * jnp.exp(-(u**2) / (2.0 * self.sigma**2))


def gaussian_pulse_for_band(f_min: float, f_max: float) -> BandPulse:
    """Design a modulated Gaussian covering [f_min, f_max] (note 12 Sec. 4.1).

    Sets f0 = (f_min + f_max)/2 and fc = (f_max - f_min)/2, so the requested
    band edges are the ~-20 dB points. f_min > 0 guarantees f0 >= fc, which
    keeps the DC content suppressed (note 12 Sec. 4.2).
    """
    if not 0.0 < f_min < f_max:
        raise ValueError(f"need 0 < f_min < f_max, got [{f_min}, {f_max}]")
    f0 = 0.5 * (f_min + f_max)
    fc = 0.5 * (f_max - f_min)
    sigma = 3.0 / (2.0 * math.sqrt(2.0) * math.pi * fc)
    t0 = 9.0 / (2.0 * math.pi * fc)
    return BandPulse(f0=f0, fc=fc, sigma=sigma, t0=t0)
