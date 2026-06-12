"""Phase 2 acceptance tests: lumped RVS port, S11 extraction, DFT monitor.

References: the RVS update, Kurokawa power waves and band excitation, plus
the 2D line-current radiation impedance

    Ez(rho) = -(w mu I / 4) H0^(2)(k rho)   =>   Re(Zin) = w mu0 / 4 [ohm/m]

(rho -> 0 limit: J0(0) = 1 fixes the real part; the reactance diverges
logarithmically and is not tested).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from gradenna import (
    MU0,
    CPMLSpec,
    Grid2D,
    alpha_max_for_fmin,
    poynting_flux_box_2d,
    simulate_tm,
)
from gradenna.fdtd2d import Port
from gradenna.sparams import (
    gaussian_pulse_for_band,
    half_step_dft,
    incident_voltage,
    port_dft,
    s11_power_wave,
)

DX = 2e-3
F_MIN, F_MAX = 1.5e9, 4.5e9  # -20 dB band of the excitation
EVAL_FREQS = (2.5e9, 2.75e9, 3.0e9, 3.25e9, 3.5e9)  # central part of the band
RS = 50.0


@pytest.fixture(scope="module")
def radiation_run():
    """RVS port radiating into free space, with full-grid DFT monitors."""
    n = 120  # 240 mm box, ~1 wavelength from port to PML at 3 GHz
    grid = Grid2D(nx=n, ny=n, dx=DX, dy=DX)
    pulse = gaussian_pulse_for_band(F_MIN, F_MAX)
    n_steps = 1400  # pulse is ~410 steps; the rest is ring-down
    t = (jnp.arange(n_steps) + 0.5) * grid.dt
    vs = pulse(t)
    c = n // 2
    cpml = CPMLSpec(thickness=10, alpha_max=alpha_max_for_fmin(F_MIN))
    res = simulate_tm(
        grid,
        ports=(Port(ij=(c, c), resistance=RS, voltage=vs),),
        dft_freqs=EVAL_FREQS,
        cpml=cpml,
    )
    return grid, res, vs


def test_radiation_resistance_matches_line_current_theory(radiation_run):
    """Re(Zin) of a free-space line port must be w mu0/4 within 5%."""
    grid, res, _ = radiation_run
    v_hat, i_hat = port_dft(res.port_v[:, 0], res.port_i[:, 0], grid.dt, EVAL_FREQS)
    zin = np.asarray(v_hat / i_hat)
    f = np.asarray(EVAL_FREQS)
    expected = 2.0 * np.pi * f * MU0 / 4.0
    rel = np.abs(zin.real - expected) / expected
    assert rel.max() < 0.05, f"Re(Zin) rel. error {rel}"
    # The reactance must be inductive (log-divergent positive), sanity only.
    assert np.all(zin.imag > 0.0)


def test_power_wave_matches_analytic_incident_wave(radiation_run):
    """a*sqrt(Z0) = (V+Z0 I)/2 must equal V_inc = Vs/2 (note 12 Sec. 3.1b).

    The only discrepancy is the gap displacement current in the Ampere loop,
    which is ~1e-5 relative here; this is a sign/scaling cross-check of the
    whole port pipeline.
    """
    grid, res, vs = radiation_run
    v_hat, i_hat = port_dft(res.port_v[:, 0], res.port_i[:, 0], grid.dt, EVAL_FREQS)
    vs_hat = half_step_dft(vs, grid.dt, EVAL_FREQS)
    a_v = 0.5 * (v_hat + RS * i_hat)
    v_inc = incident_voltage(vs_hat)
    rel = np.abs(np.asarray(a_v - v_inc)) / np.abs(np.asarray(v_inc))
    assert rel.max() < 1e-3, f"incident-wave mismatch {rel}"


def test_port_power_equals_poynting_flux(radiation_run):
    """1/2 Re(V I*) at the port must match the radiated flux within 3%."""
    grid, res, _ = radiation_run
    v_hat, i_hat = port_dft(res.port_v[:, 0], res.port_i[:, 0], grid.dt, EVAL_FREQS)
    p_port = 0.5 * np.real(np.asarray(v_hat) * np.conj(np.asarray(i_hat)))
    c = grid.nx // 2
    k = 20  # contour 20 cells from the port, well inside the CPML
    p_flux = np.asarray(
        poynting_flux_box_2d(
            res.dft_ez, res.dft_hx, res.dft_hy, grid, (c - k, c + k - 1, c - k, c + k - 1)
        )
    )
    assert np.all(p_port > 0.0)
    rel = np.abs(p_flux - p_port) / p_port
    assert rel.max() < 0.03, f"per-frequency energy-balance error {rel}"
    # Band-integrated balance (trapezoid over the eval frequencies).
    f = np.asarray(EVAL_FREQS)
    band = np.trapezoid(p_port, f)
    band_flux = np.trapezoid(p_flux, f)
    assert abs(band_flux - band) / band < 0.03


def test_pec_enclosed_port_is_totally_reflecting():
    """|S11| ~ 1 for a port enclosed by lossless PEC walls (within 2%).

    With cpml thickness 0 the outer Ez ring is PEC, so the port drives a
    lossless cavity: Zin is purely reactive and |S11| = 1 at any frequency
    once the fields have rung down through Rs.
    """
    n = 17  # 32 mm PEC box; first cavity resonance f11 ~ 6.6 GHz
    grid = Grid2D(nx=n, ny=n, dx=DX, dy=DX)
    pulse = gaussian_pulse_for_band(2e9, 8e9)
    n_steps = 30000  # ~140 ns >> L/Rs ring-down of the loaded cavity
    t = (jnp.arange(n_steps) + 0.5) * grid.dt
    res = simulate_tm(
        grid,
        ports=(Port(ij=(n // 2, n // 2), resistance=RS, voltage=pulse(t)),),
        cpml=CPMLSpec(thickness=0),
    )
    freqs = (3e9, 5e9, 6.6e9, 7.5e9)
    v_hat, i_hat = port_dft(res.port_v[:, 0], res.port_i[:, 0], grid.dt, freqs)
    s11 = np.abs(np.asarray(s11_power_wave(v_hat, i_hat, RS)))
    assert np.abs(s11 - 1.0).max() < 0.02, f"|S11| = {s11}"


# --- differentiability ------------------------------------------------------

GRID_G = Grid2D(nx=40, ny=40, dx=2e-3, dy=2e-3)
REGION_G = (slice(20, 28), slice(16, 24))  # 8x8 patch in front of the port
F_EVAL = 5e9
N_STEPS_G = 400


def _s11_loss(eps_patch):
    """|S11(F_EVAL)|^2 of an RVS port facing a dielectric patch."""
    eps_r = jnp.ones(GRID_G.shape).at[REGION_G].set(eps_patch)
    pulse = gaussian_pulse_for_band(1e9, 9e9)
    t = (jnp.arange(N_STEPS_G) + 0.5) * GRID_G.dt
    res = simulate_tm(
        GRID_G,
        ports=(Port(ij=(14, 20), resistance=RS, voltage=pulse(t)),),
        eps_r=eps_r,
        cpml=CPMLSpec(thickness=8, alpha_max=alpha_max_for_fmin(1e9)),
    )
    v_hat, i_hat = port_dft(res.port_v[:, 0], res.port_i[:, 0], GRID_G.dt, F_EVAL)
    s = s11_power_wave(v_hat, i_hat, RS)[0]
    return s.real**2 + s.imag**2  # |S11|^2, smooth at zero (note 12 Sec. 7.2)


@pytest.mark.parametrize("seed", [0, 1])
def test_s11_grad_matches_directional_finite_difference(seed):
    eps0 = 2.0 * jnp.ones((8, 8))
    grad = jax.grad(_s11_loss)(eps0)
    g_flat, _ = ravel_pytree(grad)
    assert bool(jnp.all(jnp.isfinite(g_flat)))
    assert float(jnp.abs(g_flat).max()) > 0.0

    rng = np.random.default_rng(seed)
    v = rng.normal(size=g_flat.shape)
    v = jnp.asarray(v / np.linalg.norm(v))
    d_ad = float(jnp.vdot(g_flat, v))

    flat0, unravel = ravel_pytree(eps0)
    scale = float(jnp.linalg.norm(flat0))
    errs = []
    for h in (3e-7, 1e-6, 3e-6):
        hh = h * scale
        fp = float(_s11_loss(unravel(flat0 + hh * v)))
        fm = float(_s11_loss(unravel(flat0 - hh * v)))
        d_fd = (fp - fm) / (2.0 * hh)
        errs.append(abs(d_ad - d_fd) / max(abs(d_ad), abs(d_fd)))
    assert min(errs) <= 1e-4, f"AD vs FD relative error {min(errs):.2e}"
