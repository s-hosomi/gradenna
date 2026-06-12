"""NTFF acceptance tests (research note 13).

2D: a centered line current must radiate isotropically with the analytic
power P' = omega mu0 |I|^2 / 8 per unit length; a half-wavelength two-element
in-phase array must show the cos^2((pi/2) cos phi) broadside pattern.
3D: a z-directed infinitesimal current element must give D(90 deg) = 1.5,
a sin^2(theta) pattern and P_rad = R_r |I|^2 / 2 with R_r = 80 pi^2 (dz/lambda)^2.
Differentiability: directivity must be jax.grad-able w.r.t. an eps_r patch.
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

from gradenna import (
    C0,
    MU0,
    CPMLSpec,
    Grid2D,
    alpha_max_for_fmin,
    modulated_gaussian,
    simulate_tm,
)
from gradenna.fdtd3d import Grid3D, simulate_3d, time_series_dft
from gradenna.ntff import (
    directivity_2d,
    directivity_3d,
    ntff_2d,
    ntff_3d,
    radiated_power_2d,
    radiated_power_3d,
)

# ---------------------------------------------------------------------------
# 2D setup: dx = 2 mm, lambda = 40 cells (f ~ 3.75 GHz), 150x150 grid
# ---------------------------------------------------------------------------

DX2 = 2e-3
N2 = 150
F2 = C0 / (40.0 * DX2)  # exactly 40 cells per vacuum wavelength
MARGIN2 = 25  # contour: CPML (10) + 15 cells
STEPS2 = 1100
ANGLES2 = jnp.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)


@functools.lru_cache(maxsize=2)
def _run_2d(two_element: bool):
    """Line current(s) at the grid center; returns (grid, current, e_far)."""
    grid = Grid2D(nx=N2, ny=N2, dx=DX2, dy=DX2)
    c = N2 // 2
    tau = 8.0 / (2.0 * np.pi * F2)  # sigma_f = f0/8: narrowband around f0
    t = (jnp.arange(STEPS2) + 0.5) * grid.dt
    current = modulated_gaussian(t, f0=F2, t0=6.0 * tau, tau=tau)
    if two_element:
        # Two in-phase elements, lambda/2 = 20 cells apart along x.
        source_ij = ((c - 10, c), (c + 10, c))
        source_current = jnp.stack([current, current], axis=1)
    else:
        source_ij = (c, c)
        source_current = current
    res = simulate_tm(
        grid,
        source_ij=source_ij,
        source_current=source_current,
        dft_freqs=(F2,),
        cpml=CPMLSpec(thickness=10, alpha_max=alpha_max_for_fmin(F2 / 2.0)),
    )
    e_far = ntff_2d(res.dft_ez, res.dft_hx, res.dft_hy, grid, MARGIN2, (F2,), ANGLES2)
    return grid, current, e_far


def test_line_current_isotropic_pattern():
    """|E_far(phi)| of a single line current must be angle-independent (<= 2%)."""
    _, _, e_far = _run_2d(False)
    mag = np.abs(np.asarray(e_far[0]))
    spread = (mag.max() - mag.min()) / mag.mean()
    assert spread <= 0.02, f"isotropy spread {spread:.4f} > 2%"


def test_line_current_radiated_power():
    """NTFF power must match the analytic P' = omega mu0 |I|^2 / 8 within 3%.

    Both spectra use the same dt-scaled DFT, so the normalization cancels
    in the ratio (module docstring of gradenna.ntff).
    """
    grid, current, e_far = _run_2d(False)
    p_ntff = float(radiated_power_2d(e_far, ANGLES2)[0])
    i_hat = time_series_dft(current, grid.dt, (F2,), t0=0.5 * grid.dt)[0]
    p_exact = 2.0 * np.pi * F2 * MU0 * float(jnp.abs(i_hat)) ** 2 / 8.0
    ratio = p_ntff / p_exact
    assert abs(ratio - 1.0) <= 0.03, f"P_ntff/P_exact = {ratio:.4f}"


def test_two_element_array_pattern():
    """lambda/2 in-phase pair: broadside max, endfire null, cos^2((pi/2)cos phi)."""
    _, _, e_far = _run_2d(True)
    pattern = np.abs(np.asarray(e_far[0])) ** 2
    pattern = pattern / pattern.max()
    phi = np.asarray(ANGLES2)
    analytic = np.cos(0.5 * np.pi * np.cos(phi)) ** 2

    corr = np.corrcoef(pattern, analytic)[0, 1]
    assert corr >= 0.99, f"pattern correlation {corr:.4f} < 0.99"

    # Broadside (phi = 90 or 270 deg) maximum within one angular step.
    peak_phi = phi[int(np.argmax(pattern))]
    d90 = min(abs(peak_phi - np.pi / 2), abs(peak_phi - 3 * np.pi / 2))
    assert d90 <= 2.0 * np.pi / 360 + 1e-12, f"peak at {np.degrees(peak_phi):.1f} deg"

    # Endfire (phi = 0 and 180 deg) nulls.
    assert pattern[0] <= 0.02, f"endfire level {pattern[0]:.4f}"
    assert pattern[180] <= 0.02, f"endfire level {pattern[180]:.4f}"


@functools.lru_cache(maxsize=1)
def _run_2d_endfire():
    """lambda/4-spaced pair along x; the +x element lags by 90 deg."""
    grid = Grid2D(nx=N2, ny=N2, dx=DX2, dy=DX2)
    c = N2 // 2
    tau = 8.0 / (2.0 * np.pi * F2)
    t = (jnp.arange(STEPS2) + 0.5) * grid.dt
    # 90 deg carrier lag = T/4 delay (the tau-long envelope shift is negligible).
    cur_lead = modulated_gaussian(t, f0=F2, t0=6.0 * tau, tau=tau)
    cur_lag = modulated_gaussian(t, f0=F2, t0=6.0 * tau + 0.25 / F2, tau=tau)
    source_ij = ((c - 5, c), (c + 5, c))  # lambda/4 = 10 cells apart
    res = simulate_tm(
        grid,
        source_ij=source_ij,
        source_current=jnp.stack([cur_lead, cur_lag], axis=1),
        dft_freqs=(F2,),
        cpml=CPMLSpec(thickness=10, alpha_max=alpha_max_for_fmin(F2 / 2.0)),
    )
    return ntff_2d(res.dft_ez, res.dft_hx, res.dft_hy, grid, MARGIN2, (F2,), ANGLES2)


def test_endfire_pair_propagation_phase_sign():
    """lambda/4 pair with the +x element 90 deg lagging: cardioid toward +x.

    |AF|^2 = 4 cos^2((pi/4)(cos phi - 1)) peaks at phi = 0 and has a null at
    phi = pi. This discriminates the sign of the NTFF propagation phase
    e^{+jk r'.rhat}: with the opposite sign the pattern flips (peak at
    phi = pi, null at phi = 0), which the broadside/isotropy tests above
    cannot detect (their patterns are symmetric under phi -> phi + pi).
    """
    e_far = _run_2d_endfire()
    pattern = np.abs(np.asarray(e_far[0])) ** 2
    pattern = pattern / pattern.max()
    phi = np.asarray(ANGLES2)

    analytic = np.cos(0.25 * np.pi * (np.cos(phi) - 1.0)) ** 2
    corr = np.corrcoef(pattern, analytic)[0, 1]
    assert corr >= 0.99, f"pattern correlation {corr:.4f} < 0.99"

    # Peak toward +x: the cardioid maximum is 4th-order flat at phi = 0, so
    # the argmax wanders a few degrees; +-15 deg still discriminates the
    # sign flip unambiguously (the flipped pattern peaks at 180 deg).
    peak_phi = phi[int(np.argmax(pattern))]
    d0 = min(peak_phi, 2.0 * np.pi - peak_phi)
    assert d0 <= np.radians(15.0) + 1e-12, f"peak at {np.degrees(peak_phi):.1f} deg"

    # Null toward -x (phi = 180 deg).
    assert pattern[180] <= 0.02, f"back-lobe level {pattern[180]:.4f}"
    # Sanity: strong front-to-back contrast.
    assert pattern[0] >= 0.9, f"forward level {pattern[0]:.4f}"


# ---------------------------------------------------------------------------
# 3D setup: dx = 4 mm, lambda = 25 cells (f ~ 3 GHz), 48^3 grid, 600 steps
# ---------------------------------------------------------------------------

DX3 = 4e-3
N3 = 48
F3 = C0 / (25.0 * DX3)  # exactly 25 cells per vacuum wavelength
MARGIN3 = 14  # box: CPML (10) + 4 cells
STEPS3 = 600
THETAS3 = jnp.linspace(0.0, np.pi, 37)  # index 18 is exactly theta = pi/2
PHIS3 = jnp.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)


@functools.lru_cache(maxsize=1)
def _run_3d():
    """z-directed current element at the grid center; returns (grid, I, e_far)."""
    grid = Grid3D(nx=N3, ny=N3, nz=N3, dx=DX3, dy=DX3, dz=DX3)
    c = N3 // 2
    tau = 6.0 / (2.0 * np.pi * F3)
    t = (jnp.arange(STEPS3) + 0.5) * grid.dt
    current = modulated_gaussian(t, f0=F3, t0=5.0 * tau, tau=tau)
    res = simulate_3d(
        grid,
        source_ijk=(c, c, c),
        source_current=current,
        dft_freqs=jnp.array([F3]),
        cpml=CPMLSpec(thickness=10, alpha_max=alpha_max_for_fmin(F3 / 2.0)),
    )
    e_far = ntff_3d(res.dft, grid, MARGIN3, (F3,), THETAS3, PHIS3)
    return grid, current, e_far


def test_dipole_directivity_3d():
    """D(theta = 90 deg) of an infinitesimal z-dipole must be 1.5 +- 5%."""
    _, _, e_far = _run_3d()
    d = directivity_3d(e_far[..., 0], e_far[..., 1], THETAS3, PHIS3)
    d90 = float(jnp.mean(d[0, 18, :]))  # theta = pi/2, averaged over phi
    assert abs(d90 - 1.5) <= 0.075, f"D(90 deg) = {d90:.4f}"
    # Axial symmetry: D at theta = 90 deg must not depend on phi.
    var = float((jnp.max(d[0, 18, :]) - jnp.min(d[0, 18, :])) / jnp.mean(d[0, 18, :]))
    assert var <= 0.05, f"phi spread at theta=90deg: {var:.4f}"


def test_dipole_pattern_3d():
    """U(theta) must follow sin^2(theta) with correlation >= 0.99."""
    _, _, e_far = _run_3d()
    u = np.asarray(
        jnp.abs(e_far[0, :, :, 0]) ** 2 + jnp.abs(e_far[0, :, :, 1]) ** 2
    ).mean(axis=1)
    u = u / u.max()
    analytic = np.sin(np.asarray(THETAS3)) ** 2
    corr = np.corrcoef(u, analytic)[0, 1]
    assert corr >= 0.99, f"pattern correlation {corr:.5f} < 0.99"


def test_dipole_radiated_power_3d():
    """P_rad must match R_r |I|^2 / 2 with R_r = 80 pi^2 (dz/lambda)^2 within 5%."""
    grid, current, e_far = _run_3d()
    p_ntff = float(radiated_power_3d(e_far[..., 0], e_far[..., 1], THETAS3, PHIS3)[0])
    i_hat = time_series_dft(current, grid.dt, (F3,), t0=0.5 * grid.dt)[0]
    r_r = 80.0 * np.pi**2 * (grid.dz * F3 / C0) ** 2
    p_exact = 0.5 * r_r * float(jnp.abs(i_hat)) ** 2
    ratio = p_ntff / p_exact
    assert abs(ratio - 1.0) <= 0.05, f"P_ntff/P_exact = {ratio:.4f}"


# ---------------------------------------------------------------------------
# Differentiability: jax.grad of a directivity objective w.r.t. an eps patch
# ---------------------------------------------------------------------------


def test_directivity_grad_wrt_eps():
    """d directivity(phi=0) / d eps_r of a dielectric patch: finite and nonzero."""
    grid = Grid2D(nx=60, ny=60, dx=DX2, dy=DX2)
    f0 = C0 / (20.0 * DX2)  # lambda = 20 cells
    c = 30
    margin = 12  # CPML (8) + 4 cells
    angles = jnp.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
    tau = 6.0 / (2.0 * np.pi * f0)
    t = (jnp.arange(400) + 0.5) * grid.dt
    current = modulated_gaussian(t, f0=f0, t0=5.0 * tau, tau=tau)

    def loss(eps_patch):
        eps_r = jnp.ones(grid.shape).at[33:39, 27:33].set(eps_patch)
        res = simulate_tm(
            grid,
            source_ij=(c, c),
            source_current=current,
            eps_r=eps_r,
            dft_freqs=(f0,),
            cpml=CPMLSpec(thickness=8, alpha_max=alpha_max_for_fmin(f0 / 2.0)),
        )
        e_far = ntff_2d(res.dft_ez, res.dft_hx, res.dft_hy, grid, margin, (f0,), angles)
        return directivity_2d(e_far, angles)[0, 0]  # D toward phi = 0 (+x)

    g = jax.grad(loss)(2.0)
    assert math.isfinite(float(g)), f"gradient is not finite: {g}"
    assert abs(float(g)) > 0.0, "gradient is exactly zero"
