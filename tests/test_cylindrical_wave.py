"""B1: analytic validation against the cylindrical wave of a line current.

An infinite z-directed current filament I e^{j w t} in vacuum radiates

    Ez(rho) = -(k eta I / 4) H0^(2)(k rho)        (Harrington Sec. 5-6)

The DFT kernel exp(-j w t) extracts the e^{+j w t} phasor, so the outward
wave maps onto the Hankel function of the second kind.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import hankel2

from gradenna import C0, MU0, CPMLSpec, Grid2D, alpha_max_for_fmin, modulated_gaussian, simulate_tm

F0 = 7.5e9  # lambda0 ~ 40 mm


def _phasors(dx: float, n_steps: int, radii_m):
    """Run a centered line source and return (Ez phasors at radii, I phasor)."""
    lam = C0 / F0
    n = int(round(0.36 / dx))  # 360 mm square domain
    grid = Grid2D(nx=n, ny=n, dx=dx, dy=dx)
    c = n // 2
    tau = 10.0 / (2 * np.pi * F0)  # sigma_f = f0/10: narrowband around f0
    t0 = 6 * tau
    t = (jnp.arange(n_steps) + 0.5) * grid.dt
    current = modulated_gaussian(t, f0=F0, t0=t0, tau=tau)

    probes = tuple((c + int(round(r / dx)), c) for r in radii_m)
    cpml = CPMLSpec(thickness=10, alpha_max=alpha_max_for_fmin(F0 / 2))
    res = simulate_tm(grid, source_ij=(c, c), source_current=current, probe_ij=probes, cpml=cpml)

    t_e = (jnp.arange(n_steps) + 1.0) * grid.dt  # Ez^{n+1} lives at (n+1) dt
    kern_e = jnp.exp(-2j * jnp.pi * F0 * t_e) * grid.dt
    ez_hat = (res.probe_ez * kern_e[:, None]).sum(axis=0)
    kern_j = jnp.exp(-2j * jnp.pi * F0 * t) * grid.dt
    i_hat = (current * kern_j).sum()
    assert lam / dx >= 19  # ~lambda/20 sampling
    return np.asarray(ez_hat), complex(i_hat)


def _exact(radii_m):
    k = 2 * np.pi * F0 / C0
    return hankel2(0, k * np.asarray(radii_m))


def _snapped(radii_m, dx):
    return [int(round(r / dx)) * dx for r in radii_m]


def test_radial_profile_matches_hankel():
    """Normalized profile Ez(rho)/Ez(rho_ref) must follow H0^(2)(k rho)."""
    dx = 2e-3
    radii = _snapped(np.linspace(0.030, 0.100, 8), dx)  # 15..50 cells from source
    ez_hat, _ = _phasors(dx, n_steps=1400, radii_m=radii)
    exact = _exact(radii)
    g = ez_hat / ez_hat[0]
    h = exact / exact[0]
    rel = np.abs(g - h) / np.abs(h)
    assert rel.max() < 0.03, f"profile error {rel.max():.4f}"


def test_absolute_amplitude():
    """|Ez| must match -(w mu I / 4) H0^(2)(k rho) within 10 %."""
    dx = 2e-3
    radii = _snapped([0.060], dx)
    ez_hat, i_hat = _phasors(dx, n_steps=1400, radii_m=radii)
    expected = -(2 * np.pi * F0 * MU0 / 4.0) * i_hat * _exact(radii)
    ratio = np.abs(ez_hat[0]) / np.abs(expected[0])
    assert 0.9 < ratio < 1.1, f"amplitude ratio {ratio:.4f}"


@pytest.mark.slow
def test_second_order_grid_convergence():
    """Observed order p = log2(e(dx)/e(dx/2)) must be ~2 at fixed Courant S."""
    radii = _snapped(np.linspace(0.030, 0.100, 8), 2e-3)
    errs = []
    for dx, n_steps in ((2e-3, 1400), (1e-3, 2800)):
        ez_hat, _ = _phasors(dx, n_steps, radii)
        exact = _exact(radii)
        g = ez_hat / ez_hat[0]
        h = exact / exact[0]
        errs.append(np.linalg.norm(g - h) / np.linalg.norm(h))
    p = np.log2(errs[0] / errs[1])
    assert 1.5 <= p <= 2.5, f"observed order {p:.2f}, errors {errs}"
