"""Phase 5 regression tests: NTFF directivity and multiband optimization.

Scaled-down versions of examples/optimize_directivity.py and
examples/optimize_multiband.py on an 80x80 grid (dx = 2 mm) with a 16x16
design region around an embedded RVS port. The frequencies are scaled up
(lambda = 16-24 cells) so the design region stays larger than lambda/2,
the resonant-feature threshold found in the Phase 3 demo.

- Directivity: a few Adam iterations on the NTFF realized-gain proxy
  2 pi U(phi=0) / P_avail = D(0) * P_rad / P_avail must raise D(phi=0)
  by >= 1.5x from the uniform rho = 0.5 start, with finite gradients.
- Multiband: the soft-min (logsumexp) of the radiated fractions at two
  bands must improve both bands from the start.

Everything runs in float64 (conftest sets JAX_ENABLE_X64): the gray
starting design is a weak absorber whose fluxes underflow in float32.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from gradenna import (
    C0,
    CPMLSpec,
    Grid2D,
    Port,
    alpha_max_for_fmin,
    gaussian_pulse_for_band,
    half_step_dft,
    poynting_flux_box_2d,
    sigma_from_density,
    simulate_tm,
)
from gradenna.ntff import directivity_2d, ntff_2d, radiated_power_2d
from gradenna.topopt import DesignTransform

DX = 2e-3
N = 80
RS = 50.0
PORT_IJ = (40, 40)
DESIGN = (slice(32, 48), slice(32, 48))  # 16x16 cells = 32 x 32 mm
NTFF_MARGIN = 12  # contour: CPML (8) + 4 cells
CPML = lambda fmin: CPMLSpec(thickness=8, alpha_max=alpha_max_for_fmin(fmin))  # noqa: E731

# Log-sigma interpolation (note 05), sigma in [1e-4, 1e5] S/m.
SIGMA_MIN, SIGMA_MAX = 1e-4, 1e5


def _design_sigma(rho, grid):
    """Embed the design density as conductivity; the port cell stays clear."""
    n = DESIGN[0].stop - DESIGN[0].start
    mask = np.ones((n, n), bool)
    mask[PORT_IJ[0] - DESIGN[0].start, PORT_IJ[1] - DESIGN[1].start] = False
    sig = jnp.where(jnp.asarray(mask), sigma_from_density(rho, SIGMA_MIN, SIGMA_MAX), 0.0)
    return jnp.zeros(grid.shape).at[DESIGN].set(sig)


# ---------------------------------------------------------------------------
# Directivity optimization (reduced examples/optimize_directivity.py)
# ---------------------------------------------------------------------------


def test_directivity_optimization_improves():
    """D(phi=0) must improve >= 1.5x in a short fixed run; gradients finite."""
    grid = Grid2D(nx=N, ny=N, dx=DX, dy=DX)
    f0 = C0 / (20.0 * DX)  # lambda = 20 cells: design region is 0.8 lambda
    pulse = gaussian_pulse_for_band(0.6 * f0, 1.4 * f0)
    n_steps = 900
    t = (jnp.arange(n_steps) + 0.5) * grid.dt
    vs = pulse(t)
    p_avail = jnp.abs(half_step_dft(vs, grid.dt, f0)[0]) ** 2 / (8.0 * RS)
    angles = jnp.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)
    transform = DesignTransform(radius_cells=1.5)
    beta = 8.0  # fixed (no continuation in the reduced run)

    def metrics(theta):
        rho = transform(theta, beta)
        res = simulate_tm(
            grid,
            ports=(Port(ij=PORT_IJ, resistance=RS, voltage=vs),),
            sigma=_design_sigma(rho, grid),
            dft_freqs=(f0,),
            cpml=CPML(0.5 * f0),
        )
        e_far = ntff_2d(
            res.dft_ez, res.dft_hx, res.dft_hy, grid, NTFF_MARGIN, (f0,), angles
        )
        d0 = directivity_2d(e_far, angles)[0, 0]  # D toward phi = 0 (+x)
        p_rad = radiated_power_2d(e_far, angles)[0]
        return d0, p_rad

    def loss_fn(theta):
        d0, p_rad = metrics(theta)
        # Realized-gain proxy: D(0) * e_rad = 2 pi U(0) / P_avail.
        return -d0 * p_rad / p_avail, d0

    @jax.jit
    def step(theta, opt_state):
        (loss, d0), grads = jax.value_and_grad(loss_fn, has_aux=True)(theta)
        updates, opt_state = opt.update(grads, opt_state, theta)
        return optax.apply_updates(theta, updates), opt_state, loss, d0, grads

    n_des = DESIGN[0].stop - DESIGN[0].start
    theta = jnp.zeros((n_des, n_des))  # sigmoid(0) = 0.5 uniform gray start
    opt = optax.adam(0.2)
    opt_state = opt.init(theta)

    d0_init = None
    d0_final = None
    for i in range(30):
        theta, opt_state, loss, d0, grads = step(theta, opt_state)
        assert bool(jnp.all(jnp.isfinite(grads))), f"non-finite gradient at iter {i}"
        assert math.isfinite(float(loss))
        if d0_init is None:
            d0_init = float(d0)  # D(0) of the uniform start (theta before update)
        d0_final = float(d0)

    assert d0_final >= 1.5 * d0_init, (
        f"D(0) improved only {d0_init:.3f} -> {d0_final:.3f} (< 1.5x)"
    )


# ---------------------------------------------------------------------------
# Multiband optimization (reduced examples/optimize_multiband.py)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_multiband_optimization_improves():
    """Soft-min optimization must improve the radiated fraction in both bands."""
    grid = Grid2D(nx=N, ny=N, dx=DX, dy=DX)
    f_bands = (C0 / (24.0 * DX), C0 / (16.0 * DX))  # ~6.2 and ~9.4 GHz
    pulse = gaussian_pulse_for_band(0.8 * f_bands[0], 1.2 * f_bands[1])
    n_steps = 900
    t = (jnp.arange(n_steps) + 0.5) * grid.dt
    vs = pulse(t)
    p_avail = jnp.abs(half_step_dft(vs, grid.dt, f_bands)) ** 2 / (8.0 * RS)
    flux_box = (10, 69, 10, 69)  # 2 cells outside the CPML interface
    transform = DesignTransform(radius_cells=1.5)
    beta = 8.0
    temp = 0.05  # soft-min temperature

    def fractions(theta):
        rho = transform(theta, beta)
        res = simulate_tm(
            grid,
            ports=(Port(ij=PORT_IJ, resistance=RS, voltage=vs),),
            sigma=_design_sigma(rho, grid),
            dft_freqs=f_bands,
            cpml=CPML(0.7 * f_bands[0]),
        )
        p_rad = poynting_flux_box_2d(res.dft_ez, res.dft_hx, res.dft_hy, grid, flux_box)
        return p_rad / p_avail

    def loss_fn(theta):
        frac = fractions(theta)
        # Smooth min over bands: -softmin(x) = T * logsumexp(-x / T).
        return temp * jax.scipy.special.logsumexp(-frac / temp), frac

    @jax.jit
    def step(theta, opt_state):
        (loss, frac), grads = jax.value_and_grad(loss_fn, has_aux=True)(theta)
        updates, opt_state = opt.update(grads, opt_state, theta)
        return optax.apply_updates(theta, updates), opt_state, frac, grads

    n_des = DESIGN[0].stop - DESIGN[0].start
    theta = jnp.zeros((n_des, n_des))
    opt = optax.adam(0.2)
    opt_state = opt.init(theta)

    frac_init = None
    frac_final = None
    for i in range(20):
        theta, opt_state, frac, grads = step(theta, opt_state)
        assert bool(jnp.all(jnp.isfinite(grads))), f"non-finite gradient at iter {i}"
        if frac_init is None:
            frac_init = np.asarray(frac)  # fractions of the uniform start
        frac_final = np.asarray(frac)

    assert np.all(frac_final >= 1.5 * frac_init), (
        f"radiated fractions {frac_init} -> {frac_final}: not both improved 1.5x"
    )
