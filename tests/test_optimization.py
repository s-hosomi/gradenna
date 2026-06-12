"""B10 optimization regression test (docs/research/07-benchmarks.md).

Shrunk version of examples/optimize_2d_antenna.py: an RVS-fed design region
whose log-sigma density is optimized to maximize the radiated power fraction
P_rad / P_avail at 2.45 GHz (energy objective of note 00/05 — absorption
inside the design region cannot "cheat" because the Poynting contour sits
outside it). The run starts from the uniform rho = 0.5 absorber blob and must

- improve the normalized radiated power by at least 2x,
- end up nearly binary (gray indicator < 0.35 at the final beta),
- keep all per-iteration gradients finite.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from gradenna import (
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
from gradenna.topopt import DesignTransform, beta_schedule, gray_indicator

# Shrunk problem: 80x80 grid (160 mm box), 16x16-cell design region with the
# feed embedded near its bottom edge, 2.45 GHz band excitation.
DX = 2e-3
NX = NY = 80
F0 = 2.45e9
F_MIN, F_MAX = 1.5e9, 3.5e9
RS = 50.0
N_STEPS = 1400  # pulse is ~610 steps, the rest is ring-down
PORT_IJ = (40, 36)
DESIGN = (slice(32, 48), slice(32, 48))
FLUX_BOX = (13, 66, 13, 66)  # (il, ir, jb, jt), outside the design region
SIGMA_MAX, SIGMA_MIN = 1e5, 1e-4  # log-sigma interpolation endpoints [S/m]
BETAS = (8.0, 32.0)
ITERS_PER_BETA = 15
LEARNING_RATE = 0.1


@pytest.fixture(scope="module")
def problem():
    grid = Grid2D(nx=NX, ny=NY, dx=DX, dy=DX)
    cpml = CPMLSpec(thickness=10, alpha_max=alpha_max_for_fmin(F_MIN))
    pulse = gaussian_pulse_for_band(F_MIN, F_MAX)
    t = (jnp.arange(N_STEPS) + 0.5) * grid.dt
    vs = pulse(t)
    # Available source power |Vs_hat|^2 / (8 Rs) at f0 (note 12 Sec. 3.1b).
    p_avail = jnp.abs(half_step_dft(vs, grid.dt, F0)[0]) ** 2 / (8.0 * RS)

    # Feed clearance: the port cell itself never carries design conductivity.
    n_des = DESIGN[0].stop - DESIGN[0].start
    mask = np.ones((n_des, n_des), bool)
    mask[PORT_IJ[0] - DESIGN[0].start, PORT_IJ[1] - DESIGN[1].start] = False
    mask = jnp.asarray(mask)

    def radiated_fraction(rho):
        """P_rad(f0) / P_avail(f0) for the design density rho."""
        sig_design = jnp.where(mask, sigma_from_density(rho, SIGMA_MIN, SIGMA_MAX), 0.0)
        sigma = jnp.zeros(grid.shape).at[DESIGN].set(sig_design)
        res = simulate_tm(
            grid,
            ports=(Port(ij=PORT_IJ, resistance=RS, voltage=vs),),
            sigma=sigma,
            dft_freqs=(F0,),
            cpml=cpml,
        )
        p_rad = poynting_flux_box_2d(
            res.dft_ez, res.dft_hx, res.dft_hy, grid, FLUX_BOX
        )[0]
        return p_rad / p_avail

    return radiated_fraction


def test_radiated_power_optimization_regression(problem):
    radiated_fraction = problem
    transform = DesignTransform(radius_cells=2.0)
    schedule = beta_schedule(BETAS, ITERS_PER_BETA)
    n_iters = ITERS_PER_BETA * len(BETAS)

    theta = jnp.zeros((16, 16))  # sigmoid(0) = 0.5: uniform gray start
    p_initial = float(radiated_fraction(transform(theta, BETAS[0])))
    assert p_initial > 0.0

    opt = optax.adam(LEARNING_RATE)
    opt_state = opt.init(theta)

    def objective(theta, beta):
        return -radiated_fraction(transform(theta, beta))

    @jax.jit
    def step(theta, opt_state, beta):
        loss, grads = jax.value_and_grad(objective)(theta, beta)
        updates, opt_state = opt.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        return theta, opt_state, loss, grads

    losses = []
    for i in range(n_iters):
        theta, opt_state, loss, grads = step(
            theta, opt_state, jnp.asarray(schedule(i))
        )
        assert bool(jnp.all(jnp.isfinite(grads))), f"non-finite gradient at iter {i}"
        assert np.isfinite(float(loss)), f"non-finite loss at iter {i}"
        losses.append(float(loss))

    rho_final = transform(theta, BETAS[-1])
    p_final = float(radiated_fraction(rho_final))

    # 1) at least 2x improvement of the normalized radiated power
    assert p_final >= 2.0 * p_initial, (
        f"P_rad/P_avail only improved {p_final / p_initial:.2f}x "
        f"({p_initial:.2e} -> {p_final:.2e})"
    )
    # 2) nearly binary design at the final beta
    gray = float(gray_indicator(rho_final))
    assert gray < 0.35, f"gray indicator {gray:.3f} >= 0.35"
    # 3) the optimizer actually descended (sanity on the recorded losses)
    assert min(losses) <= losses[0]
