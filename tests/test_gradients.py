"""B5: reverse-mode AD gradients vs central finite differences.

Random-direction directional derivatives (2 simulations per direction)
instead of a full O(N) finite-difference sweep; float64 with the step
swept around h* ~ eps_machine^(1/3) and the best (V-shape bottom) taken.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from gradenna import CPMLSpec, Grid2D, alpha_max_for_fmin, gaussian_derivative, simulate_tm

GRID = Grid2D(nx=40, ny=40, dx=2e-3, dy=2e-3)
NPML = 8
N_STEPS = 250
REGION = (slice(16, 24), slice(16, 24))  # 8x8 design patch between source and probe


def _loss(design: dict) -> jnp.ndarray:
    """Transmission-like loss: integrated probe energy behind the design patch."""
    eps_r = jnp.ones(GRID.shape).at[REGION].set(design["eps_r"])
    sigma = jnp.zeros(GRID.shape).at[REGION].set(design["sigma"])
    t = (jnp.arange(N_STEPS) + 0.5) * GRID.dt
    tau = 15 * GRID.dt
    current = gaussian_derivative(t, t0=6 * tau, tau=tau)
    res = simulate_tm(
        GRID,
        source_ij=(12, 20),
        source_current=current,
        probe_ij=((28, 20),),
        eps_r=eps_r,
        sigma=sigma,
        cpml=CPMLSpec(thickness=NPML, alpha_max=alpha_max_for_fmin(1e9)),
    )
    return jnp.sum(res.probe_ez**2) * GRID.dt


def _design0():
    return {
        "eps_r": 2.0 * jnp.ones((8, 8)),
        "sigma": 0.05 * jnp.ones((8, 8)),
    }


def _directional_fd(f, x, v, h):
    flat, unravel = ravel_pytree(x)
    fp = f(unravel(flat + h * v))
    fm = f(unravel(flat - h * v))
    return (fp - fm) / (2.0 * h)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_grad_matches_directional_finite_difference(seed):
    design = _design0()
    grad = jax.grad(_loss)(design)
    g_flat, _ = ravel_pytree(grad)
    flat, _ = ravel_pytree(design)
    assert bool(jnp.all(jnp.isfinite(g_flat)))

    rng = np.random.default_rng(seed)
    v = rng.normal(size=flat.shape)
    v = jnp.asarray(v / np.linalg.norm(v))

    d_ad = float(jnp.vdot(g_flat, v))
    scale = float(jnp.linalg.norm(flat))
    errs = []
    for h in (3e-7, 1e-6, 3e-6):
        d_fd = float(_directional_fd(_loss, design, v, h * scale))
        errs.append(abs(d_ad - d_fd) / max(abs(d_ad), abs(d_fd)))
    assert min(errs) <= 1e-4, f"AD vs FD relative error {min(errs):.2e}"


def test_grad_has_no_nan_with_and_without_cfs():
    """CPML coefficient guards must keep gradients NaN-free (alpha=0 corner case)."""
    for alpha in (0.0, alpha_max_for_fmin(1e9)):
        def loss(eps_patch, alpha=alpha):
            eps_r = jnp.ones(GRID.shape).at[REGION].set(eps_patch)
            t = (jnp.arange(120) + 0.5) * GRID.dt
            current = gaussian_derivative(t, t0=90 * GRID.dt, tau=15 * GRID.dt)
            res = simulate_tm(
                GRID,
                source_ij=(12, 20),
                source_current=current,
                probe_ij=((28, 20),),
                eps_r=eps_r,
                cpml=CPMLSpec(thickness=NPML, alpha_max=alpha),
            )
            return jnp.sum(res.probe_ez**2)

        g = jax.grad(loss)(2.0 * jnp.ones((8, 8)))
        assert bool(jnp.all(jnp.isfinite(g)))
        assert float(jnp.abs(g).max()) > 0.0


def test_grad_wrt_source_current():
    """The loss must also be differentiable through the source injection."""
    t = (jnp.arange(150) + 0.5) * GRID.dt
    current0 = gaussian_derivative(t, t0=90 * GRID.dt, tau=15 * GRID.dt)

    def loss(current):
        res = simulate_tm(
            GRID,
            source_ij=(12, 20),
            source_current=current,
            probe_ij=((28, 20),),
            cpml=CPMLSpec(thickness=NPML),
        )
        return jnp.sum(res.probe_ez**2)

    g = jax.grad(loss)(current0)
    assert g.shape == current0.shape
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.abs(g).max()) > 0.0
