"""B3: Courant stability — both sides of the limit — and bounded energy."""

import jax.numpy as jnp

from gradenna import CPMLSpec, Grid2D, gaussian_derivative, simulate_tm

PEC_BOX = CPMLSpec(thickness=0)


def _run(courant: float, n_steps: int):
    grid = Grid2D(nx=60, ny=60, dx=2e-3, dy=2e-3, courant=courant)
    tau = 20 * grid.dt
    t = (jnp.arange(n_steps) + 0.5) * grid.dt
    current = gaussian_derivative(t, t0=6 * tau, tau=tau)
    return simulate_tm(
        grid, source_ij=(30, 30), source_current=current, cpml=PEC_BOX, record_energy=True
    )


def test_stable_below_courant_limit():
    """At S = 0.99 the energy in a lossless PEC box must not grow."""
    res = _run(courant=0.99, n_steps=2000)
    # Source is off well before step 400; afterwards energy may oscillate
    # within the staggered-time envelope but must not grow secularly.
    after = res.energy[400:]
    assert float(after.max()) <= 1.02 * float(after[:100].max())
    assert jnp.all(jnp.isfinite(res.energy))


def test_unstable_above_courant_limit():
    """At S = 1.01 the scheme must blow up within a few hundred steps."""
    res = _run(courant=1.01, n_steps=600)
    seed = float(res.energy[100])
    end = res.energy[-1]
    assert (not bool(jnp.isfinite(end))) or float(end) > 1e6 * seed
