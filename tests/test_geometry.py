"""Axis-confusion guards: non-square grids, dx != dy, multi-source, validation."""

import subprocess
import sys
import textwrap

import jax.numpy as jnp
import pytest

from gradenna import CPMLSpec, Grid2D, gaussian_derivative, simulate_tm


def test_mirror_symmetry_on_anisotropic_grid():
    """A centered source on a 91x61 grid with dx != dy must produce fields
    mirror-symmetric about both axes — broadcasting a coefficient table
    along the wrong axis would break this."""
    grid = Grid2D(nx=91, ny=61, dx=1e-3, dy=2e-3)
    c = (45, 30)
    t = (jnp.arange(300) + 0.5) * grid.dt
    current = gaussian_derivative(t, t0=120 * grid.dt, tau=20 * grid.dt)
    probes = [
        (45 + 20, 30), (45 - 20, 30),  # +/- x pair
        (45, 30 + 12), (45, 30 - 12),  # +/- y pair
    ]
    res = simulate_tm(
        grid,
        source_ij=c,
        source_current=current,
        probe_ij=probes,
        cpml=CPMLSpec(thickness=10),
    )
    px, mx, py, my = (res.probe_ez[:, k] for k in range(4))
    scale = float(jnp.abs(res.probe_ez).max())
    assert float(jnp.abs(px - mx).max()) <= 1e-10 * scale
    assert float(jnp.abs(py - my).max()) <= 1e-10 * scale


def test_two_sources_superpose():
    """Two simultaneous sources must equal the sum of the individual runs
    (the scheme is linear)."""
    grid = Grid2D(nx=80, ny=80, dx=2e-3, dy=2e-3)
    t = (jnp.arange(250) + 0.5) * grid.dt
    cur = gaussian_derivative(t, t0=120 * grid.dt, tau=20 * grid.dt)
    probe = ((40, 60),)
    kw = dict(probe_ij=probe, cpml=CPMLSpec(thickness=10))

    both = simulate_tm(
        grid,
        source_ij=[(30, 40), (50, 40)],
        source_current=jnp.stack([cur, 0.5 * cur], axis=1),
        **kw,
    )
    a = simulate_tm(grid, source_ij=(30, 40), source_current=cur, **kw)
    b = simulate_tm(grid, source_ij=(50, 40), source_current=0.5 * cur, **kw)

    diff = jnp.abs(both.probe_ez - (a.probe_ez + b.probe_ez)).max()
    assert float(diff) <= 1e-12 * float(jnp.abs(both.probe_ez).max())


def test_input_validation():
    grid = Grid2D(nx=40, ny=40, dx=1e-3, dy=1e-3)
    t = jnp.zeros(10)
    with pytest.raises(ValueError, match="too small"):
        simulate_tm(grid, source_ij=(20, 20), source_current=t, cpml=CPMLSpec(thickness=19))
    with pytest.raises(ValueError, match="source_ij"):
        simulate_tm(grid, source_ij=(0, 20), source_current=t)
    with pytest.raises(ValueError, match="columns"):
        simulate_tm(grid, source_ij=[(20, 20), (21, 21)], source_current=t)
    with pytest.raises(ValueError, match="at least 3x3"):
        Grid2D(nx=2, ny=40, dx=1e-3, dy=1e-3)


def test_float32_mode_smoke():
    """Without x64, the solver must stay in float32 and remain finite.
    Runs in a subprocess because the x64 flag is process-global."""
    code = textwrap.dedent(
        """
        import jax.numpy as jnp
        from gradenna import CPMLSpec, Grid2D, gaussian_derivative, simulate_tm

        grid = Grid2D(nx=60, ny=60, dx=2e-3, dy=2e-3)
        t = (jnp.arange(200) + 0.5) * grid.dt
        cur = gaussian_derivative(t, t0=120 * grid.dt, tau=20 * grid.dt)
        res = simulate_tm(grid, source_ij=(30, 30), source_current=cur,
                          probe_ij=((30, 45),), cpml=CPMLSpec(thickness=8))
        assert res.probe_ez.dtype == jnp.float32, res.probe_ez.dtype
        assert bool(jnp.all(jnp.isfinite(res.probe_ez)))
        assert float(jnp.abs(res.probe_ez).max()) > 0.0
        """
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env={"JAX_ENABLE_X64": "0", "JAX_PLATFORMS": "cpu", "PATH": ""},
    )
