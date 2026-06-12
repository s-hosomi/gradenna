"""Acceptance tests for the design-region-limited 3D running-DFT monitor.

The region-limited DFT accumulates each field component only on a static
slab (or not at all), cutting the simulation's DFT carry and the
frequency-domain adjoint's backward residuals. These tests check:

(a) numerical identity of the slab spectra against the full-grid DFT
    (``assert_array_equal`` -- the limited DFT must be bit-identical on the
    cells it covers);
(b) gradient parity of ``freq_adjoint_gradient_3d(objective_kind=...)``
    against ``objective_kind=None`` for the port and field objectives
    (cos >= 1 - 1e-12, relative directional error <= 1e-10);
(c) the same for the NTFF box objective;
(e) the memory property: the region-limited forward DFT carry is far smaller
    than the full-grid carry (~ design-region sized).

Run with::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/test_dft_region3d.py -q
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradenna import CPMLSpec, Grid3D  # noqa: E402
from gradenna.dft_region import (  # noqa: E402
    design_region_to_slabs,
    full_field_shapes,
    ntff_box_regions,
    port_regions,
    scatter_full,
)
from gradenna.fdtd3d import DFTMonitor, simulate_3d  # noqa: E402
from gradenna.freq_adjoint import freq_adjoint_gradient_3d  # noqa: E402
from gradenna.ntff import directivity_3d, ntff_3d  # noqa: E402
from gradenna.sparams import s11_power_wave  # noqa: E402

# Small grid: the gradient tests need steady state, so keep the grid tiny and
# the run length moderate (these are functional, not performance, tests).
GRID = Grid3D(nx=24, ny=24, nz=24, dx=3e-3, dy=3e-3, dz=3e-3)
NPML = 6
F0 = 4e9
N_STEPS = 4000
RAMP = 300

DR = (slice(9, 13), slice(9, 13), slice(9, 13))
DR_SHAPE = (4, 4, 4)
CENTER = (12, 12, 12)


def _cw(grid, n_steps=N_STEPS, ramp=RAMP, f0=F0):
    n = jnp.arange(n_steps)
    env = jnp.clip(n / float(ramp), 0.0, 1.0)
    cur = env * jnp.sin(2.0 * jnp.pi * f0 * (n + 0.5) * grid.dt)
    return cur, env


def _compare(g_or, g_fa, seed=0):
    g_or = np.asarray(g_or).ravel()
    g_fa = np.asarray(g_fa).ravel()
    assert np.all(np.isfinite(g_fa))
    cos = float(np.dot(g_or, g_fa) / (np.linalg.norm(g_or) * np.linalg.norm(g_fa)))
    rng = np.random.default_rng(seed)
    v = rng.normal(size=g_or.shape)
    v /= np.linalg.norm(v)
    d_or = float(np.dot(g_or, v))
    d_fa = float(np.dot(g_fa, v))
    rel = abs(d_fa - d_or) / max(abs(d_or), 1e-30)
    return cos, rel


# ---------------------------------------------------------------------------
# (a) Region-limited DFT == full-grid DFT on the covered cells (bit-identical)
# ---------------------------------------------------------------------------


def test_limited_dft_matches_full_grid():
    cur, _ = _cw(GRID)
    common = dict(
        eps_r=1.0,
        source_ijk=(10, 12, 12),
        source_current=cur,
        dft_freqs=(F0,),
        dft_dtype=jnp.complex128,
        cpml=CPMLSpec(thickness=NPML),
    )
    full = simulate_3d(GRID, **common)
    assert isinstance(full.dft, DFTMonitor)

    full_shapes = full_field_shapes(GRID.nx, GRID.ny, GRID.nz)
    # Cover all three E components (design slab) plus port/NTFF style H slabs.
    regions = ntff_box_regions(GRID, NPML + 1, full_shapes)
    lim = simulate_3d(GRID, dft_regions=regions, **common)

    full_comps = (full.dft.ex, full.dft.ey, full.dft.ez,
                  full.dft.hx, full.dft.hy, full.dft.hz)
    lim_comps = (lim.dft.ex, lim.dft.ey, lim.dft.ez,
                 lim.dft.hx, lim.dft.hy, lim.dft.hz)
    for slab, full_arr, lim_arr in zip(regions, full_comps, lim_comps):
        assert slab is not None and lim_arr is not None
        sub = np.asarray(full_arr)[(slice(None),) + slab.slices()]
        np.testing.assert_array_equal(np.asarray(lim_arr), sub)

    # scatter_full reproduces the full-grid array exactly on the slab cells.
    scattered = scatter_full(lim.dft, full_shapes)
    for slab, full_arr, sc_arr in zip(
        regions, full_comps, (scattered.ex, scattered.ey, scattered.ez,
                              scattered.hx, scattered.hy, scattered.hz)
    ):
        sl = (slice(None),) + slab.slices()
        np.testing.assert_array_equal(
            np.asarray(sc_arr)[sl], np.asarray(full_arr)[sl]
        )


def test_none_components_are_not_accumulated():
    """A DFTRegions with only E3 slabs leaves Hx/Hy/Hz as None (off the carry)."""
    cur, _ = _cw(GRID)
    full_shapes = full_field_shapes(GRID.nx, GRID.ny, GRID.nz)
    regions = design_region_to_slabs(DR, full_shapes)
    assert regions.hx is None and regions.hy is None and regions.hz is None
    res = simulate_3d(
        GRID,
        source_ijk=(10, 12, 12),
        source_current=cur,
        dft_freqs=(F0,),
        dft_dtype=jnp.complex128,
        dft_regions=regions,
        cpml=CPMLSpec(thickness=NPML),
    )
    assert res.dft.ex is not None and res.dft.hx is None


# ---------------------------------------------------------------------------
# (b) Port and field objective gradient parity vs the full-grid path
# ---------------------------------------------------------------------------


def test_port_objective_gradient_parity():
    cur, env = _cw(GRID)
    rs = 50.0

    def obj(ph):
        s = s11_power_wave(ph.port_v[0, 0], ph.port_i[0, 0], rs)
        return s.real**2 + s.imag**2

    sig0 = 0.05 * jnp.ones(DR_SHAPE)
    common = dict(
        design_region=DR,
        dft_freqs=(F0,),
        port_ijk=CENTER,
        port_voltage=cur,
        port_resistance=rs,
        env=env,
        cpml=CPMLSpec(thickness=NPML),
    )
    g_full = freq_adjoint_gradient_3d(GRID, obj, design_sigma=sig0, **common)["sigma"]
    g_lim = freq_adjoint_gradient_3d(
        GRID, obj, design_sigma=sig0, objective_kind="port", **common
    )["sigma"]
    cos, rel = _compare(g_full, g_lim)
    assert cos >= 1 - 1e-12, f"port cosine {cos:.15f}"
    assert rel <= 1e-10, f"port rel. error {rel:.2e}"


def test_field_objective_gradient_parity():
    cur, env = _cw(GRID)
    mon = (16, 12, 12)
    monitor_region = (slice(16, 17), slice(12, 13), slice(12, 13))

    def obj(ph):
        e = ph.dft_ez[0, mon[0], mon[1], mon[2]]
        return e.real**2 + e.imag**2

    sig0 = 0.02 * jnp.ones(DR_SHAPE)
    common = dict(
        design_region=DR,
        dft_freqs=(F0,),
        source_ijk=(10, 12, 12),
        source_current=cur,
        env=env,
        cpml=CPMLSpec(thickness=NPML),
    )
    g_full = freq_adjoint_gradient_3d(GRID, obj, design_sigma=sig0, **common)["sigma"]
    g_lim = freq_adjoint_gradient_3d(
        GRID, obj, design_sigma=sig0,
        objective_kind="field", monitor_region=monitor_region, **common
    )["sigma"]
    cos, rel = _compare(g_full, g_lim)
    assert cos >= 1 - 1e-12, f"field cosine {cos:.15f}"
    assert rel <= 1e-10, f"field rel. error {rel:.2e}"


def test_field_objective_eps_r_parity():
    cur, env = _cw(GRID)
    mon = (16, 12, 12)
    monitor_region = (slice(16, 17), slice(12, 13), slice(12, 13))

    def obj(ph):
        e = ph.dft_ez[0, mon[0], mon[1], mon[2]]
        return e.real**2 + e.imag**2

    eps0 = 2.0 * jnp.ones(DR_SHAPE)
    common = dict(
        design_region=DR,
        dft_freqs=(F0,),
        source_ijk=(10, 12, 12),
        source_current=cur,
        env=env,
        cpml=CPMLSpec(thickness=NPML),
    )
    g_full = freq_adjoint_gradient_3d(GRID, obj, design_eps_r=eps0, **common)["eps_r"]
    g_lim = freq_adjoint_gradient_3d(
        GRID, obj, design_eps_r=eps0,
        objective_kind="field", monitor_region=monitor_region, **common
    )["eps_r"]
    cos, rel = _compare(g_full, g_lim)
    assert cos >= 1 - 1e-12, f"field/eps_r cosine {cos:.15f}"
    assert rel <= 1e-10, f"field/eps_r rel. error {rel:.2e}"


# ---------------------------------------------------------------------------
# (c) NTFF box objective gradient parity
# ---------------------------------------------------------------------------


def test_ntff_objective_gradient_parity():
    cur, env = _cw(GRID)
    freqs = jnp.array([F0])
    thetas = jnp.linspace(0.05, np.pi - 0.05, 7)
    phis = jnp.linspace(0.0, 2 * np.pi, 8, endpoint=False)
    th0, ph0 = 3, 0

    def obj(ph):
        mon = DFTMonitor(
            freqs, ph.dft_ex, ph.dft_ey, ph.dft_ez, ph.dft_hx, ph.dft_hy, ph.dft_hz
        )
        ef = ntff_3d(mon, GRID, NPML + 1, freqs, thetas, phis)
        d = directivity_3d(ef[..., 0], ef[..., 1], thetas, phis)
        return -d[0, th0, ph0]

    sig0 = 0.02 * jnp.ones(DR_SHAPE)
    common = dict(
        design_region=DR,
        dft_freqs=(F0,),
        source_ijk=CENTER,
        source_current=cur,
        env=env,
        cpml=CPMLSpec(thickness=NPML),
    )
    g_full = freq_adjoint_gradient_3d(GRID, obj, design_sigma=sig0, **common)["sigma"]
    g_lim = freq_adjoint_gradient_3d(
        GRID, obj, design_sigma=sig0,
        objective_kind="ntff_box", box_margin=NPML + 1, **common
    )["sigma"]
    cos, rel = _compare(g_full, g_lim)
    assert cos >= 1 - 1e-12, f"NTFF cosine {cos:.15f}"
    assert rel <= 1e-10, f"NTFF rel. error {rel:.2e}"


# ---------------------------------------------------------------------------
# (e) Memory: the region-limited forward DFT carry is far smaller than full
# ---------------------------------------------------------------------------


def _dft_carry_bytes(dft_tuple):
    total = 0
    for a in dft_tuple:
        if a is None:
            continue
        total += int(np.prod(a.shape)) * np.dtype(a.dtype).itemsize
    return total


def test_region_dft_carry_is_small():
    cur, _ = _cw(GRID)
    full_shapes = full_field_shapes(GRID.nx, GRID.ny, GRID.nz)
    n_freq = 1
    common = dict(
        source_ijk=(10, 12, 12),
        source_current=cur,
        dft_freqs=(F0,),
        dft_dtype=jnp.complex128,
        cpml=CPMLSpec(thickness=NPML),
    )

    full = simulate_3d(GRID, **common)
    full_bytes = _dft_carry_bytes(
        (full.dft.ex, full.dft.ey, full.dft.ez,
         full.dft.hx, full.dft.hy, full.dft.hz)
    )

    regions = design_region_to_slabs(DR, full_shapes)
    lim = simulate_3d(GRID, dft_regions=regions, **common)
    lim_bytes = _dft_carry_bytes(
        (lim.dft.ex, lim.dft.ey, lim.dft.ez,
         lim.dft.hx, lim.dft.hy, lim.dft.hz)
    )

    # The design-region E3 carry: 3 components x n_design x n_freq x 16 B,
    # give or take the per-component shape clamping (all >= the box volume).
    n_design = DR_SHAPE[0] * DR_SHAPE[1] * DR_SHAPE[2]
    expected = 3 * n_design * n_freq * 16
    assert lim_bytes <= 2 * expected, (
        f"region DFT carry {lim_bytes} B should be O(3 x design x freq) ~{expected} B"
    )
    assert lim_bytes < full_bytes / 50, (
        f"region DFT carry {lim_bytes} B should be << full-grid {full_bytes} B"
    )
