"""3D FDTD core acceptance tests.

B6 (infinitesimal dipole radiation resistance), CPML energy decay, mirror
symmetry, AD-vs-finite-difference gradients and sqrt-N checkpointing
equivalence.
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from gradenna import CPMLSpec, alpha_max_for_fmin, gaussian_derivative
from gradenna.constants import C0
from gradenna.fdtd3d import Grid3D, port_impedance, simulate_3d, time_series_dft

DX = 4e-3  # cubic cells; lambda/dx >= 25 at 3 GHz


def _openems_gauss(t, f0: float, fc: float):
    """openEMS SetGaussExcite-compatible modulated Gaussian (note 12, sec. 4.1).

    -20 dB cutoffs at f0 +- fc; turn-on step ~ -78 dB; no DC for f0 >= fc.
    """
    sigma = 3.0 / (2.0 * math.sqrt(2.0) * math.pi * fc)
    t0 = 9.0 / (2.0 * math.pi * fc)
    return jnp.cos(2.0 * math.pi * f0 * (t - t0)) * jnp.exp(-0.5 * ((t - t0) / sigma) ** 2)


@functools.lru_cache(maxsize=2)  # shared by the two port tests
def _dipole_run(n: int, n_steps: int):
    """Free space + CPML, z-directed 1-cell RVS port at the grid center."""
    grid = Grid3D(nx=n, ny=n, nz=n, dx=DX, dy=DX, dz=DX)
    t_half = (jnp.arange(n_steps) + 0.5) * grid.dt
    vs = _openems_gauss(t_half, f0=2.5e9, fc=1.0e9)
    c = n // 2
    res = simulate_3d(
        grid,
        port_ijk=(c, c, c),
        port_voltage=vs,
        port_resistance=50.0,
        cpml=CPMLSpec(thickness=10, alpha_max=alpha_max_for_fmin(1.5e9)),
    )
    return grid, res


# ---------------------------------------------------------------------------
# B6: infinitesimal dipole radiation resistance
# ---------------------------------------------------------------------------


def test_small_dipole_radiation_resistance():
    """Re{Zin} of a 1-cell RVS-fed dipole must match R_r = 80 pi^2 (dz/lambda)^2.

    The raw Ampere-loop V/I includes the gap displacement current, so the
    exactly-known discrete gap susceptance is de-embedded (note 12 sec. 6.1);
    the 15% budget absorbs the remaining feed-gap/dispersion systematics.
    """
    grid, res = _dipole_run(n=48, n_steps=800)
    freqs = jnp.linspace(1.8e9, 3.2e9, 8)
    zin = port_impedance(res, grid, freqs)

    r_theory = 80.0 * math.pi**2 * (grid.dz * freqs / C0) ** 2
    rel_err = jnp.abs(jnp.real(zin) - r_theory) / r_theory
    assert float(rel_err.max()) <= 0.15, (
        f"R_r relative error {np.asarray(rel_err)} exceeds 15%"
    )
    # The bare 1-cell dipole must look strongly capacitive across the band.
    assert bool(jnp.all(jnp.imag(zin) < 0.0))


def test_raw_port_impedance_shows_gap_shunt():
    """Without de-embedding, Re{V/I} is biased upward by the gap shunt
    (1 - w C_gap X)^-2 with X < 0 — the known systematic of note 12 sec. 6.1."""
    grid, res = _dipole_run(n=48, n_steps=800)
    freqs = jnp.linspace(2.0e9, 3.0e9, 3)
    z_raw = port_impedance(res, grid, freqs, deembed_gap=False)
    z_cor = port_impedance(res, grid, freqs, deembed_gap=True)
    assert bool(jnp.all(jnp.real(z_raw) > jnp.real(z_cor)))


# ---------------------------------------------------------------------------
# CPML energy decay
# ---------------------------------------------------------------------------


def test_cpml_energy_decay():
    """After the pulse leaves, total energy must decay to a deep floor.

    The floor (~2e-16 of the peak, i.e. field norm ~1.5e-8) is the static
    remnant of the discretely sampled derivative-Gaussian current, whose
    time integral is not exactly zero; the resulting static dipole field
    cannot propagate into the CPML. 1e-15 is seven orders of magnitude
    below the acceptance requirement of 1e-8.
    """
    n = 36
    grid = Grid3D(nx=n, ny=n, nz=n, dx=DX, dy=DX, dz=DX)
    tau = 1.0 / (2.0 * math.pi * 2.5e9)
    t = (jnp.arange(1400) + 0.5) * grid.dt
    current = gaussian_derivative(t, t0=6 * tau, tau=tau)
    c = n // 2
    res = simulate_3d(
        grid,
        source_ijk=(c, c, c),
        source_current=current,
        cpml=CPMLSpec(thickness=10, alpha_max=alpha_max_for_fmin(0.5e9)),
        record_energy=True,
    )
    assert bool(jnp.all(jnp.isfinite(res.energy)))
    ratio = float(res.energy[-1] / res.energy.max())
    assert ratio <= 1e-15, f"residual energy ratio {ratio:.2e} > 1e-15"


# ---------------------------------------------------------------------------
# Mirror symmetry (axis mix-up detector)
# ---------------------------------------------------------------------------


def test_mirror_symmetry_x_and_y():
    """A center z-current on an odd cubic grid: Ez at +-x and +-y offsets
    must agree at every step (catches transposed axes / wrong curl signs)."""
    n = 31  # odd: source sits exactly on the symmetry planes
    grid = Grid3D(nx=n, ny=n, nz=n, dx=DX, dy=DX, dz=DX)
    c = n // 2
    t = (jnp.arange(300) + 0.5) * grid.dt
    current = gaussian_derivative(t, t0=60 * grid.dt, tau=12 * grid.dt)
    probes = ((c + 4, c, c), (c - 4, c, c), (c, c + 4, c), (c, c - 4, c))
    res = simulate_3d(
        grid,
        source_ijk=(c, c, c),
        source_current=current,
        probe_ijk=probes,
        cpml=CPMLSpec(thickness=8),
    )
    p = np.asarray(res.probe_ez)
    spread = np.abs(p - p[:, :1]).max() / np.abs(p).max()
    assert spread <= 1e-12, f"symmetric probes differ by {spread:.2e}"


# ---------------------------------------------------------------------------
# B5 (3D): gradients vs directional finite differences
# ---------------------------------------------------------------------------

GRAD_GRID = Grid3D(nx=20, ny=20, nz=20, dx=DX, dy=DX, dz=DX)
GRAD_STEPS = 120
GRAD_REGION = (slice(8, 12), slice(8, 12), slice(8, 12))  # 4^3 patch


def _grad_loss(design: dict) -> jnp.ndarray:
    """Transmission-like loss: probe energy behind an eps/sigma patch."""
    eps_r = jnp.ones(GRAD_GRID.shape).at[GRAD_REGION].set(design["eps_r"])
    sigma = jnp.zeros(GRAD_GRID.shape).at[GRAD_REGION].set(design["sigma"])
    t = (jnp.arange(GRAD_STEPS) + 0.5) * GRAD_GRID.dt
    current = gaussian_derivative(t, t0=30 * GRAD_GRID.dt, tau=8 * GRAD_GRID.dt)
    res = simulate_3d(
        GRAD_GRID,
        source_ijk=(6, 10, 9),
        source_current=current,
        probe_ijk=((14, 10, 9),),
        eps_r=eps_r,
        sigma=sigma,
        cpml=CPMLSpec(thickness=4, alpha_max=alpha_max_for_fmin(2e9)),
    )
    return jnp.sum(res.probe_ez**2) * GRAD_GRID.dt


@pytest.mark.parametrize("seed", [0, 1])
def test_grad_matches_directional_finite_difference(seed):
    design = {"eps_r": 2.0 * jnp.ones((4, 4, 4)), "sigma": 0.05 * jnp.ones((4, 4, 4))}
    grad = jax.grad(_grad_loss)(design)
    g_flat, _ = ravel_pytree(grad)
    assert bool(jnp.all(jnp.isfinite(g_flat)))
    assert float(jnp.abs(g_flat).max()) > 0.0

    flat, unravel = ravel_pytree(design)
    rng = np.random.default_rng(seed)
    v = rng.normal(size=flat.shape)
    v = jnp.asarray(v / np.linalg.norm(v))

    d_ad = float(jnp.vdot(g_flat, v))
    scale = float(jnp.linalg.norm(flat))
    errs = []
    for h in (3e-7, 1e-6, 3e-6):
        hh = h * scale
        d_fd = float((_grad_loss(unravel(flat + hh * v)) - _grad_loss(unravel(flat - hh * v))) / (2 * hh))
        errs.append(abs(d_ad - d_fd) / max(abs(d_ad), abs(d_fd)))
    assert min(errs) <= 1e-4, f"AD vs FD relative error {min(errs):.2e}"


# ---------------------------------------------------------------------------
# sqrt-N checkpointing equivalence
# ---------------------------------------------------------------------------

CKPT_GRID = Grid3D(nx=16, ny=16, nz=16, dx=DX, dy=DX, dz=DX)
CKPT_STEPS = 80
CKPT_FREQS = jnp.array([2.0e9, 3.0e9])


def _ckpt_loss(eps_patch, segments):
    """Loss touching every output kind: probes, port V/I and the DFT monitor."""
    eps_r = jnp.ones(CKPT_GRID.shape).at[6:10, 6:10, 6:10].set(eps_patch)
    t = (jnp.arange(CKPT_STEPS) + 0.5) * CKPT_GRID.dt
    vs = gaussian_derivative(t, t0=25 * CKPT_GRID.dt, tau=7 * CKPT_GRID.dt)
    res = simulate_3d(
        CKPT_GRID,
        port_ijk=(8, 8, 7),
        port_voltage=vs,
        eps_r=eps_r,
        probe_ijk=((11, 8, 7),),
        dft_freqs=CKPT_FREQS,
        cpml=CPMLSpec(thickness=4),
        checkpoint_segments=segments,
    )
    return (
        jnp.sum(res.probe_ez**2)
        + 1e-3 * jnp.sum(jnp.abs(res.dft.ez) ** 2)
        + 1e-2 * jnp.sum(res.port_v**2)
        + 1e-2 * jnp.sum(res.port_i**2)
    )


def test_checkpoint_segments_match_flat_scan():
    eps_patch = 2.5 * jnp.ones((4, 4, 4))
    loss_flat = _ckpt_loss(eps_patch, None)
    loss_ckpt = _ckpt_loss(eps_patch, 4)
    assert float(jnp.abs(loss_flat - loss_ckpt) / jnp.abs(loss_flat)) <= 1e-9

    g_flat = jax.grad(lambda e: _ckpt_loss(e, None))(eps_patch)
    g_ckpt = jax.grad(lambda e: _ckpt_loss(e, 4))(eps_patch)
    rel = float(jnp.abs(g_flat - g_ckpt).max() / jnp.abs(g_flat).max())
    assert rel <= 1e-9, f"checkpointed gradient deviates by {rel:.2e}"


def test_checkpoint_segments_must_divide_n_steps():
    with pytest.raises(ValueError, match="checkpoint_segments"):
        _ckpt_loss(2.5 * jnp.ones((4, 4, 4)), 7)  # 80 % 7 != 0


# ---------------------------------------------------------------------------
# DFT monitor consistency and basic API validation
# ---------------------------------------------------------------------------


def test_dft_monitor_matches_post_hoc_dft():
    """The running DFT must equal the exact-phase DFT of the recorded series."""
    n = 16
    grid = Grid3D(nx=n, ny=n, nz=n, dx=DX, dy=DX, dz=DX)
    n_steps = 100
    t = (jnp.arange(n_steps) + 0.5) * grid.dt
    current = gaussian_derivative(t, t0=30 * grid.dt, tau=8 * grid.dt)
    freqs = jnp.array([1.5e9, 2.5e9, 3.5e9])
    probe = (10, 8, 7)
    res = simulate_3d(
        grid,
        source_ijk=(8, 8, 7),
        source_current=current,
        probe_ijk=(probe,),
        dft_freqs=freqs,
        cpml=CPMLSpec(thickness=4),
    )
    post_hoc = time_series_dft(res.probe_ez[:, 0], grid.dt, freqs, t0=grid.dt)
    running = res.dft.ez[:, probe[0], probe[1], probe[2]]
    err = float(jnp.abs(running - post_hoc).max() / jnp.abs(post_hoc).max())
    assert err <= 1e-12
    # H accumulators must be populated too (exercised at (n+1/2) dt phases).
    assert float(jnp.abs(res.dft.hx).max()) > 0.0
    assert float(jnp.abs(res.dft.hy).max()) > 0.0


def test_grid3d_validation_and_dt():
    grid = Grid3D(nx=10, ny=12, nz=14, dx=1e-3, dy=2e-3, dz=3e-3, courant=0.5)
    expected = 0.5 / (C0 * math.sqrt(1 / 1e-3**2 + 1 / 2e-3**2 + 1 / 3e-3**2))
    assert math.isclose(grid.dt, expected, rel_tol=1e-12)
    assert grid.shape == (10, 12, 14)
    with pytest.raises(ValueError):
        Grid3D(nx=2, ny=10, nz=10, dx=1e-3, dy=1e-3, dz=1e-3)
    with pytest.raises(ValueError):
        Grid3D(nx=10, ny=10, nz=10, dx=-1e-3, dy=1e-3, dz=1e-3)


def test_argument_validation():
    grid = Grid3D(nx=16, ny=16, nz=16, dx=DX, dy=DX, dz=DX)
    cur = jnp.zeros(10)
    with pytest.raises(ValueError, match="given together"):
        simulate_3d(grid, source_ijk=(8, 8, 8), cpml=CPMLSpec(thickness=4))
    with pytest.raises(ValueError, match="at least one"):
        simulate_3d(grid, cpml=CPMLSpec(thickness=4))
    with pytest.raises(ValueError, match="outside the valid Ez range"):
        simulate_3d(
            grid,
            source_ijk=(0, 8, 8),
            source_current=cur,
            cpml=CPMLSpec(thickness=4),
        )
    with pytest.raises(ValueError, match="too small for CPML"):
        simulate_3d(grid, source_ijk=(8, 8, 8), source_current=cur, cpml=CPMLSpec(thickness=10))


# ---------------------------------------------------------------------------
# Slow: finer-grid radiation resistance (tighter tolerance)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_small_dipole_radiation_resistance_strict():
    """Same benchmark, larger box and longer run: de-embedded Re{Zin} within 5%."""
    grid, res = _dipole_run(n=60, n_steps=1100)
    freqs = jnp.linspace(1.8e9, 3.2e9, 15)
    zin = port_impedance(res, grid, freqs)
    r_theory = 80.0 * math.pi**2 * (grid.dz * freqs / C0) ** 2
    rel_err = jnp.abs(jnp.real(zin) - r_theory) / r_theory
    assert float(rel_err.max()) <= 0.05, f"strict R_r error {np.asarray(rel_err)}"
