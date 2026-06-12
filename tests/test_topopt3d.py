"""Acceptance tests for 3D topology optimization (Phase 4).

Three groups:

1. ``conic_filter`` in 3D: the nd kernel of ``gradenna.topopt`` must preserve
   a constant field (renormalized boundary handling), conserve interior mass,
   and stay finite/non-zero under ``jax.grad`` on a 3D density.
2. A reduced 3D topology-optimization regression: a thin conductive-sheet
   design layer above a ground plane, fed by a lumped RVS port, optimized to
   raise the 2.45 GHz radiated power. A few Adam steps must (a) keep the
   gradient finite and non-trivial and (b) increase the radiated power. Kept
   small enough to run in well under 90 s on CPU (``fast``); the slower
   binarization-trend check is marked ``slow``.
3. ``gradenna.estimate``: the memory model must reproduce the expected
   order of magnitude and order the cost layers sensibly.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradenna import (
    CPMLSpec,
    Grid3D,
    alpha_max_for_fmin,
    gaussian_pulse_for_band,
    half_step_dft,
    sheet_conductivity,
    sigma_from_density,
    simulate_3d,
)
from gradenna.constants import EPS0
from gradenna.estimate import fdtd3d_memory_estimate, fits_gpu, gpu_fit_report
from gradenna.ntff import ntff_3d, radiated_power_3d
from gradenna.topopt import DesignTransform, conic_filter, gray_indicator


# ---------------------------------------------------------------------------
# conic_filter: 3D coverage of the nd kernel
# ---------------------------------------------------------------------------


class TestConicFilter3D:
    def test_constant_field_preserved_including_boundary(self):
        rho = jnp.full((13, 11, 9), 0.42)
        out = conic_filter(rho, radius_cells=3.5)
        assert out.shape == rho.shape
        np.testing.assert_allclose(np.asarray(out), 0.42, rtol=0.0, atol=1e-12)

    def test_mass_conservation_interior(self):
        # Blob supported away from every boundary: the normalized kernel
        # conserves total mass there.
        rho = np.zeros((24, 24, 24))
        rho[8:16, 9:15, 10:14] = 1.0
        rho[10:13, 11:13, 11:13] = 0.5
        out = conic_filter(jnp.asarray(rho), radius_cells=3.0)
        np.testing.assert_allclose(float(jnp.sum(out)), rho.sum(), rtol=1e-10)

    def test_radius_one_is_identity(self):
        key = jax.random.PRNGKey(0)
        rho = jax.random.uniform(key, (7, 9, 5))
        np.testing.assert_array_equal(
            np.asarray(conic_filter(rho, 1.0)), np.asarray(rho)
        )

    def test_smoothing_reduces_total_variation(self):
        key = jax.random.PRNGKey(1)
        rho = (jax.random.uniform(key, (16, 16, 16)) > 0.5).astype(jnp.float64)
        out = conic_filter(rho, radius_cells=3.0)
        assert float(out.min()) >= -1e-12
        assert float(out.max()) <= 1.0 + 1e-12
        tv = lambda a: float(jnp.abs(jnp.diff(a, axis=0)).sum())
        assert tv(out) < tv(rho)

    def test_gradient_finite_and_nonzero(self):
        transform = DesignTransform(radius_cells=2.5)
        key = jax.random.PRNGKey(2)
        theta = jax.random.normal(key, (10, 10, 10))

        def objective(theta):
            rho = transform(theta, beta=8.0)
            return jnp.sum(rho**2)

        grad = jax.grad(objective)(theta)
        assert grad.shape == theta.shape
        assert bool(jnp.all(jnp.isfinite(grad)))
        assert float(jnp.linalg.norm(grad)) > 0.0


# ---------------------------------------------------------------------------
# Reduced 3D topology-optimization regression
# ---------------------------------------------------------------------------

F0 = 2.45e9
F_MIN, F_MAX = 1.5e9, 3.5e9
EPS_FR4 = 4.3
H_SUB = 1.6e-3
TAN_D = 0.02
T_COPPER = 35e-6
N_GAP = 3
SIG_FIXED = 1.0e7
RS = 50.0

# A deliberately small grid: 44 x 44 x 23 cells (design region 18 cells).
# Coarse, lossy and short, so the run is a regression of the *machinery*
# (finite gradient, radiated fraction responds to the design and improves),
# not a converged antenna. The objective is the radiated fraction
# P_rad(f0) / P_avail(f0) — the same O(1e-3) figure of merit as the example;
# optimizing the raw P_rad (~1e-24 with the dt-scaled DFT normalization)
# would put the gradient below Adam's epsilon floor and stall.
DXY = 2.6e-3
N_DES = 18
M_GND = 3
M_AIR = 4
N_PML = 6
N_AIR_ABOVE = 5
N_STEPS = 1600
CKPT = 40
THETAS = jnp.linspace(0.0, np.pi, 9)
PHIS = jnp.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)


class _Mini3D:
    """A tiny patch-style 3D problem reusing the example's physics."""

    def __init__(self):
        dz = H_SUB / N_GAP
        margin = M_GND + M_AIR + N_PML
        nx = ny = N_DES + 2 * margin
        kg = N_PML + 2
        k_patch = kg + N_GAP
        nz = k_patch + 1 + N_AIR_ABOVE + N_PML
        self.grid = Grid3D(nx=nx, ny=ny, nz=nz, dx=DXY, dy=DXY, dz=dz)
        self.kg, self.k_patch = kg, k_patch
        self.d0, self.d1 = margin, margin + N_DES
        g0, g1 = self.d0 - M_GND, self.d1 + M_GND

        eps_r_model = EPS_FR4 * (N_GAP - 1) / N_GAP
        sig_sub = 2.0 * math.pi * F0 * EPS0 * eps_r_model * TAN_D

        eps_r = np.ones(self.grid.shape)
        sigma = np.zeros(self.grid.shape)
        eps_r[g0:g1, g0:g1, kg + 1 : k_patch] = eps_r_model
        sigma[g0:g1, g0:g1, kg + 1 : k_patch] = sig_sub
        sigma[g0:g1, g0:g1, kg] = SIG_FIXED  # ground sheet
        pi = (self.d0 + self.d1) // 2
        pj = self.d0 + max(1, N_DES // 4)
        sigma[pi, pj, kg + 1 : k_patch - 1] = SIG_FIXED  # probe pin
        sigma[pi, pj, k_patch] = SIG_FIXED  # feed pad
        self.port_ijk = (pi, pj, k_patch - 1)
        feed_ij = (pi - self.d0, pj - self.d0)

        self.eps_r = jnp.asarray(eps_r)
        self.sigma_fixed = jnp.asarray(sigma)
        mask = np.ones((N_DES, N_DES), bool)
        mask[feed_ij] = False
        self.design_mask = jnp.asarray(mask)

        self.cpml = CPMLSpec(thickness=N_PML, alpha_max=alpha_max_for_fmin(F_MIN))
        self.pulse = gaussian_pulse_for_band(F_MIN, F_MAX)
        self.ntff_margin = N_PML + 1
        t = (jnp.arange(N_STEPS) + 0.5) * self.grid.dt
        self.p_avail_f0 = float(
            jnp.abs(half_step_dft(self.pulse(t), self.grid.dt, (F0,))[0]) ** 2
            / (8.0 * RS)
        )

    def sigma_total(self, rho):
        sheet = sheet_conductivity(1.0, T_COPPER, self.grid.dz) * sigma_from_density(
            rho, 1e-4, 1e5
        )
        sheet = jnp.where(self.design_mask, sheet, 0.0)
        design = jnp.zeros(self.grid.shape).at[
            self.d0 : self.d1, self.d0 : self.d1, self.k_patch
        ].set(sheet)
        return jnp.maximum(self.sigma_fixed, design)

    def radiated_fraction_f0(self, rho):
        """P_rad(f0) / P_avail(f0): the O(1e-3) radiated-fraction objective."""
        t = (jnp.arange(N_STEPS) + 0.5) * self.grid.dt
        res = simulate_3d(
            self.grid,
            eps_r=self.eps_r,
            sigma=self.sigma_total(rho),
            port_ijk=self.port_ijk,
            port_voltage=self.pulse(t),
            port_resistance=RS,
            cpml=self.cpml,
            dft_freqs=(F0,),
            checkpoint_segments=CKPT,
        )
        e_far = ntff_3d(res.dft, self.grid, self.ntff_margin, (F0,), THETAS, PHIS)
        p_rad = radiated_power_3d(e_far[..., 0], e_far[..., 1], THETAS, PHIS)[0]
        return p_rad / self.p_avail_f0


@pytest.fixture(scope="module")
def mini():
    return _Mini3D()


class TestReduced3DOptimization:
    def test_radiated_fraction_gradient_finite(self, mini):
        """grad of P_rad/P_avail(f0) w.r.t. the 2D design is finite & non-zero."""
        transform = DesignTransform(radius_cells=2.0)
        theta = jnp.zeros((N_DES, N_DES))  # gray start

        def obj(theta):
            rho = transform(theta, beta=8.0)
            return mini.radiated_fraction_f0(rho)

        val, grad = jax.value_and_grad(obj)(theta)
        assert np.isfinite(float(val)) and float(val) > 0.0
        assert grad.shape == (N_DES, N_DES)
        assert bool(jnp.all(jnp.isfinite(grad)))
        assert float(jnp.linalg.norm(grad)) > 0.0

    def test_few_adam_steps_increase_radiated_fraction(self, mini):
        """A handful of Adam steps must raise the 2.45 GHz radiated fraction."""
        import optax

        transform = DesignTransform(radius_cells=2.0)
        theta = jnp.zeros((N_DES, N_DES))
        opt = optax.adam(0.2)
        opt_state = opt.init(theta)

        def loss(theta, beta):
            return -mini.radiated_fraction_f0(transform(theta, beta))

        grad_fn = jax.jit(jax.value_and_grad(loss))
        fracs = []
        for _ in range(4):
            val, grad = grad_fn(theta, jnp.asarray(8.0))
            assert bool(jnp.all(jnp.isfinite(grad)))
            fracs.append(-float(val))
            updates, opt_state = opt.update(grad, opt_state, theta)
            theta = optax.apply_updates(theta, updates)
        final = -float(loss(theta, jnp.asarray(8.0)))
        assert final > 1.5 * fracs[0], (
            f"radiated fraction did not improve: {fracs[0]:.3e} -> {final:.3e}"
        )

    @pytest.mark.slow
    def test_beta_continuation_binarizes(self, mini):
        """Higher beta drives the projected density toward binary."""
        import optax

        transform = DesignTransform(radius_cells=2.0)
        theta = jnp.zeros((N_DES, N_DES))
        opt = optax.adam(0.2)
        opt_state = opt.init(theta)

        def loss(theta, beta):
            return -mini.radiated_fraction_f0(transform(theta, beta))

        grad_fn = jax.jit(jax.value_and_grad(loss))
        for beta in (jnp.asarray(8.0), jnp.asarray(8.0), jnp.asarray(32.0),
                     jnp.asarray(32.0)):
            _, grad = grad_fn(theta, beta)
            updates, opt_state = opt.update(grad, opt_state, theta)
            theta = optax.apply_updates(theta, updates)
        gray_lo = float(gray_indicator(transform(theta, 8.0)))
        gray_hi = float(gray_indicator(transform(theta, 64.0)))
        assert gray_hi < gray_lo  # sharper projection => more binary


# ---------------------------------------------------------------------------
# estimate: order-of-magnitude and layering checks
# ---------------------------------------------------------------------------


class TestMemoryEstimate:
    def test_doc_table_order_of_magnitude(self):
        """Memory-model order of magnitude: 82x82x51, Nt=2000, fp32 ~ tens
        of GB naive, ~1-2 GB checkpointed."""
        g = Grid3D(nx=82, ny=82, nz=51, dx=1.3e-3, dy=1.3e-3, dz=1.3e-3)
        est = fdtd3d_memory_estimate(
            g, 2000, n_dft_freqs=0, dtype="float32", cpml_thickness=10
        )
        # Forward state is a few tens of MB (doc: M_fwd ~ 18 MB for 6 fields;
        # this includes psi so it is somewhat larger but still < 0.1 GB).
        assert 0.005 < est["fields_gb"] < 0.05
        assert 0.01 < est["forward_gb"] < 0.1
        # Naive AD is the dominant, tens-of-GB figure; checkpointing cuts it
        # by more than an order of magnitude into the low-single-digit GB.
        assert 10.0 < est["adjoint_naive_gb"] < 100.0
        assert 0.5 < est["adjoint_checkpoint_gb"] < 5.0
        assert est["adjoint_checkpoint_gb"] < 0.2 * est["adjoint_naive_gb"]

    def test_strip_psi_cheaper_than_full(self):
        g = Grid3D(nx=82, ny=82, nz=51, dx=1.3e-3, dy=1.3e-3, dz=1.3e-3)
        est = fdtd3d_memory_estimate(
            g, 2000, n_dft_freqs=1, dtype="float32", cpml_thickness=10
        )
        # Slab psi storage spans only the PML planes => strictly cheaper.
        assert 0.0 < est["psi_strip_gb"] < est["psi_full_gb"]
        assert est["adjoint_checkpoint_strip_gb"] < est["adjoint_checkpoint_gb"]

    def test_dft_scales_with_freqs_and_dtype(self):
        g = Grid3D(nx=40, ny=40, nz=20, dx=1e-3, dy=1e-3, dz=1e-3)
        e0 = fdtd3d_memory_estimate(g, 1000, n_dft_freqs=0, dtype="float32")
        e1 = fdtd3d_memory_estimate(g, 1000, n_dft_freqs=1, dtype="float32")
        e4 = fdtd3d_memory_estimate(g, 1000, n_dft_freqs=4, dtype="float32")
        assert e0["dft_monitor_gb"] == 0.0
        # Complex accumulator: 2 floats x n_freqs x six field copies.
        np.testing.assert_allclose(e4["dft_monitor_gb"], 4 * e1["dft_monitor_gb"],
                                   rtol=1e-12)
        # float64 doubles every per-value figure.
        e1_64 = fdtd3d_memory_estimate(g, 1000, n_dft_freqs=1, dtype="float64")
        np.testing.assert_allclose(e1_64["fields_gb"], 2 * e1["fields_gb"],
                                   rtol=1e-12)

    def test_checkpoint_default_is_sqrt_n(self):
        g = Grid3D(nx=40, ny=40, nz=20, dx=1e-3, dy=1e-3, dz=1e-3)
        est = fdtd3d_memory_estimate(g, 2500, dtype="float32")
        assert est["checkpoint_segments"] == 50  # round(sqrt(2500))

    def test_gpu_fit_report_keys_and_monotonic(self):
        g = Grid3D(nx=82, ny=82, nz=51, dx=1.3e-3, dy=1.3e-3, dz=1.3e-3)
        est = fdtd3d_memory_estimate(g, 2000, dtype="float32", cpml_thickness=10)
        report = gpu_fit_report(est)
        assert set(report) == {24.0, 16.0, 8.0}
        # If it fits an 8 GB card it must fit the larger ones too.
        if report[8.0]:
            assert report[16.0] and report[24.0]
        assert fits_gpu(1.0, 24.0) and not fits_gpu(100.0, 24.0)
