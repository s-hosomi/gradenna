"""Acceptance tests for the topology-optimization toolkit (gradenna.topopt)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradenna.topopt import (
    DesignTransform,
    beta_schedule,
    conic_filter,
    connected_to_seed,
    gray_indicator,
    minimum_feature_size,
    optimize,
    tanh_projection,
)


# ---------------------------------------------------------------------------
# conic_filter
# ---------------------------------------------------------------------------


class TestConicFilter:
    def test_constant_field_preserved_including_boundary(self):
        rho = jnp.full((33, 41), 0.37)
        out = conic_filter(rho, radius_cells=4.5)
        np.testing.assert_allclose(np.asarray(out), 0.37, rtol=0.0, atol=1e-12)

    def test_mass_conservation_interior(self):
        # Blob supported well away from the boundary: renormalization is
        # inactive there and the normalized kernel conserves total mass.
        rho = np.zeros((48, 48))
        rho[20:28, 18:30] = 1.0
        rho[22:25, 22:26] = 0.5
        out = conic_filter(jnp.asarray(rho), radius_cells=4.0)
        np.testing.assert_allclose(float(jnp.sum(out)), rho.sum(), rtol=1e-12)

    def test_radius_zero_is_identity(self):
        key = jax.random.PRNGKey(0)
        rho = jax.random.uniform(key, (17, 23))
        np.testing.assert_array_equal(
            np.asarray(conic_filter(rho, 0.0)), np.asarray(rho)
        )
        # Radius <= 1 cell means the kernel support is a single cell.
        np.testing.assert_array_equal(
            np.asarray(conic_filter(rho, 1.0)), np.asarray(rho)
        )

    def test_output_range_and_smoothing(self):
        key = jax.random.PRNGKey(1)
        rho = (jax.random.uniform(key, (32, 32)) > 0.5).astype(jnp.float64)
        out = conic_filter(rho, radius_cells=3.0)
        assert float(out.min()) >= -1e-12
        assert float(out.max()) <= 1.0 + 1e-12
        # Filtering must reduce total variation of a random binary field.
        tv = lambda a: float(jnp.abs(jnp.diff(a, axis=0)).sum())
        assert tv(out) < tv(rho)


# ---------------------------------------------------------------------------
# tanh_projection
# ---------------------------------------------------------------------------


class TestTanhProjection:
    def test_large_beta_snaps_to_binary(self):
        rho = jnp.asarray([0.0, 0.1, 0.4, 0.6, 0.9, 1.0])
        out = np.asarray(tanh_projection(rho, beta=1000.0, eta=0.5))
        np.testing.assert_allclose(out, [0, 0, 0, 1, 1, 1], atol=1e-10)

    def test_eta_half_symmetry(self):
        # P(1 - rho) = 1 - P(rho) for eta = 0.5.
        rho = jnp.linspace(0.0, 1.0, 21)
        for beta in (1.0, 8.0, 64.0):
            p = np.asarray(tanh_projection(rho, beta, eta=0.5))
            q = np.asarray(tanh_projection(1.0 - rho, beta, eta=0.5))
            np.testing.assert_allclose(p, 1.0 - q, atol=1e-12)

    def test_value_half_at_eta(self):
        for eta in (0.3, 0.5, 0.7):
            for beta in (2.0, 16.0, 64.0):
                out = float(tanh_projection(jnp.asarray(eta), beta, eta=eta))
                if eta == 0.5:
                    np.testing.assert_allclose(out, 0.5, atol=1e-12)
                else:
                    # rho = eta always maps to tanh(beta*eta)/den.
                    expected = float(
                        jnp.tanh(beta * eta)
                        / (jnp.tanh(beta * eta) + jnp.tanh(beta * (1 - eta)))
                    )
                    np.testing.assert_allclose(out, expected, atol=1e-12)

    def test_endpoints_fixed(self):
        for beta in (1.0, 8.0, 64.0):
            np.testing.assert_allclose(
                float(tanh_projection(jnp.asarray(0.0), beta)), 0.0, atol=1e-12
            )
            np.testing.assert_allclose(
                float(tanh_projection(jnp.asarray(1.0), beta)), 1.0, atol=1e-12
            )


# ---------------------------------------------------------------------------
# DesignTransform
# ---------------------------------------------------------------------------


class TestDesignTransform:
    def test_gradient_finite_and_nonzero(self):
        transform = DesignTransform(radius_cells=3.0)
        key = jax.random.PRNGKey(2)
        theta = jax.random.normal(key, (20, 20))

        def objective(theta):
            rho = transform(theta, beta=8.0)
            return jnp.sum(rho**2)

        grad = jax.grad(objective)(theta)
        assert bool(jnp.all(jnp.isfinite(grad)))
        assert float(jnp.linalg.norm(grad)) > 0.0

    def test_output_in_unit_interval(self):
        transform = DesignTransform(radius_cells=2.0)
        theta = jnp.asarray(np.random.default_rng(0).normal(size=(16, 16)) * 5)
        rho = transform(theta, beta=32.0)
        assert float(rho.min()) >= -1e-12
        assert float(rho.max()) <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# beta_schedule / gray_indicator
# ---------------------------------------------------------------------------


class TestBetaSchedule:
    def test_stages_and_saturation(self):
        sched = beta_schedule(betas=(8, 16, 32, 64), steps_per_beta=100)
        assert sched(0) == 8.0
        assert sched(99) == 8.0
        assert sched(100) == 16.0
        assert sched(250) == 32.0
        assert sched(399) == 64.0
        assert sched(10_000) == 64.0  # stays at the final beta


class TestGrayIndicator:
    def test_binary_is_zero(self):
        rho = jnp.asarray(np.random.default_rng(1).integers(0, 2, (12, 12)))
        np.testing.assert_allclose(float(gray_indicator(rho)), 0.0, atol=1e-12)

    def test_half_is_one(self):
        rho = jnp.full((9, 9), 0.5)
        np.testing.assert_allclose(float(gray_indicator(rho)), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Post-processing (connectivity, minimum feature size)
# ---------------------------------------------------------------------------


class TestConnectedToSeed:
    def test_keeps_seed_component_drops_island(self):
        rho = np.zeros((20, 20))
        rho[2:5, 2:15] = 1.0  # trace connected to the feed
        rho[10:13, 10:13] = 1.0  # isolated island
        mask = connected_to_seed(rho, seed_ij=(3, 3))
        assert mask[3, 10]  # on the connected trace
        assert not mask[11, 11]  # island excluded
        assert mask.sum() == (rho[2:5, 2:15] > 0).sum()

    def test_seed_on_void_returns_empty(self):
        rho = np.zeros((10, 10))
        rho[7:9, 7:9] = 1.0
        mask = connected_to_seed(rho, seed_ij=(0, 0))
        assert not mask.any()


class TestMinimumFeatureSize:
    def test_thick_line_passes(self):
        rho = np.zeros((24, 24))
        rho[10:15, :] = 1.0  # 5-cell-wide line spanning the domain
        violations = minimum_feature_size(rho, width_cells=3)
        assert not violations.any()

    def test_thin_line_flagged(self):
        rho = np.zeros((24, 24))
        rho[12, :] = 1.0  # 1-cell-wide line
        violations = minimum_feature_size(rho, width_cells=3)
        assert violations.any()
        assert violations[12].all()

    def test_width_one_never_flags(self):
        rho = np.zeros((8, 8))
        rho[4, 4] = 1.0
        assert not minimum_feature_size(rho, width_cells=1).any()


# ---------------------------------------------------------------------------
# optimize
# ---------------------------------------------------------------------------


class TestOptimize:
    @pytest.fixture(scope="class")
    def result(self):
        n = 24
        target = np.zeros((n, n))
        target[7:17, 7:17] = 1.0
        target = jnp.asarray(target)

        def loss_fn(rho, beta):
            return jnp.mean((rho - target) ** 2)

        theta0 = jnp.zeros((n, n))
        return optimize(
            loss_fn,
            theta0,
            n_steps=200,
            betas=(8, 16, 32, 64),
            learning_rate=0.05,
            transform=DesignTransform(radius_cells=2.0),
            n_snapshots=4,
        )

    def test_loss_converges(self, result):
        loss = result["loss"]
        assert loss[-1] < 0.05 * loss[0]
        assert loss[-1] < 0.01

    def test_gray_decreases_with_beta(self, result):
        gray, beta = result["gray"], result["beta"]
        # Mean gray level per beta stage must be monotonically decreasing.
        stage_means = [gray[beta == b].mean() for b in (8, 16, 32, 64)]
        assert all(a > b for a, b in zip(stage_means, stage_means[1:]))
        assert gray[-1] < 0.05  # nearly binary at the final beta

    def test_history_shapes_and_snapshots(self, result):
        assert result["loss"].shape == (200,)
        assert result["gray"].shape == (200,)
        assert result["beta"].shape == (200,)
        assert len(result["snapshots"]) == 4
        for step, density in result["snapshots"]:
            assert 0 <= step < 200
            assert density.shape == (24, 24)
        assert result["rho"].shape == (24, 24)
