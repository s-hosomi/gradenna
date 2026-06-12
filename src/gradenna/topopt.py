"""Density-based topology optimization toolkit (three-field scheme).

Implements the standard photonics inverse-design recipe documented in
``docs/research/10-topology-optimization-theory.md``:

    latent theta -> sigmoid -> conic (hat) filter -> tanh projection -> rho

- :func:`conic_filter`: renormalized convolution with a conic kernel
  (Bourdin 2001; boundary handling by dividing by the local kernel mass,
  so constant fields are preserved exactly).
- :func:`tanh_projection`: smoothed Heaviside (Wang, Lazarov & Sigmund 2011).
- :class:`DesignTransform`: the composed, differentiable parameterization.
- :func:`beta_schedule`: standard doubling continuation (beta = 8 -> 64).
- :func:`gray_indicator`: binarization metric 4*rho*(1-rho).
- :func:`connected_to_seed`, :func:`minimum_feature_size`: non-differentiable
  numpy/scipy post-processing checks (feed connectivity, minimum linewidth).
- :func:`optimize`: optax.adam loop with beta continuation.

All functions above the post-processing section are JAX-differentiable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.scipy.signal import fftconvolve
from scipy import ndimage

__all__ = [
    "DesignTransform",
    "beta_schedule",
    "conic_filter",
    "connected_to_seed",
    "gray_indicator",
    "minimum_feature_size",
    "optimize",
    "tanh_projection",
]


# ---------------------------------------------------------------------------
# Filtering and projection (differentiable)
# ---------------------------------------------------------------------------


def _conic_kernel(radius_cells: float, ndim: int = 2) -> np.ndarray:
    """Normalized conic (hat) kernel w(d) = max(0, 1 - d/R) on the cell grid.

    Built for any number of dimensions (1D/2D/3D, ...): the kernel is the
    radially symmetric hat function sampled on the integer lattice.
    """
    radius = float(radius_cells)
    # Largest integer offset with strictly positive weight (d < R).
    n = int(np.ceil(radius)) - 1
    offsets = np.meshgrid(*([np.arange(-n, n + 1)] * ndim), indexing="ij")
    d2 = sum(o.astype(np.float64) ** 2 for o in offsets)
    w = np.maximum(0.0, 1.0 - np.sqrt(d2) / radius)
    return w / w.sum()


def conic_filter(rho: jax.Array, radius_cells: float) -> jax.Array:
    """Conic (hat) density filter with renormalized boundary handling.

    Computes ``(K * rho) / (K * 1)`` via FFT convolution, i.e. the kernel mass
    that falls inside the domain is used for normalization near boundaries.
    Constant fields are therefore preserved exactly (up to FFT roundoff), and
    total mass is conserved for fields supported at least ``radius_cells``
    away from the boundary.

    Args:
        rho: density field of any dimensionality (the conic kernel is built
            with ``rho.ndim`` axes, so 2D and 3D design regions both work).
        radius_cells: filter radius R in units of grid cells (static Python
            number). ``R <= 1`` makes the kernel a single cell, i.e. identity.

    Returns:
        Filtered field with the same shape as ``rho``.
    """
    radius = float(radius_cells)
    if radius <= 1.0:
        # Kernel support is a single cell: the filter is the identity.
        return rho
    kernel = jnp.asarray(_conic_kernel(radius, rho.ndim), dtype=rho.dtype)
    num = fftconvolve(rho, kernel, mode="same")
    den = fftconvolve(jnp.ones_like(rho), kernel, mode="same")
    return num / den


def tanh_projection(rho: jax.Array, beta, eta: float = 0.5) -> jax.Array:
    """Smoothed Heaviside projection of Wang, Lazarov & Sigmund (2011).

    rho_bar = (tanh(beta*eta) + tanh(beta*(rho - eta)))
              / (tanh(beta*eta) + tanh(beta*(1 - eta)))

    ``beta = 1`` is close to the identity; ``beta -> inf`` approaches a sharp
    threshold at ``eta``.
    """
    beta = jnp.asarray(beta, dtype=rho.dtype)
    num = jnp.tanh(beta * eta) + jnp.tanh(beta * (rho - eta))
    den = jnp.tanh(beta * eta) + jnp.tanh(beta * (1.0 - eta))
    return num / den


@dataclass(frozen=True)
class DesignTransform:
    """Three-field design parameterization: theta -> sigmoid -> filter -> project.

    The latent variable ``theta`` is unconstrained; ``sigmoid`` maps it to
    [0, 1], the conic filter enforces a length scale, and the tanh projection
    pushes the result toward a binary design as ``beta`` grows.

    Attributes:
        radius_cells: conic filter radius in grid cells.
        eta: projection threshold (0.5 for the nominal/blueprint design).
    """

    radius_cells: float
    eta: float = 0.5

    def __call__(self, theta: jax.Array, beta) -> jax.Array:
        rho = jax.nn.sigmoid(theta)
        rho = conic_filter(rho, self.radius_cells)
        return tanh_projection(rho, beta, self.eta)


def beta_schedule(
    betas: Sequence[float] = (8.0, 16.0, 32.0, 64.0),
    steps_per_beta: int = 100,
) -> Callable[[int], float]:
    """Standard beta-continuation schedule (doubling every ``steps_per_beta``).

    Returns a function mapping the iteration number (0-based) to the value of
    beta for that iteration. Iterations past the last stage stay at the final
    beta.
    """
    if steps_per_beta < 1:
        raise ValueError("steps_per_beta must be >= 1")
    stages = tuple(float(b) for b in betas)
    if not stages:
        raise ValueError("betas must be non-empty")

    def schedule(step: int) -> float:
        idx = min(step // steps_per_beta, len(stages) - 1)
        return stages[idx]

    return schedule


def gray_indicator(rho: jax.Array) -> jax.Array:
    """Mean of 4*rho*(1-rho): 0 for a fully binary field, 1 at rho = 0.5."""
    return jnp.mean(4.0 * rho * (1.0 - rho))


# ---------------------------------------------------------------------------
# Post-processing checks (numpy, non-differentiable)
# ---------------------------------------------------------------------------


def connected_to_seed(rho_binary, seed_ij: tuple[int, int]) -> np.ndarray:
    """Boolean mask of the solid connected component containing the seed.

    Uses ``scipy.ndimage.label`` with the default 4-connectivity. Intended as
    a post-processing step to detect copper islands not connected to the feed
    point; isolated islands are simply absent from the returned mask.

    Args:
        rho_binary: 2D array; values > 0.5 are treated as solid.
        seed_ij: (i, j) index of the feed point.

    Returns:
        Boolean array, True on the component connected to the seed. All False
        if the seed pixel itself is not solid.
    """
    solid = np.asarray(rho_binary) > 0.5
    labels, _ = ndimage.label(solid)
    seed_label = labels[tuple(seed_ij)]
    if seed_label == 0:
        return np.zeros_like(solid, dtype=bool)
    return labels == seed_label


def _disk_footprint(width_cells: int) -> np.ndarray:
    """Discrete disk footprint whose diameter matches ``width_cells``."""
    radius = (width_cells - 1) / 2.0
    n = int(np.floor(radius))
    y, x = np.mgrid[-n : n + 1, -n : n + 1]
    return (x * x + y * y) <= radius * radius + 1e-9


def minimum_feature_size(rho_binary, width_cells: int) -> np.ndarray:
    """Detect solid pixels violating the minimum linewidth.

    Performs a binary opening with a disk footprint of diameter
    ``width_cells``; solid pixels removed by the opening belong to features
    thinner than the minimum width (imageruler-style morphological check).

    Args:
        rho_binary: 2D array; values > 0.5 are treated as solid.
        width_cells: minimum linewidth in grid cells.

    Returns:
        Boolean array, True where the minimum-width rule is violated.
        ``width_cells <= 1`` never flags anything.
    """
    solid = np.asarray(rho_binary) > 0.5
    if width_cells <= 1:
        return np.zeros_like(solid, dtype=bool)
    footprint = _disk_footprint(int(width_cells))
    # Erode with border_value=1 so features clipped by the domain edge are not
    # spuriously flagged as too thin.
    eroded = ndimage.binary_erosion(solid, structure=footprint, border_value=1)
    opened = ndimage.binary_dilation(eroded, structure=footprint, border_value=0)
    return solid & ~opened


# ---------------------------------------------------------------------------
# Optimization driver
# ---------------------------------------------------------------------------


def optimize(
    loss_fn: Callable,
    theta0: jax.Array,
    n_steps: int = 400,
    betas: Sequence[float] = (8.0, 16.0, 32.0, 64.0),
    learning_rate: float = 0.02,
    transform: DesignTransform | None = None,
    n_snapshots: int = 4,
) -> dict:
    """Adam loop with beta continuation over the three-field parameterization.

    Args:
        loss_fn: callable ``loss_fn(rho, beta) -> scalar`` taking the projected
            density and the current beta. Must be JAX-differentiable in rho.
        theta0: initial latent design (unconstrained).
        n_steps: total number of iterations, split evenly across the betas.
        betas: beta-continuation stages (doubling schedule by default).
        learning_rate: Adam learning rate.
        transform: design parameterization; defaults to
            ``DesignTransform(radius_cells=2.0)``.
        n_snapshots: number of density snapshots stored in the history.

    Returns:
        dict with keys:
            "theta": final latent design,
            "rho": final projected density (at the final beta),
            "loss": per-step loss values (np.ndarray),
            "gray": per-step gray indicator of rho (np.ndarray),
            "beta": per-step beta values (np.ndarray),
            "snapshots": list of (step, density np.ndarray) pairs.
    """
    if transform is None:
        transform = DesignTransform(radius_cells=2.0)
    steps_per_beta = max(1, n_steps // len(tuple(betas)))
    schedule = beta_schedule(betas, steps_per_beta)

    opt = optax.adam(learning_rate)
    opt_state = opt.init(theta0)
    theta = theta0

    def total_loss(theta, beta):
        rho = transform(theta, beta)
        return loss_fn(rho, beta), rho

    @jax.jit
    def step(theta, opt_state, beta):
        (loss, rho), grads = jax.value_and_grad(total_loss, has_aux=True)(theta, beta)
        updates, opt_state = opt.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        return theta, opt_state, loss, rho

    snapshot_steps = set(
        int(round(s)) for s in np.linspace(0, n_steps - 1, max(1, n_snapshots))
    )

    losses, grays, beta_hist, snapshots = [], [], [], []
    rho = transform(theta, schedule(0))
    for i in range(n_steps):
        beta = schedule(i)
        # beta enters as a traced array so jit compiles only once.
        theta, opt_state, loss, rho = step(theta, opt_state, jnp.asarray(beta))
        losses.append(float(loss))
        grays.append(float(gray_indicator(rho)))
        beta_hist.append(beta)
        if i in snapshot_steps:
            snapshots.append((i, np.asarray(rho)))

    # The loop's rho lags one update behind theta; re-project the final theta
    # so "rho" really is the final density at the final beta (docstring).
    rho = transform(theta, schedule(max(0, n_steps - 1)))

    return {
        "theta": theta,
        "rho": rho,
        "loss": np.asarray(losses),
        "gray": np.asarray(grays),
        "beta": np.asarray(beta_hist),
        "snapshots": snapshots,
    }
