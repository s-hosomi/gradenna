"""Beam-steering demo checks: array-factor agreement and gradient sanity.

Small grids / short runs so the whole module stays well under ~30 s. The
physics (multi-source phased line array, NTFF directivity, differentiable
carrier-delay phasing) mirrors examples/optimize_beamsteering.py but is set
up self-contained here to keep the test fast and independent.
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradenna import (
    C0,
    CPMLSpec,
    Grid2D,
    alpha_max_for_fmin,
    modulated_gaussian,
    simulate_tm,
)
from gradenna.ntff import directivity_2d, ntff_2d

# --- compact problem (lambda = 30 cells, lambda/2 = 15 cells spacing) --------

DX = 2e-3
F0 = 5.0e9
F_MIN = 3.0e9
N_ELEMENTS = 4
SPACING_CELLS = int(round(0.5 * (C0 / F0) / DX))  # ~15 cells = lambda/2
NTFF_MARGIN = 13
NX = NY = 2 * NTFF_MARGIN + (N_ELEMENTS - 1) * SPACING_CELLS + 40
N_STEPS = 1500
TAU = 8.0 / (2.0 * np.pi * F0)

GRID = Grid2D(nx=NX, ny=NY, dx=DX, dy=DX)
CPML = CPMLSpec(thickness=10, alpha_max=alpha_max_for_fmin(F_MIN))


def _element_positions():
    ic = GRID.nx // 2
    jc = GRID.ny // 2
    offsets = (np.arange(N_ELEMENTS) - 0.5 * (N_ELEMENTS - 1)) * SPACING_CELLS
    js = np.round(jc + offsets).astype(int)
    ijs = np.stack([np.full(N_ELEMENTS, ic), js], axis=1)
    y_rel = (js - 0.5 * (GRID.ny - 1)) * GRID.dy
    return ijs, jnp.asarray(y_rel)


def _theory_phases(y_rel, steer_deg):
    k = 2.0 * np.pi * F0 / C0
    return -k * np.asarray(y_rel) * np.sin(np.deg2rad(steer_deg))


def _currents(amps, phases):
    t = (jnp.arange(N_STEPS) + 0.5) * GRID.dt
    t0 = 6.0 * TAU
    # DFT kernel e^{-j w t}: a carrier delay Dt gives phasor phase -w*Dt, so
    # feed phase phi is realized by advancing the carrier (Dt = -phi/(2 pi f0)).
    delays = -phases / (2.0 * jnp.pi * F0)
    return jax.vmap(
        lambda d, a: a * modulated_gaussian(t, f0=F0, t0=t0 + d, tau=TAU),
        in_axes=(0, 0),
        out_axes=1,
    )(delays, amps)


def _directivity(source_ij, amps, phases, angles):
    res = simulate_tm(
        GRID,
        source_ij=source_ij,
        source_current=_currents(amps, phases),
        dft_freqs=(F0,),
        cpml=CPML,
    )
    e_far = ntff_2d(res.dft_ez, res.dft_hx, res.dft_hy, GRID, NTFF_MARGIN, (F0,), angles)
    return directivity_2d(e_far, angles)[0]


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


@pytest.mark.parametrize("steer_deg", [-30.0, 0.0, 30.0])
def test_main_lobe_matches_array_factor_theory(steer_deg):
    """Linear-phase feed steers the FDTD+NTFF main lobe to the predicted angle.

    With elements along y, progressive phase phi_n = -k y_n sin(theta_s)
    makes the array factor peak at theta_s (from the +x broadside axis).
    """
    source_ij, y_rel = _element_positions()
    phases = jnp.asarray(_theory_phases(y_rel, steer_deg))
    amps = jnp.ones(N_ELEMENTS)
    angles = jnp.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)

    d = np.asarray(_directivity(source_ij, amps, phases, angles))
    # Restrict to the forward half-plane (the array radiates symmetrically into
    # +/-x; the main lobe of interest is the forward beam near +x).
    ang_w = _wrap(np.asarray(angles))
    forward = np.abs(ang_w) <= np.deg2rad(80.0)
    idx = np.where(forward)[0]
    lobe_deg = np.rad2deg(ang_w[idx[np.argmax(d[idx])]])
    assert abs(lobe_deg - steer_deg) <= 5.0, (
        f"main lobe {lobe_deg:.1f} deg vs theory {steer_deg:.1f} deg"
    )


def test_grad_directivity_matches_finite_difference():
    """jax.grad of D(theta_s) w.r.t. feed weights vs central differences.

    Follows tests/test_gradients.py: random-direction directional derivative,
    step swept around eps^(1/3), best relative error taken.
    """
    source_ij, y_rel = _element_positions()
    steer_deg = 30.0
    target = np.deg2rad(steer_deg)
    angles = jnp.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
    ai = int(np.argmin(np.abs(_wrap(np.asarray(angles)) - target)))

    params0 = {
        "amps": jnp.ones(N_ELEMENTS),
        "phases": jnp.asarray(_theory_phases(y_rel, steer_deg)),
    }

    def loss(params):
        d = _directivity(source_ij, params["amps"], params["phases"], angles)
        return d[ai]

    from jax.flatten_util import ravel_pytree

    grad = jax.grad(loss)(params0)
    g_flat, _ = ravel_pytree(grad)
    flat, unravel = ravel_pytree(params0)
    assert bool(jnp.all(jnp.isfinite(g_flat)))
    assert float(jnp.abs(g_flat).max()) > 0.0

    rng = np.random.default_rng(0)
    v = rng.normal(size=flat.shape)
    v = jnp.asarray(v / np.linalg.norm(v))
    d_ad = float(jnp.vdot(g_flat, v))
    scale = float(jnp.linalg.norm(flat))

    def fd(h):
        fp = loss(unravel(flat + h * v))
        fm = loss(unravel(flat - h * v))
        return float((fp - fm) / (2.0 * h))

    errs = [abs(d_ad - fd(h * scale)) / max(abs(d_ad), abs(fd(h * scale)))
            for h in (3e-7, 1e-6, 3e-6)]
    assert min(errs) <= 1e-4, f"AD vs FD relative error {min(errs):.2e}"
