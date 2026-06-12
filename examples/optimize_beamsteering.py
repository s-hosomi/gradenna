"""Phase 5 demo: beam steering of a small phased line array (2D TM).

A 4-element uniformly spaced line array of point (line) sources is excited
through the multi-source path of `simulate_tm`. Each element n carries a
complex feed weight w_n = a_n * exp(i*phi_n); the far-field array factor of
the array (elements along y at positions y_n) is

    AF(theta) = sum_n w_n exp(i k y_n sin(theta)),

so a progressive phase phi_n = -k y_n sin(theta_s) steers the main lobe to
theta_s (theta measured from the +x broadside axis). With lambda/2 spacing
the visible region is grating-lobe-free. The elements are isotropic line
sources with no reflector, so the pattern is mirror-symmetric about the y
array axis: each steered forward beam at theta_s has an equal back lobe at
180 - theta_s. We report and optimize the forward (+x) beam.

Differentiable phasing
----------------------
The solver injects *real* time-domain currents, so the complex weight is
realized as an amplitude scaling plus a carrier time delay,

    i_n(t) = a_n * modulated_gaussian(t, f0, t0 - phi_n / (2 pi f0), tau),

since the running-DFT kernel e^{-j w t} turns a carrier delay Dt into phasor
phase -w*Dt, so phase phi_n is realized by advancing the carrier by
phi_n/(2 pi f0). This is smooth in both a_n and phi_n, so D(theta_s) is
differentiable w.r.t. the feed weights and we optimize them with optax to
maximize radiation toward each target steering angle.

Run:  JAX_ENABLE_X64=1 uv run python examples/optimize_beamsteering.py
      JAX_ENABLE_X64=1 uv run python examples/optimize_beamsteering.py --quick
(a couple of minutes on a laptop CPU; float64 recommended). Writes
optimize_beamsteering.png next to this script unless --no-plot is given.
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")  # before jax import

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from gradenna import (
    C0,
    CPMLSpec,
    Grid2D,
    alpha_max_for_fmin,
    modulated_gaussian,
    simulate_tm,
)
from gradenna.ntff import directivity_2d, ntff_2d

# --- problem definition -----------------------------------------------------

DX = 2e-3  # 2 mm cells
F0 = 5.0e9  # operating frequency: lambda = 60 mm = 30 cells
F_MIN = 3.0e9  # low edge of the excitation band (sets CPML alpha)
N_ELEMENTS = 4  # array elements
NTFF_MARGIN = 15  # NTFF contour: CPML (10) + 5 cells

# Geometry: lambda = C0 / F0 = 60 mm -> 30 cells; spacing d = lambda/2 = 15 cells.
WAVELENGTH = C0 / F0
SPACING_CELLS = int(round(0.5 * WAVELENGTH / DX))  # ~15 cells = lambda/2
# A box large enough that the array (~3*d = 45 cells) and the NTFF contour
# both sit comfortably inside, with >lambda/2 clearance to the contour.
NY = 2 * NTFF_MARGIN + (N_ELEMENTS - 1) * SPACING_CELLS + 60
NX = NY  # square box; array radiates broadside along +/-x

N_ANGLES = 180  # far-field sampling for the loss (2 deg); angle 0 = +x

# --- optimization hyperparameters -------------------------------------------

LEARNING_RATE = 0.05
N_ITERS = 120
N_ITERS_QUICK = 30
STEER_ANGLES_DEG = (-30.0, 0.0, 30.0)


def _build_grid():
    return Grid2D(nx=NX, ny=NY, dx=DX, dy=DX)


def _element_positions(grid):
    """(i, j) cells of the N elements, centered, spaced along +y."""
    ic = grid.nx // 2
    jc = grid.ny // 2
    offsets = (np.arange(N_ELEMENTS) - 0.5 * (N_ELEMENTS - 1)) * SPACING_CELLS
    js = np.round(jc + offsets).astype(int)
    ijs = np.stack([np.full(N_ELEMENTS, ic), js], axis=1)
    # Physical y of each element relative to the grid center (NTFF reference).
    y_rel = (js - 0.5 * (grid.ny - 1)) * grid.dy
    return ijs, jnp.asarray(y_rel)


def theory_phases(y_rel, steer_deg):
    """Progressive phases phi_n = -k y_n sin(theta_s) that steer to theta_s."""
    k = 2.0 * np.pi * F0 / C0
    theta_s = np.deg2rad(steer_deg)
    return -k * np.asarray(y_rel) * np.sin(theta_s)


def make_currents(grid, n_steps, amps, phases, tau):
    """(n_steps, N) real per-element currents a_n*mod_gauss(t; delay phi_n)."""
    t = (jnp.arange(n_steps) + 0.5) * grid.dt
    t0 = 6.0 * tau  # envelope launch time (well inside the window)
    # Phasor convention: the solver accumulates the DFT with kernel e^{-j w t},
    # so a carrier delay Dt yields phasor phase -w*Dt. To realize feed phase
    # phi_n the carrier must therefore be *advanced*: Dt_n = -phi_n/(2 pi f0).
    delays = -phases / (2.0 * jnp.pi * F0)  # phase -> carrier time delay
    # (n_steps, N): each column is the carrier delayed by its element phase.
    cols = jax.vmap(
        lambda d, a: a * modulated_gaussian(t, f0=F0, t0=t0 + d, tau=tau),
        in_axes=(0, 0),
        out_axes=1,
    )(delays, amps)
    return cols


def far_field_pattern(grid, cpml, source_ij, amps, phases, angles, n_steps, tau):
    """Differentiable NTFF directivity D(angle) of the phased array at f0."""
    currents = make_currents(grid, n_steps, amps, phases, tau)
    res = simulate_tm(
        grid,
        source_ij=source_ij,
        source_current=currents,
        dft_freqs=(F0,),
        cpml=cpml,
    )
    e_far = ntff_2d(res.dft_ez, res.dft_hx, res.dft_hy, grid, NTFF_MARGIN, (F0,), angles)
    return directivity_2d(e_far, angles)[0]  # (n_angles,)


# --- optimization -----------------------------------------------------------


def optimize_for_angle(grid, cpml, source_ij, angles, steer_deg, n_iters, n_steps, tau,
                       init_phases):
    """Gradient-ascent on the feed weights to maximize D toward theta_s."""
    target = np.deg2rad(steer_deg)
    # Loss = -D at the steering angle (continuous interpolation onto the grid
    # would be overkill; the angle grid contains the steering angles exactly).
    ai = int(np.argmin(np.abs(_wrap(np.asarray(angles)) - target)))

    params = {
        "amps": jnp.ones(N_ELEMENTS),  # start from uniform amplitude
        "phases": jnp.asarray(init_phases),  # warm-start at the theory phases
    }
    opt = optax.adam(LEARNING_RATE)
    opt_state = opt.init(params)

    def loss_fn(params):
        d = far_field_pattern(
            grid, cpml, source_ij, params["amps"], params["phases"],
            angles, n_steps, tau,
        )
        return -d[ai], d

    @jax.jit
    def step(params, opt_state):
        (loss, d), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, d

    history = []
    for i in range(n_iters):
        params, opt_state, loss, d = step(params, opt_state)
        history.append(-float(loss))
    d_final = np.asarray(d)
    return params, d_final, history


def _wrap(angles):
    """Wrap radians to (-pi, pi]."""
    return (np.asarray(angles) + np.pi) % (2.0 * np.pi) - np.pi


def main_lobe_angle_deg(d_pattern, angles, half_window_deg=80.0):
    """Forward-beam peak direction in degrees, wrapped to (-180, 180].

    Point-source elements (no reflector) radiate symmetrically into +/-x, so
    every steered beam has a mirror lobe at 180 - theta_s. We report the
    *forward* lobe (near +x, |theta| <= half_window) which is the steered beam
    of interest; the back lobe is its exact mirror.
    """
    ang_w = _wrap(np.asarray(angles))
    fwd = np.where(np.abs(ang_w) <= np.deg2rad(half_window_deg))[0]
    ai = fwd[int(np.argmax(np.asarray(d_pattern)[fwd]))]
    return float(np.rad2deg(ang_w[ai]))


# --- driver -----------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fewer iterations/steps")
    parser.add_argument("--no-plot", action="store_true", help="skip the figure")
    args = parser.parse_args()

    n_iters = N_ITERS_QUICK if args.quick else N_ITERS
    n_steps = 1400 if args.quick else 2200
    tau = 8.0 / (2.0 * np.pi * F0)  # narrowband pulse around f0

    grid = _build_grid()
    cpml = CPMLSpec(thickness=10, alpha_max=alpha_max_for_fmin(F_MIN))
    source_ij, y_rel = _element_positions(grid)
    angles = jnp.linspace(0.0, 2.0 * np.pi, N_ANGLES, endpoint=False)

    print(
        f"{N_ELEMENTS}-element array, spacing {SPACING_CELLS} cells "
        f"(lambda/2 = {0.5 * WAVELENGTH * 1e3:.1f} mm) at {F0 / 1e9:.1f} GHz, "
        f"grid {NX}x{NY}, {n_steps} FDTD steps, {n_iters} iters/angle"
    )

    results = {}
    t_start = time.time()
    for steer in STEER_ANGLES_DEG:
        phi0 = theory_phases(y_rel, steer)
        params, d_final, hist = optimize_for_angle(
            grid, cpml, source_ij, angles, steer, n_iters, n_steps, tau, phi0,
        )
        lobe = main_lobe_angle_deg(d_final, angles)
        ai = int(np.argmin(np.abs(_wrap(np.asarray(angles)) - np.deg2rad(steer))))
        d_at_target = float(d_final[ai])
        results[steer] = (params, d_final, hist, lobe, d_at_target)
        print(
            f"\nsteer {steer:+6.1f} deg | theory phases (deg): "
            f"{np.rad2deg(phi0).round(1)}"
        )
        print(
            f"  optimized phases (deg): "
            f"{np.rad2deg(np.asarray(params['phases'])).round(1)}"
        )
        print(f"  optimized amps:         {np.asarray(params['amps']).round(3)}")
        print(
            f"  main-lobe direction: {lobe:+6.1f} deg  "
            f"(target {steer:+.1f} deg, err {lobe - steer:+.1f} deg)"
        )
        print(
            f"  D(target) {d_at_target:.3f}  peak D {float(d_final.max()):.3f}  "
            f"(isotropic D = 1)"
        )
    print(f"\ntotal time: {time.time() - t_start:.0f} s")

    if not args.no_plot:
        _plot(grid, angles, results)


def _plot(grid, angles, results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1, projection="polar")
    ax.set_theta_zero_location("E")
    floor = -20.0
    for c, (steer, (_, d_final, _, lobe, _)) in zip(("C0", "C1", "C2"), results.items()):
        d_db = 10.0 * np.log10(np.maximum(np.asarray(d_final), 1e-3))
        ax.plot(
            np.asarray(angles), np.maximum(d_db, floor), c,
            label=f"steer {steer:+.0f} deg (lobe {lobe:+.0f})",
        )
        ax.plot([np.deg2rad(steer)], [floor], c + "v")
    ax.set_rmin(floor)
    ax.set_title(f"steered far-field directivity at {F0 / 1e9:.1f} GHz [dB]")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.1), fontsize=8)

    ax = fig.add_subplot(1, 2, 2)
    for c, (steer, (_, _, hist, _, _)) in zip(("C0", "C1", "C2"), results.items()):
        ax.plot(hist, c, label=f"steer {steer:+.0f} deg")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$D(\theta_s)$")
    ax.set_title("convergence of directivity toward target")
    ax.legend()

    fig.suptitle("Small phased line array: differentiable beam steering")
    fig.tight_layout()
    out = Path(__file__).with_name("optimize_beamsteering.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
