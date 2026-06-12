"""Phase 5 demo: NTFF-based directive beam forming at 2.45 GHz.

The setup follows examples/optimize_2d_antenna.py (2D TM, lumped 50 ohm RVS
port embedded in a conductivity design region, log-sigma interpolation,
three-field parameterization with beta continuation), but the objective is
computed in the *far field* through the differentiable NTFF of
gradenna.ntff (research note 13): maximize the realized-gain proxy toward
the +x direction,

    G_proxy(phi=0) = D(phi=0) * P_rad / P_avail = 2 pi U(phi=0) / P_avail,

with U(phi) = |E_far(phi)|^2 / (2 eta0) the NTFF radiation intensity and
P_avail = |Vs|^2 / (8 Rs) the available source power. Maximizing the
directivity alone would be satisfied by a design that radiates almost
nothing (and superdirective, high-Q solutions; note 13 Sec. 6.3); the
efficiency factor keeps the optimizer honest, exactly like the
radiated-energy objective of the Phase 3 demo.

Geometry: the port sits at the *center* of a 104 x 104 mm design region
(> lambda/2 = 61 mm at 2.45 GHz on every side of the feed, the resonant
feature threshold found in Phase 3), so reflector material can grow on the
-x side and director material on the +x side.

Run:  JAX_ENABLE_X64=1 uv run python examples/optimize_directivity.py
(~15 min on a laptop CPU; float64 is required, see the Phase 3 demo
docstring.) Writes optimize_directivity.png next to this script.
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")  # before jax import

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax

from gradenna import (
    CPMLSpec,
    Grid2D,
    Port,
    alpha_max_for_fmin,
    gaussian_pulse_for_band,
    half_step_dft,
    sigma_from_density,
    simulate_tm,
)
from gradenna.ntff import directivity_2d, ntff_2d, radiated_power_2d
from gradenna.topopt import DesignTransform, beta_schedule, gray_indicator

# --- problem definition -----------------------------------------------------

DX = 2e-3  # 2 mm cells: lambda/61 at 2.45 GHz
NX = NY = 140  # 280 mm box
F0 = 2.45e9  # objective frequency
F_MIN, F_MAX = 1.5e9, 3.5e9  # -20 dB band of the excitation
RS = 50.0  # port resistance [ohm m]
N_STEPS = 3500  # pulse + ring-down of resonant designs (Phase 3 demo)
PORT_IJ = (70, 70)  # feed at the center of the design region
DESIGN = (slice(44, 96), slice(44, 96))  # 52x52 cells = 104 x 104 mm
NTFF_MARGIN = 15  # NTFF contour: CPML (10) + 5 cells
SIGMA_MAX, SIGMA_MIN = 1e5, 1e-4  # log-sigma interpolation endpoints [S/m]
N_ANGLES = 72  # far-field sampling for the loss (5 deg); angle 0 = +x

# --- optimization hyperparameters (note 10 Sec. 9) ---------------------------

FILTER_RADIUS = 3.0  # conic filter radius [cells]
BETAS = (8.0, 16.0, 32.0, 64.0)
ITERS_PER_BETA = 45  # 180 iterations total
LEARNING_RATE = 0.15

grid = Grid2D(nx=NX, ny=NY, dx=DX, dy=DX)
cpml = CPMLSpec(thickness=10, alpha_max=alpha_max_for_fmin(F_MIN))
pulse = gaussian_pulse_for_band(F_MIN, F_MAX)
t = (jnp.arange(N_STEPS) + 0.5) * grid.dt
vs = pulse(t)
# Available source power (spectral) |Vs_hat|^2 / (8 Rs) (note 12 Sec. 3.1b).
p_avail_f0 = jnp.abs(half_step_dft(vs, grid.dt, F0)[0]) ** 2 / (8.0 * RS)
ANGLES = jnp.linspace(0.0, 2.0 * np.pi, N_ANGLES, endpoint=False)

n_des = DESIGN[0].stop - DESIGN[0].start
# Feed clearance: the port cell itself never carries design conductivity.
_mask = np.ones((n_des, n_des), bool)
_mask[PORT_IJ[0] - DESIGN[0].start, PORT_IJ[1] - DESIGN[1].start] = False
design_mask = jnp.asarray(_mask)


def simulate_design(rho, dft_freqs, n_steps=N_STEPS):
    """FDTD run of the design density rho (port + DFT field monitors)."""
    tt = (jnp.arange(n_steps) + 0.5) * grid.dt
    vs_t = pulse(tt)
    sig_design = jnp.where(design_mask, sigma_from_density(rho, SIGMA_MIN, SIGMA_MAX), 0.0)
    sigma = jnp.zeros(grid.shape).at[DESIGN].set(sig_design)
    return simulate_tm(
        grid,
        ports=(Port(ij=PORT_IJ, resistance=RS, voltage=vs_t),),
        sigma=sigma,
        dft_freqs=dft_freqs,
        cpml=cpml,
    )


def far_field(rho, angles, n_steps=N_STEPS):
    """NTFF pattern amplitude E_far(f0, phi) of the design (differentiable)."""
    res = simulate_design(rho, (F0,), n_steps)
    return ntff_2d(res.dft_ez, res.dft_hx, res.dft_hy, grid, NTFF_MARGIN, (F0,), angles)


def gain_metrics(rho):
    """(G_proxy(0), D(0), e_rad) of the design at f0; all differentiable."""
    e_far = far_field(rho, ANGLES)
    d = directivity_2d(e_far, ANGLES)[0, 0]  # D toward phi = 0 (+x)
    e_rad = radiated_power_2d(e_far, ANGLES)[0] / p_avail_f0
    return d * e_rad, d, e_rad


# --- optimization loop -------------------------------------------------------


def main():
    transform = DesignTransform(radius_cells=FILTER_RADIUS)
    schedule = beta_schedule(BETAS, ITERS_PER_BETA)
    n_iters = ITERS_PER_BETA * len(BETAS)

    theta = jnp.zeros((n_des, n_des))  # sigmoid(0) = 0.5: uniform gray start
    opt = optax.adam(LEARNING_RATE)
    opt_state = opt.init(theta)

    def objective(theta, beta):
        rho = transform(theta, beta)
        g, d, e_rad = gain_metrics(rho)
        return -g, (rho, d, e_rad)

    @jax.jit
    def step(theta, opt_state, beta):
        (loss, aux), grads = jax.value_and_grad(objective, has_aux=True)(theta, beta)
        updates, opt_state = opt.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        return theta, opt_state, loss, aux, grads

    print(
        f"design region {n_des}x{n_des} cells, {n_iters} iterations, "
        f"betas {BETAS}, lr {LEARNING_RATE}, {N_STEPS} FDTD steps/sim"
    )
    history = {"g": [], "d0": [], "erad": [], "gray": [], "beta": []}
    t_start = time.time()
    d0_init = None
    for i in range(n_iters):
        beta = schedule(i)
        theta, opt_state, loss, (rho, d0, e_rad), grads = step(
            theta, opt_state, jnp.asarray(beta)
        )
        if not bool(jnp.all(jnp.isfinite(grads))):
            raise FloatingPointError(f"non-finite gradient at iteration {i}")
        if d0_init is None:
            d0_init = float(d0)  # metrics of the uniform rho = 0.5 start
        history["g"].append(-float(loss))
        history["d0"].append(float(d0))
        history["erad"].append(float(e_rad))
        history["gray"].append(float(gray_indicator(rho)))
        history["beta"].append(beta)
        if i % 10 == 0 or i == n_iters - 1:
            print(
                f"iter {i:3d}  beta {beta:4.0f}  G_proxy(0) {-float(loss):.4f}  "
                f"D(0) {float(d0):.3f}  e_rad {float(e_rad):.4f}  "
                f"gray {float(gray_indicator(rho)):.3f}  "
                f"[{time.time() - t_start:5.0f} s]"
            )
    t_opt = time.time() - t_start

    # --- final evaluation (longer run, fine angular grid) ---------------------
    rho_final = transform(theta, BETAS[-1])
    n_eval = 6000  # extra ring-down so the resonance DFT is well converged
    angles_fine = jnp.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    res = simulate_design(rho_final, (F0,), n_steps=n_eval)
    e_far = ntff_2d(
        res.dft_ez, res.dft_hx, res.dft_hy, grid, NTFF_MARGIN, (F0,), angles_fine
    )
    d_fine = np.asarray(directivity_2d(e_far, angles_fine)[0])
    tt = (jnp.arange(n_eval) + 0.5) * grid.dt
    p_avail_eval = jnp.abs(half_step_dft(pulse(tt), grid.dt, F0)[0]) ** 2 / (8.0 * RS)
    e_rad_f = float(radiated_power_2d(e_far, angles_fine)[0] / p_avail_eval)
    ez_mag = np.abs(np.asarray(res.dft_ez[0]))

    d0_f = d_fine[0]  # phi = 0
    fb_db = 10.0 * np.log10(d_fine[0] / d_fine[180])  # front-to-back, phi = 180
    gray_f = float(gray_indicator(rho_final))
    print(
        f"\nD(phi=0) at {F0 / 1e9:.2f} GHz: {d0_init:.3f} (uniform rho=0.5) "
        f"-> {d0_f:.3f}  (isotropic radiator: D = 1)"
    )
    print(f"front-to-back ratio D(0)/D(180 deg): {fb_db:.1f} dB")
    print(f"radiation efficiency P_rad/P_avail: {e_rad_f:.4f}")
    print(f"realized-gain proxy G(0): {d0_f * e_rad_f:.4f}")
    print(f"gray indicator at beta={BETAS[-1]:.0f}: {gray_f:.4f}")
    print(f"optimization time: {t_opt:.0f} s ({t_opt / n_iters:.1f} s/iter)")

    # --- 4-panel figure -------------------------------------------------------
    fig = plt.figure(figsize=(12, 10))
    mm = 1e3 * DX

    ax = fig.add_subplot(2, 2, 1)
    ext = [DESIGN[0].start * mm, DESIGN[0].stop * mm, DESIGN[1].start * mm, DESIGN[1].stop * mm]
    im = ax.imshow(
        np.asarray(rho_final).T, origin="lower", cmap="gray_r", vmin=0, vmax=1, extent=ext
    )
    ax.plot(PORT_IJ[0] * mm, PORT_IJ[1] * mm, "r*", ms=12, label="50 $\\Omega$ port")
    ax.annotate(
        "beam", xy=(ext[1], PORT_IJ[1] * mm), xytext=(ext[1] - 18, PORT_IJ[1] * mm + 12),
        arrowprops=dict(arrowstyle="->", color="C3"), color="C3",
    )
    ax.legend(loc="lower right")
    ax.set_title(f"final density (gray indicator {gray_f:.3f})")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    fig.colorbar(im, ax=ax, label=r"$\rho$")

    ax = fig.add_subplot(2, 2, 2, projection="polar")
    d_db = 10.0 * np.log10(np.maximum(d_fine, 1e-6))
    floor = -20.0
    ax.plot(np.asarray(angles_fine), np.maximum(d_db, floor), "C0")
    ax.plot([0.0], [d_db[0]], "C3o", label=f"D(0) = {d0_f:.2f} ({d_db[0]:.1f} dB)")
    ax.set_rmin(floor)
    ax.set_rmax(max(d_db.max() + 1.0, 6.0))
    ax.set_title(f"far-field directivity at {F0 / 1e9:.2f} GHz [dB]  (F/B {fb_db:.1f} dB)")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.1, -0.1))

    ax = fig.add_subplot(2, 2, 3)
    it = np.arange(len(history["g"]))
    ax.plot(it, history["g"], "C0", label=r"$G_{proxy}(0) = D(0) P_{rad}/P_{avail}$")
    ax.plot(it, history["erad"], "C2", alpha=0.7, label=r"$P_{rad}/P_{avail}$")
    for b in range(1, len(BETAS)):
        ax.axvline(b * ITERS_PER_BETA, color="0.8", ls="--")
    ax.set_xlabel("iteration")
    ax.set_ylabel("objective")
    ax.legend(loc="center left")
    ax2 = ax.twinx()
    ax2.plot(it, history["d0"], "C3", alpha=0.6)
    ax2.set_ylabel(r"$D(\phi=0)$", color="C3")
    ax.set_title(r"convergence ($\beta$: " + ", ".join(f"{b:.0f}" for b in BETAS) + ")")

    ax = fig.add_subplot(2, 2, 4)
    im = ax.imshow(
        ez_mag.T,
        origin="lower",
        cmap="inferno",
        extent=[0, NX * mm, 0, NY * mm],
        norm=matplotlib.colors.LogNorm(vmin=ez_mag.max() * 1e-3, vmax=ez_mag.max()),
    )
    ax.add_patch(
        plt.Rectangle(
            (ext[0], ext[2]), ext[1] - ext[0], ext[3] - ext[2],
            fill=False, edgecolor="cyan", lw=1.0, ls="--",
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (NTFF_MARGIN * mm, NTFF_MARGIN * mm),
            (NX - 1 - 2 * NTFF_MARGIN) * mm, (NY - 1 - 2 * NTFF_MARGIN) * mm,
            fill=False, edgecolor="lime", lw=1.0, ls=":",
        )
    )
    ax.set_title(f"$|E_z|$ at {F0 / 1e9:.2f} GHz (log scale; NTFF contour dotted)")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    fig.colorbar(im, ax=ax, label="V/m (spectral)")

    fig.suptitle(
        "2D topology optimization: NTFF realized-gain maximization toward $+x$",
        y=0.995,
    )
    fig.tight_layout()
    out_path = Path(__file__).with_name("optimize_directivity.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
