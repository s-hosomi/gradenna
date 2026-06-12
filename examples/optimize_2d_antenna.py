"""Phase 3 demo: a 2.45 GHz antenna grows out of gradient descent.

A lumped 50 ohm RVS port drives the lower-middle of an empty 280 x 280 mm
2D TM box. Above/around the feed sits a 104 x 104 mm design region whose
density rho is mapped to electric conductivity through the log-sigma
interpolation of docs/research/00-summary.md / 05-metal-representation.md
(`gradenna.materials.sigma_from_density`, sigma in [1e-4, 1e5] S/m)
and optimized with the three-field scheme of gradenna.topopt (conic filter,
tanh projection, beta continuation 8 -> 64) and optax.adam.

Objective (the absorption-cheat killer of note 00, design decision 2): the
*radiated* power through a Poynting contour OUTSIDE the design region at
f0 = 2.45 GHz, normalized by the available source power |Vs|^2 / (8 Rs).
S11 minimization would also reward dumping power into gray absorber; the
radiated-energy objective penalizes loss automatically, so the design is
pushed toward binary metal/air all by itself.

Physics notes specific to the 2D TM slice (found while tuning this demo):

- Metal in 2D TM only reflects (currents are z-directed): there is no
  "dipole arm" growth; what grows is a resonant enclosure that tunes out
  the feed reactance and transforms the line-source radiation resistance
  (~ omega mu0 / 4 ~ 4.8 kohm at 2.45 GHz, log-divergent inductive
  reactance) toward the 50 ohm source.
- Resonant features need >~ lambda/2 = 61 mm, hence the 104 mm design
  region (a 80 mm region only resonates above 2.9 GHz).
- float64 is used here for reference-quality output, but plain float32
  also works since the slab-psi / float64-phase-table refactors (verified
  in tests/test_float32_opt.py); for very strong attenuation between the
  source and the flux contour, pass dft_dtype=jnp.complex128 (needs x64).

Run:  JAX_ENABLE_X64=1 uv run python examples/optimize_2d_antenna.py
(~10 min on a laptop CPU.) Writes optimize_2d_antenna.png next to this
script and logs every iteration.
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")  # before jax import; see docstring

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
    port_dft,
    poynting_flux_box_2d,
    s11_power_wave,
    sigma_from_density,
    simulate_tm,
)
from gradenna.topopt import DesignTransform, beta_schedule, gray_indicator

# --- problem definition -----------------------------------------------------

DX = 2e-3  # 2 mm cells: lambda/61 at 2.45 GHz
NX = NY = 140  # 280 mm box
F0 = 2.45e9  # objective frequency
F_MIN, F_MAX = 1.5e9, 3.5e9  # -20 dB band of the excitation
RS = 50.0  # port resistance [ohm m] (= ohm for the 1 m slice)
N_STEPS = 3500  # ~610-step pulse + ring-down of resonant designs
PORT_IJ = (70, 50)  # feed: lower-middle, embedded in the design region
DESIGN = (slice(44, 96), slice(40, 92))  # 52x52 cells = 104 x 104 mm
FLUX_BOX = (15, 124, 15, 124)  # (il, ir, jb, jt) Poynting contour, 5 cells
#   outside the CPML interface and well outside the design region
SIGMA_MAX, SIGMA_MIN = 1e5, 1e-4  # log-sigma interpolation endpoints [S/m]

# --- optimization hyperparameters (note 10 Sec. 9) ---------------------------

FILTER_RADIUS = 3.0  # conic filter radius [cells]
BETAS = (8.0, 16.0, 32.0, 64.0)
ITERS_PER_BETA = 50
LEARNING_RATE = 0.15

grid = Grid2D(nx=NX, ny=NY, dx=DX, dy=DX)
cpml = CPMLSpec(thickness=10, alpha_max=alpha_max_for_fmin(F_MIN))
pulse = gaussian_pulse_for_band(F_MIN, F_MAX)
t = (jnp.arange(N_STEPS) + 0.5) * grid.dt
vs = pulse(t)
# Available source power (spectral) |Vs_hat|^2 / (8 Rs) (note 12 Sec. 3.1b).
p_avail_f0 = jnp.abs(half_step_dft(vs, grid.dt, F0)[0]) ** 2 / (8.0 * RS)

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


def radiated_fraction(rho):
    """P_rad(f0) / P_avail(f0): the figure of merit (1 = perfectly matched
    lossless radiator; the empty box scores ~0.0055)."""
    res = simulate_design(rho, (F0,))
    p_rad = poynting_flux_box_2d(res.dft_ez, res.dft_hx, res.dft_hy, grid, FLUX_BOX)[0]
    return p_rad / p_avail_f0


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
        return -radiated_fraction(rho), rho

    @jax.jit
    def step(theta, opt_state, beta):
        (loss, rho), grads = jax.value_and_grad(objective, has_aux=True)(theta, beta)
        updates, opt_state = opt.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        return theta, opt_state, loss, rho, grads

    print(
        f"design region {n_des}x{n_des} cells, {n_iters} iterations, "
        f"betas {BETAS}, lr {LEARNING_RATE}, {N_STEPS} FDTD steps/sim"
    )
    history = {"p": [], "gray": [], "beta": []}
    t_start = time.time()
    for i in range(n_iters):
        beta = schedule(i)
        theta, opt_state, loss, rho, grads = step(
            theta, opt_state, jnp.asarray(beta)
        )
        if not bool(jnp.all(jnp.isfinite(grads))):
            raise FloatingPointError(f"non-finite gradient at iteration {i}")
        history["p"].append(-float(loss))
        history["gray"].append(float(gray_indicator(rho)))
        history["beta"].append(beta)
        if i % 10 == 0 or i == n_iters - 1:
            print(
                f"iter {i:3d}  beta {beta:4.0f}  P_rad/P_avail {-float(loss):.4f}  "
                f"gray {float(gray_indicator(rho)):.3f}  "
                f"[{time.time() - t_start:5.0f} s]"
            )
    t_opt = time.time() - t_start

    # --- final evaluation (longer run, full band) ----------------------------
    rho_final = transform(theta, BETAS[-1])
    n_eval = 6000  # extra ring-down so the resonance DFT is well converged
    eval_freqs = tuple(np.linspace(1.8e9, 3.2e9, 57))
    res = simulate_design(rho_final, eval_freqs + (F0,), n_steps=n_eval)

    tt = (jnp.arange(n_eval) + 0.5) * grid.dt
    vs_hat = half_step_dft(pulse(tt), grid.dt, eval_freqs)
    p_avail = jnp.abs(vs_hat) ** 2 / (8.0 * RS)
    p_rad = poynting_flux_box_2d(res.dft_ez, res.dft_hx, res.dft_hy, grid, FLUX_BOX)
    frac = np.asarray(p_rad[:-1] / p_avail)
    v_hat, i_hat = port_dft(res.port_v[:, 0], res.port_i[:, 0], grid.dt, eval_freqs)
    s11_db = 20.0 * np.log10(np.abs(np.asarray(s11_power_wave(v_hat, i_hat, RS))))
    ez_mag = np.abs(np.asarray(res.dft_ez[-1]))  # |Ez(f0)| of the final design

    p0, pf = history["p"][0], float(radiated_fraction(rho_final))
    gray_f = float(gray_indicator(rho_final))
    k = int(np.argmin(s11_db))
    f_ghz = np.asarray(eval_freqs) / 1e9
    print(
        f"\nP_rad/P_avail at {F0 / 1e9:.2f} GHz: {p0:.2e} (uniform rho=0.5 "
        f"absorber) -> {pf:.4f}  ({pf / 0.0055:.1f}x the empty box's 0.0055)"
    )
    print(f"S11 minimum: {s11_db[k]:.2f} dB at {f_ghz[k]:.2f} GHz")
    print(f"gray indicator at beta={BETAS[-1]:.0f}: {gray_f:.4f}")
    print(f"optimization time: {t_opt:.0f} s ({t_opt / n_iters:.1f} s/iter)")

    # --- 4-panel figure -------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    mm = 1e3 * DX

    ax = axes[0, 0]
    ext = [DESIGN[0].start * mm, DESIGN[0].stop * mm, DESIGN[1].start * mm, DESIGN[1].stop * mm]
    im = ax.imshow(
        np.asarray(rho_final).T, origin="lower", cmap="gray_r", vmin=0, vmax=1, extent=ext
    )
    ax.plot(PORT_IJ[0] * mm, PORT_IJ[1] * mm, "r*", ms=12, label="50 $\\Omega$ port")
    ax.legend(loc="lower right")
    ax.set_title(f"final density (gray indicator {gray_f:.3f})")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    fig.colorbar(im, ax=ax, label=r"$\rho$")

    ax = axes[0, 1]
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
    fb = FLUX_BOX
    ax.add_patch(
        plt.Rectangle(
            ((fb[0] + 0.5) * mm, (fb[2] + 0.5) * mm),
            (fb[1] - fb[0]) * mm, (fb[3] - fb[2]) * mm,
            fill=False, edgecolor="lime", lw=1.0, ls=":",
        )
    )
    ax.set_title(f"$|E_z|$ at {F0 / 1e9:.2f} GHz (log scale)")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    fig.colorbar(im, ax=ax, label="V/m (spectral)")

    ax = axes[1, 0]
    it = np.arange(len(history["p"]))
    ax.plot(it, history["p"], "C0", label=r"$P_{rad}/P_{avail}$ at $f_0$")
    for b in range(1, len(BETAS)):
        ax.axvline(b * ITERS_PER_BETA, color="0.8", ls="--")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$P_{rad}/P_{avail}$")
    ax2 = ax.twinx()
    ax2.plot(it, history["gray"], "C3", alpha=0.6, label="gray indicator")
    ax2.set_ylabel("gray indicator", color="C3")
    ax.set_title(r"convergence ($\beta$ stages: " + ", ".join(f"{b:.0f}" for b in BETAS) + ")")
    ax.legend(loc="center right")

    ax = axes[1, 1]
    ax.plot(f_ghz, s11_db, "C0", label=r"$|S_{11}|$ [dB]")
    ax.axvline(F0 / 1e9, color="0.7", ls="--", label=r"$f_0$")
    ax.set_xlabel("f [GHz]")
    ax.set_ylabel(r"$|S_{11}|$ [dB]", color="C0")
    ax2 = ax.twinx()
    ax2.plot(f_ghz, frac, "C2", label=r"$P_{rad}/P_{avail}$")
    ax2.set_ylabel(r"$P_{rad}/P_{avail}$", color="C2")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], loc="lower right")
    ax.set_title("S11 and radiated fraction of the final design")

    fig.suptitle(
        "2D topology optimization: radiated-energy maximization at 2.45 GHz", y=0.995
    )
    fig.tight_layout()
    out_path = Path(__file__).with_name("optimize_2d_antenna.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
