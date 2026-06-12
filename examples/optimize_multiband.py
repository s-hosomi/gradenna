"""Phase 5 demo: multiband (2.0 + 3.0 GHz) radiated-energy maximization.

Same setup as examples/optimize_2d_antenna.py (2D TM, lumped 50 ohm RVS
port embedded in a conductivity design region, log-sigma interpolation,
three-field parameterization with beta continuation), but the objective
asks for radiation in *two* bands simultaneously. The per-band figure of
merit is the radiated fraction through a Poynting contour,

    x_b = P_rad(f_b) / P_avail(f_b),    f_b in {2.0, 3.0} GHz,

and the loss is the smooth worst-case (min over bands) relaxation of
research note 13 Sec. 6.2,

    loss = -softmin_T(x) = T * logsumexp(-x / T),

so the gradient automatically concentrates on whichever band is currently
worse (softmax(-x/T) weights) instead of letting a single resonance win,
which is what the plain average objective tends to do. T = 0.05 keeps the
relaxation within T*ln(2) ~ 0.035 of the true min at convergence.

Run:  JAX_ENABLE_X64=1 uv run python examples/optimize_multiband.py
(~8 min on a laptop CPU; float64 required, see the Phase 3 demo
docstring.) Writes optimize_multiband.png next to this script.
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
    poynting_flux_box_2d,
    sigma_from_density,
    simulate_tm,
)
from gradenna.topopt import DesignTransform, beta_schedule, gray_indicator

# --- problem definition -----------------------------------------------------

DX = 2e-3  # 2 mm cells
NX = NY = 140  # 280 mm box
F_BANDS = (2.0e9, 3.0e9)  # the two objective bands
F_MIN, F_MAX = 1.5e9, 3.5e9  # -20 dB band of the excitation (covers both)
RS = 50.0  # port resistance [ohm m]
N_STEPS = 3500  # pulse + ring-down of resonant designs (Phase 3 demo)
PORT_IJ = (70, 50)  # feed: lower-middle, embedded in the design region
DESIGN = (slice(44, 96), slice(40, 92))  # 52x52 cells = 104 x 104 mm
FLUX_BOX = (15, 124, 15, 124)  # Poynting contour outside CPML + design
SIGMA_MAX, SIGMA_MIN = 1e5, 1e-4  # log-sigma interpolation endpoints [S/m]
T_SOFTMIN = 0.05  # softmin temperature (note 13 Sec. 6.2)

# --- optimization hyperparameters --------------------------------------------

FILTER_RADIUS = 3.0  # conic filter radius [cells]
BETAS = (8.0, 16.0, 32.0, 64.0)
ITERS_PER_BETA = 25  # 100 iterations total (lightweight demo)
LEARNING_RATE = 0.15

grid = Grid2D(nx=NX, ny=NY, dx=DX, dy=DX)
cpml = CPMLSpec(thickness=10, alpha_max=alpha_max_for_fmin(F_MIN))
pulse = gaussian_pulse_for_band(F_MIN, F_MAX)
t = (jnp.arange(N_STEPS) + 0.5) * grid.dt
vs = pulse(t)
# Available source power (spectral) per band, |Vs_hat|^2 / (8 Rs).
p_avail = jnp.abs(half_step_dft(vs, grid.dt, F_BANDS)) ** 2 / (8.0 * RS)

n_des = DESIGN[0].stop - DESIGN[0].start
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


def radiated_fractions(rho):
    """(n_bands,) radiated fractions P_rad(f_b) / P_avail(f_b)."""
    res = simulate_design(rho, F_BANDS)
    p_rad = poynting_flux_box_2d(res.dft_ez, res.dft_hx, res.dft_hy, grid, FLUX_BOX)
    return p_rad / p_avail


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
        frac = radiated_fractions(rho)
        # Smooth worst case over bands: -softmin(x) = T logsumexp(-x/T).
        loss = T_SOFTMIN * jax.scipy.special.logsumexp(-frac / T_SOFTMIN)
        return loss, (rho, frac)

    @jax.jit
    def step(theta, opt_state, beta):
        (loss, aux), grads = jax.value_and_grad(objective, has_aux=True)(theta, beta)
        updates, opt_state = opt.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        return theta, opt_state, loss, aux, grads

    print(
        f"design region {n_des}x{n_des} cells, {n_iters} iterations, "
        f"betas {BETAS}, lr {LEARNING_RATE}, {N_STEPS} FDTD steps/sim, "
        f"bands {tuple(f / 1e9 for f in F_BANDS)} GHz, T {T_SOFTMIN}"
    )
    history = {"f1": [], "f2": [], "gray": [], "beta": []}
    frac_init = None
    t_start = time.time()
    for i in range(n_iters):
        beta = schedule(i)
        theta, opt_state, loss, (rho, frac), grads = step(
            theta, opt_state, jnp.asarray(beta)
        )
        if not bool(jnp.all(jnp.isfinite(grads))):
            raise FloatingPointError(f"non-finite gradient at iteration {i}")
        if frac_init is None:
            frac_init = np.asarray(frac)  # fractions of the uniform start
        history["f1"].append(float(frac[0]))
        history["f2"].append(float(frac[1]))
        history["gray"].append(float(gray_indicator(rho)))
        history["beta"].append(beta)
        if i % 10 == 0 or i == n_iters - 1:
            print(
                f"iter {i:3d}  beta {beta:4.0f}  "
                f"frac {float(frac[0]):.4f} / {float(frac[1]):.4f}  "
                f"gray {float(gray_indicator(rho)):.3f}  "
                f"[{time.time() - t_start:5.0f} s]"
            )
    t_opt = time.time() - t_start

    # --- final evaluation (longer run, full spectrum) -------------------------
    rho_final = transform(theta, BETAS[-1])
    n_eval = 6000  # extra ring-down so the resonance DFT is well converged
    eval_freqs = tuple(np.linspace(1.6e9, 3.4e9, 73))
    res = simulate_design(rho_final, eval_freqs + F_BANDS, n_steps=n_eval)
    tt = (jnp.arange(n_eval) + 0.5) * grid.dt
    vs_hat = half_step_dft(pulse(tt), grid.dt, eval_freqs + F_BANDS)
    p_avail_eval = jnp.abs(vs_hat) ** 2 / (8.0 * RS)
    p_rad = poynting_flux_box_2d(res.dft_ez, res.dft_hx, res.dft_hy, grid, FLUX_BOX)
    frac_eval = np.asarray(p_rad / p_avail_eval)
    spectrum, frac_final = frac_eval[: len(eval_freqs)], frac_eval[len(eval_freqs):]
    gray_f = float(gray_indicator(rho_final))

    print("\nradiated fraction P_rad/P_avail (uniform rho=0.5 -> final):")
    for fb, x0, x1 in zip(F_BANDS, frac_init, frac_final):
        print(f"  {fb / 1e9:.2f} GHz: {x0:.2e} -> {x1:.4f}  ({x1 / x0:.0f}x)")
    print(f"worst band: {frac_final.min():.4f}")
    print(f"gray indicator at beta={BETAS[-1]:.0f}: {gray_f:.4f}")
    print(f"optimization time: {t_opt:.0f} s ({t_opt / n_iters:.1f} s/iter)")

    # --- 3-panel figure -------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    mm = 1e3 * DX

    ax = axes[0]
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

    ax = axes[1]
    it = np.arange(len(history["f1"]))
    ax.plot(it, history["f1"], "C0", label=f"{F_BANDS[0] / 1e9:.1f} GHz")
    ax.plot(it, history["f2"], "C1", label=f"{F_BANDS[1] / 1e9:.1f} GHz")
    for b in range(1, len(BETAS)):
        ax.axvline(b * ITERS_PER_BETA, color="0.8", ls="--")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$P_{rad}/P_{avail}$")
    ax.legend(loc="upper left")
    ax.set_title(r"per-band convergence (softmin objective, $T$ = " + f"{T_SOFTMIN})")

    ax = axes[2]
    f_ghz = np.asarray(eval_freqs) / 1e9
    ax.plot(f_ghz, spectrum, "C2")
    for fb, x1 in zip(F_BANDS, frac_final):
        ax.axvline(fb / 1e9, color="0.7", ls="--")
        ax.plot([fb / 1e9], [x1], "C3o")
    ax.set_xlabel("f [GHz]")
    ax.set_ylabel(r"$P_{rad}/P_{avail}$")
    ax.set_title("radiation spectrum of the final design (bands dashed)")

    fig.suptitle(
        "2D multiband topology optimization: softmin radiated fraction at "
        f"{F_BANDS[0] / 1e9:.1f} + {F_BANDS[1] / 1e9:.1f} GHz",
        y=1.02,
    )
    fig.tight_layout()
    out_path = Path(__file__).with_name("optimize_multiband.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
