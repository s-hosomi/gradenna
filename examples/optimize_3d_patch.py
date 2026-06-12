"""3D topology optimization of a 2.45 GHz PCB patch radiator (Phase 4 demo).

A planar copper layer on a JLCPCB-style FR-4 stackup is grown by gradient
descent. The setup mirrors the validated patch benchmark
(tests/test_patch_antenna.py): finite ground plane, lossy FR-4 substrate
spanned by 3 cell layers with the pin-layer permittivity compensation
(eps_r_model = eps_r * n_free / n_gap), and a vertical probe feed driven by
a lumped 50 ohm RVS port. The *design layer* is a 2D density map rho on the
patch plane, mapped to a conductive-sheet conductivity (note 00, design
decision 2)

    sigma_plane = sheet_conductivity(sigma_from_density(rho), t=35e-6, dz),

i.e. the log-sigma interpolation [1e-4, 1e5] S/m followed by the 1-cell
sheet smearing sigma_eff = sigma * t / dz of 1 oz copper foil. The ground
plane, probe pin and feed pad are fixed (non-design) metal.

Objective: the total radiated power at f0 = 2.45 GHz — full-grid running
DFT -> `ntff_3d` -> `radiated_power_3d` — normalized by the available
source power |Vs|^2 / (8 Rs). As in the 2D demo, the radiated-energy
objective (not S11) automatically penalizes gray absorbing material, so
beta continuation can binarize the design without an explicit penalty.
The optimization uses the three-field scheme of `gradenna.topopt`
(sigmoid -> conic filter -> tanh projection, the density stays 2D) with
optax.adam, and `simulate_3d(checkpoint_segments=K)` for sqrt-N adjoint
memory.

After the final binarization, re-verify the design with the real copper
sheet (sigma_eff = 5.8e7 * t / dz ~ 4e6 S/m) — the optimization cap of
1e5 S/m is the response-saturation point, not the physical foil value.

Presets
-------
``--preset cpu-demo`` (default): reduced grid (54 x 54 x 30 cells,
dx = 2.2 mm), 3200 steps/simulation, 2 beta stages x 8 iterations.
Runs in well under 30 minutes on a laptop CPU.

``--preset gpu-24gb``: benchmark-resolution grid (88 x 88 x 39 cells,
dx = 1.2 mm — the resolution validated against the Balanis patch design),
6400 steps, 4 beta stages x 50 iterations. At startup the script prints
the `gradenna.estimate` memory breakdown and asserts that the
checkpointed-adjoint peak fits in 24 GB of VRAM.

Running on a cloud GPU (RunPod RTX 4090, ~$0.34/h, or any CUDA box)
-------------------------------------------------------------------
    git clone <repo> && cd gradenna
    uv sync                                  # CPU jax + dev deps
    uv pip install -U "jax[cuda12]"          # swap in the CUDA wheel
    uv run python examples/optimize_3d_patch.py --preset gpu-24gb

JAX_ENABLE_X64 is set by the script itself (float64; the gray rho = 0.5
start produces fluxes that can underflow in float32, see the 2D demo).
On a 24 GB card also consider XLA_PYTHON_CLIENT_MEM_FRACTION=0.9.

Local run:  JAX_ENABLE_X64=1 uv run python examples/optimize_3d_patch.py
Writes optimize_3d_patch.png next to this script.
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")  # before jax import; see docstring

import argparse
import math
import time
from dataclasses import dataclass
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
    Grid3D,
    alpha_max_for_fmin,
    gaussian_pulse_for_band,
    half_step_dft,
    s11_power_wave,
    sheet_conductivity,
    sigma_from_density,
    simulate_3d,
    time_series_dft,
)
from gradenna.constants import EPS0
from gradenna.estimate import fdtd3d_memory_estimate, fits_gpu, gpu_fit_report
from gradenna.ntff import ntff_3d, radiated_power_3d
from gradenna.topopt import DesignTransform, beta_schedule, gray_indicator

# --- shared physical problem -------------------------------------------------

F0 = 2.45e9
F_MIN, F_MAX = 1.5e9, 3.5e9  # -20 dB band of the excitation
EPS_FR4 = 4.3
H_SUB = 1.6e-3
TAN_D = 0.02
T_COPPER = 35e-6  # 1 oz foil
N_GAP = 3  # substrate cell layers (incl. the ground's dragged pin layer)
SIG_FIXED = 1.0e7  # ground/pin/feed-pad thin-sheet PEC surrogate [S/m]
SIGMA_MIN, SIGMA_MAX = 1e-4, 1e5  # log-sigma interpolation endpoints [S/m]
RS = 50.0

N_THETA, N_PHI = 13, 16  # far-field quadrature for radiated_power_3d


@dataclass(frozen=True)
class Preset:
    """Grid / optimization sizing of one run mode."""

    name: str
    dxy: float  # in-plane cell [m]
    n_des: int  # design region edge [cells] (square, on the patch plane)
    m_gnd: int  # ground margin beyond the design region [cells]
    m_air: int  # air gap between ground edge and CPML [cells]
    n_pml: int
    n_air_above: int  # air cells between dragged-pin layer and CPML
    n_steps: int
    ckpt: int  # checkpoint_segments (must divide n_steps)
    betas: tuple
    iters_per_beta: int
    lr: float
    filter_radius: float  # conic filter radius [cells]
    eval_steps: int  # longer final-evaluation run


PRESETS = {
    "cpu-demo": Preset(
        name="cpu-demo",
        dxy=2.2e-3,
        n_des=22,  # 48.4 mm: covers the 37.6 x 29.1 mm Balanis patch
        m_gnd=3,
        m_air=5,
        n_pml=8,
        n_air_above=7,
        n_steps=3200,
        ckpt=64,
        betas=(8.0, 16.0),
        iters_per_beta=8,
        lr=0.15,
        filter_radius=2.0,
        eval_steps=4800,
    ),
    "gpu-24gb": Preset(
        name="gpu-24gb",
        dxy=1.2e-3,  # the resolution of the validated patch benchmark
        n_des=40,  # 48 mm
        m_gnd=6,
        m_air=8,
        n_pml=10,
        n_air_above=12,
        n_steps=6400,
        ckpt=80,
        betas=(8.0, 16.0, 32.0, 64.0),
        iters_per_beta=50,
        lr=0.1,
        filter_radius=2.5,
        eval_steps=9600,
    ),
}


class Setup:
    """Grid, fixed structures and monitors derived from a preset."""

    def __init__(self, p: Preset):
        self.p = p
        dz = H_SUB / N_GAP
        margin = p.m_gnd + p.m_air + p.n_pml
        nx = ny = p.n_des + 2 * margin
        kg = p.n_pml + 3  # ground sheet cell layer
        k_patch = kg + N_GAP  # patch/design sheet cell layer
        nz = k_patch + 1 + p.n_air_above + p.n_pml
        self.grid = Grid3D(nx=nx, ny=ny, nz=nz, dx=p.dxy, dy=p.dxy, dz=dz)
        self.kg, self.k_patch = kg, k_patch

        # Design region footprint and ground footprint.
        self.d0 = margin
        self.d1 = margin + p.n_des
        g0, g1 = self.d0 - p.m_gnd, self.d1 + p.m_gnd

        # Pin-layer compensation (test_patch_antenna.py module docstring).
        eps_r_model = EPS_FR4 * (N_GAP - 1) / N_GAP
        sig_sub = 2.0 * math.pi * F0 * EPS0 * eps_r_model * TAN_D
        self.eps_r_port = eps_r_model

        eps_r = np.ones(self.grid.shape)
        sigma = np.zeros(self.grid.shape)
        eps_r[g0:g1, g0:g1, kg + 1 : k_patch] = eps_r_model
        sigma[g0:g1, g0:g1, kg + 1 : k_patch] = sig_sub
        sigma[g0:g1, g0:g1, kg] = SIG_FIXED  # ground sheet
        # Probe feed: design-center in x, 1/4 from the lower design edge in
        # y (near the radiating edge of a centered patch); vertical pin up
        # to the last free Ez edge = the 1-cell RVS port gap, plus a fixed
        # feed pad cell on the patch plane connecting the port to rho.
        pi = (self.d0 + self.d1) // 2
        pj = self.d0 + max(1, p.n_des // 4)
        sigma[pi, pj, kg + 1 : k_patch - 1] = SIG_FIXED
        sigma[pi, pj, k_patch] = SIG_FIXED
        self.port_ijk = (pi, pj, k_patch - 1)
        self.feed_ij = (pi - self.d0, pj - self.d0)  # in design coordinates

        self.eps_r = jnp.asarray(eps_r)
        self.sigma_fixed = jnp.asarray(sigma)
        mask = np.ones((p.n_des, p.n_des), bool)
        mask[self.feed_ij] = False  # the feed pad is fixed metal, not design
        self.design_mask = jnp.asarray(mask)

        self.cpml = CPMLSpec(thickness=p.n_pml, alpha_max=alpha_max_for_fmin(F_MIN))
        self.pulse = gaussian_pulse_for_band(F_MIN, F_MAX)
        self.ntff_margin = p.n_pml + 2  # box: 2 cells outside the CPML face
        self.thetas = jnp.linspace(0.0, np.pi, N_THETA)
        self.phis = jnp.linspace(0.0, 2.0 * np.pi, N_PHI, endpoint=False)

    def sigma_total(self, rho):
        """Fixed metal + the design layer's conductive-sheet conductivity."""
        sig_sheet = sheet_conductivity(1.0, T_COPPER, self.grid.dz) * (
            sigma_from_density(rho, SIGMA_MIN, SIGMA_MAX)
        )
        sig_sheet = jnp.where(self.design_mask, sig_sheet, 0.0)
        design = jnp.zeros(self.grid.shape).at[
            self.d0 : self.d1, self.d0 : self.d1, self.k_patch
        ].set(sig_sheet)
        return jnp.maximum(self.sigma_fixed, design)

    def simulate(self, rho, dft_freqs, n_steps, ckpt):
        t = (jnp.arange(n_steps) + 0.5) * self.grid.dt
        return simulate_3d(
            self.grid,
            eps_r=self.eps_r,
            sigma=self.sigma_total(rho),
            port_ijk=self.port_ijk,
            port_voltage=self.pulse(t),
            port_resistance=RS,
            cpml=self.cpml,
            dft_freqs=dft_freqs,
            checkpoint_segments=ckpt,
        )

    def p_avail(self, freqs, n_steps):
        """Available source power spectrum |Vs|^2 / (8 Rs) (note 12)."""
        t = (jnp.arange(n_steps) + 0.5) * self.grid.dt
        return jnp.abs(half_step_dft(self.pulse(t), self.grid.dt, freqs)) ** 2 / (
            8.0 * RS
        )

    def radiated_power(self, dft, freqs):
        e_far = ntff_3d(dft, self.grid, self.ntff_margin, freqs, self.thetas, self.phis)
        return radiated_power_3d(e_far[..., 0], e_far[..., 1], self.thetas, self.phis)

    def radiated_fraction_f0(self, rho):
        """P_rad(f0) / P_avail(f0), the figure of merit."""
        res = self.simulate(rho, (F0,), self.p.n_steps, self.p.ckpt)
        p_rad = self.radiated_power(res.dft, (F0,))[0]
        return p_rad / self.p_avail((F0,), self.p.n_steps)[0]


def print_memory_estimate(setup: Setup, assert_24gb: bool) -> None:
    p = setup.p
    dtype = "float64" if jax.config.read("jax_enable_x64") else "float32"
    est = fdtd3d_memory_estimate(
        setup.grid,
        p.n_steps,
        checkpoint_segments=p.ckpt,
        n_dft_freqs=1,
        dtype=dtype,
        cpml_thickness=p.n_pml,
    )
    g = setup.grid
    print(
        f"memory estimate ({g.nx}x{g.ny}x{g.nz} cells, {p.n_steps} steps, "
        f"K={p.ckpt}, {dtype}):"
    )
    for key in (
        "fields_gb",
        "psi_full_gb",
        "psi_strip_gb",
        "dft_monitor_gb",
        "state_full_gb",
        "forward_gb",
        "adjoint_checkpoint_gb",
        "adjoint_checkpoint_strip_gb",
    ):
        print(f"  {key:28s} {est[key]:10.3f} GB")
    verdict = gpu_fit_report(est)
    print(
        "  fits (checkpointed adjoint): "
        + ", ".join(f"{int(v)} GB: {'OK' if ok else 'NO'}" for v, ok in verdict.items())
    )
    if assert_24gb:
        assert fits_gpu(est["adjoint_checkpoint_gb"], 24.0), (
            f"checkpointed adjoint needs {est['adjoint_checkpoint_gb']:.1f} GB, "
            "which does not fit a 24 GB GPU"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--preset", choices=sorted(PRESETS), default="cpu-demo")
    ap.add_argument("--out", default=None, help="output PNG path")
    args = ap.parse_args()
    p = PRESETS[args.preset]
    setup = Setup(p)
    g = setup.grid

    print(
        f"preset {p.name}: grid {g.nx}x{g.ny}x{g.nz} "
        f"(dx={p.dxy * 1e3:.2f} mm, dz={g.dz * 1e3:.3f} mm), design "
        f"{p.n_des}x{p.n_des} cells ({p.n_des * p.dxy * 1e3:.1f} mm), "
        f"{p.n_steps} steps/sim, port at {setup.port_ijk}"
    )
    print_memory_estimate(setup, assert_24gb=(p.name == "gpu-24gb"))

    # --- optimization loop (2D density on the patch plane) -------------------
    transform = DesignTransform(radius_cells=p.filter_radius)
    schedule = beta_schedule(p.betas, p.iters_per_beta)
    n_iters = p.iters_per_beta * len(p.betas)

    theta = jnp.zeros((p.n_des, p.n_des))  # sigmoid(0) = 0.5: gray start
    opt = optax.adam(p.lr)
    opt_state = opt.init(theta)

    def objective(theta, beta):
        rho = transform(theta, beta)
        return -setup.radiated_fraction_f0(rho), rho

    @jax.jit
    def step(theta, opt_state, beta):
        (loss, rho), grads = jax.value_and_grad(objective, has_aux=True)(theta, beta)
        updates, opt_state = opt.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        return theta, opt_state, loss, rho, grads

    history = {"p": [], "gray": [], "beta": []}
    t_start = time.time()
    for i in range(n_iters):
        beta = schedule(i)
        theta, opt_state, loss, rho, grads = step(theta, opt_state, jnp.asarray(beta))
        if not bool(jnp.all(jnp.isfinite(grads))):
            raise FloatingPointError(f"non-finite gradient at iteration {i}")
        history["p"].append(-float(loss))
        history["gray"].append(float(gray_indicator(rho)))
        history["beta"].append(beta)
        print(
            f"iter {i:3d}  beta {beta:4.0f}  P_rad/P_avail {-float(loss):.4f}  "
            f"gray {float(gray_indicator(rho)):.3f}  "
            f"[{time.time() - t_start:6.0f} s]"
        )
    t_opt = time.time() - t_start

    # --- final evaluation: longer run, full band ------------------------------
    rho_final = transform(theta, p.betas[-1])
    eval_freqs = tuple(np.linspace(1.8e9, 3.2e9, 21))
    res = setup.simulate(rho_final, eval_freqs + (F0,), p.eval_steps, None)

    p_rad = setup.radiated_power(res.dft, eval_freqs + (F0,))
    p_av = setup.p_avail(eval_freqs + (F0,), p.eval_steps)
    frac = np.asarray(p_rad / p_av)
    v_hat = time_series_dft(res.port_v, g.dt, eval_freqs, t0=0.5 * g.dt)
    i_hat = time_series_dft(res.port_i, g.dt, eval_freqs, t0=0.5 * g.dt)
    s11_db = 20.0 * np.log10(np.abs(np.asarray(s11_power_wave(v_hat, i_hat, RS))))

    p0, pf = history["p"][0], float(frac[-1])
    gray_f = float(gray_indicator(rho_final))
    f_ghz = np.asarray(eval_freqs) / 1e9
    k_min = int(np.argmin(s11_db))
    print(
        f"\nP_rad/P_avail at {F0 / 1e9:.2f} GHz: {p0:.4f} (gray rho = 0.5 start)"
        f" -> {pf:.4f}"
    )
    print(f"S11 minimum: {s11_db[k_min]:.2f} dB at {f_ghz[k_min]:.2f} GHz")
    print(f"gray indicator at beta={p.betas[-1]:.0f}: {gray_f:.4f}")
    print(f"optimization time: {t_opt:.0f} s ({t_opt / n_iters:.1f} s/iter)")

    # --- figure ---------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    mm = 1e3 * p.dxy
    ext = [setup.d0 * mm, setup.d1 * mm, setup.d0 * mm, setup.d1 * mm]

    ax = axes[0, 0]
    im = ax.imshow(
        np.asarray(rho_final).T, origin="lower", cmap="gray_r", vmin=0, vmax=1,
        extent=ext,
    )
    ax.plot(setup.port_ijk[0] * mm, setup.port_ijk[1] * mm, "r*", ms=12,
            label="probe feed")
    ax.legend(loc="lower right")
    ax.set_title(f"final copper density (gray indicator {gray_f:.3f})")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    fig.colorbar(im, ax=ax, label=r"$\rho$")

    ax = axes[0, 1]
    e_mag = np.sqrt(
        np.abs(np.asarray(res.dft.ex[-1, :, :-1, setup.k_patch])) ** 2
        + np.abs(np.asarray(res.dft.ey[-1, :-1, :, setup.k_patch])) ** 2
    )
    im = ax.imshow(
        e_mag.T, origin="lower", cmap="inferno",
        extent=[0, g.nx * mm, 0, g.ny * mm],
        norm=matplotlib.colors.LogNorm(vmin=e_mag.max() * 1e-3, vmax=e_mag.max()),
    )
    ax.add_patch(plt.Rectangle((ext[0], ext[2]), ext[1] - ext[0], ext[3] - ext[2],
                               fill=False, edgecolor="cyan", lw=1.0, ls="--"))
    ax.set_title(f"in-plane $|E|$ on the patch plane at {F0 / 1e9:.2f} GHz")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    fig.colorbar(im, ax=ax, label="V/m (spectral)")

    ax = axes[1, 0]
    it = np.arange(len(history["p"]))
    ax.plot(it, history["p"], "C0", label=r"$P_{rad}/P_{avail}$ at $f_0$")
    for b in range(1, len(p.betas)):
        ax.axvline(b * p.iters_per_beta, color="0.8", ls="--")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$P_{rad}/P_{avail}$")
    ax2 = ax.twinx()
    ax2.plot(it, history["gray"], "C3", alpha=0.6)
    ax2.set_ylabel("gray indicator", color="C3")
    ax.set_title(
        r"convergence ($\beta$: " + ", ".join(f"{b:.0f}" for b in p.betas) + ")"
    )
    ax.legend(loc="center right")

    ax = axes[1, 1]
    ax.plot(f_ghz, s11_db, "C0", label=r"$|S_{11}|$ [dB]")
    ax.axvline(F0 / 1e9, color="0.7", ls="--", label=r"$f_0$")
    ax.set_xlabel("f [GHz]")
    ax.set_ylabel(r"$|S_{11}|$ [dB]", color="C0")
    ax2 = ax.twinx()
    ax2.plot(f_ghz, frac[:-1], "C2")
    ax2.set_ylabel(r"$P_{rad}/P_{avail}$", color="C2")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], loc="lower right")
    ax.set_title("S11 and radiated fraction of the final design")

    fig.suptitle(
        f"3D patch-layer topology optimization at 2.45 GHz ({p.name})", y=0.995
    )
    fig.tight_layout()
    out_path = Path(args.out) if args.out else Path(__file__).with_name(
        "optimize_3d_patch.png"
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
