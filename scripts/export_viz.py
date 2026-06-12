#!/usr/bin/env python
"""Export visualization data for the three.js viewer.

Produces three JSON files in *outdir* (default: web/app/public/data):

    optimization.json   2D topology-optimization run at 2.45 GHz (radiated
                        power maximization), ~40-60 iterations, density maps
                        and objective per frame.
    farfield3d.json     3D far-field directivity grid on 25 x 49 (theta, phi)
                        points from the benchmark patch-antenna simulation.
    s11.json            S11 dB vs frequency of the same patch, with the
                        committed openEMS reference interpolated onto the same
                        frequency grid (when available).

Run:
    JAX_ENABLE_X64=1 .venv/bin/python scripts/export_viz.py [--outdir DIR]

Self-checks (assertions) are run after writing each file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: make the repo's src/ and the benchmark geometry importable.
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
_SRC = _REPO_ROOT / "src"
_REFS_DIR = _REPO_ROOT / "benchmarks" / "openems_refs"

for _p in (_SRC, _REFS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

os.environ.setdefault("JAX_ENABLE_X64", "1")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _r4(x):
    """Round a float to 4 significant figures for compact JSON."""
    if x == 0.0:
        return 0.0
    mag = math.floor(math.log10(abs(x)))
    factor = 10 ** (3 - mag)
    return round(x * factor) / factor


def _r4_list(arr):
    """Round a 1-D iterable to 4-sig-fig floats."""
    return [_r4(float(v)) for v in arr]


def _write_json(path: Path, obj: dict) -> int:
    """Write *obj* as compact JSON to *path* and return file size in bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")
    return len(text.encode())


# ---------------------------------------------------------------------------
# 1. optimization.json
# ---------------------------------------------------------------------------


def run_optimization(outdir: Path) -> dict:
    """Run a short 2D topology-optimization and return the JSON object."""
    import jax
    import jax.numpy as jnp
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
    from gradenna.topopt import DesignTransform, beta_schedule

    # --- problem definition (mirrors examples/optimize_2d_antenna.py) -------
    # Reduced grid (70x70 cells = 140 mm) and fewer iterations (~48 total)
    # to complete in a few minutes on CPU while still showing convergence.
    # The exact problem of examples/optimize_2d_antenna.py (the validated
    # "antenna grows from uniform gray" demo), so the recorded animation is
    # the canonical one.
    DX = 2e-3           # 2 mm cells
    NX = NY = 140       # 280 mm box
    F0 = 2.45e9
    F_MIN, F_MAX = 1.5e9, 3.5e9
    RS = 50.0
    N_STEPS = 3500      # pulse + ring-down of resonant designs
    PORT_IJ = (70, 50)  # feed: lower-middle, embedded in the design region

    # Design region: 52x52 cells = 104 mm
    DESIGN = (slice(44, 96), slice(40, 92))
    n_des = DESIGN[0].stop - DESIGN[0].start  # 52

    # Poynting flux box: a few cells inside the CPML interface
    FLUX_BOX = (20, 120, 20, 120)

    SIGMA_MAX, SIGMA_MIN = 1e5, 1e-4
    FILTER_RADIUS = 3.0
    BETAS = (8.0, 16.0, 32.0, 64.0)
    ITERS_PER_BETA = 50   # 200 total (canonical schedule)
    LEARNING_RATE = 0.15
    FRAME_STRIDE = 2      # store every other frame to keep the JSON small

    grid = Grid2D(nx=NX, ny=NY, dx=DX, dy=DX)
    cpml = CPMLSpec(thickness=8, alpha_max=alpha_max_for_fmin(F_MIN))
    pulse = gaussian_pulse_for_band(F_MIN, F_MAX)
    t0 = (jnp.arange(N_STEPS) + 0.5) * grid.dt
    vs = pulse(t0)
    p_avail_f0 = jnp.abs(half_step_dft(vs, grid.dt, F0)[0]) ** 2 / (8.0 * RS)

    # Port mask: feed cell itself gets no design conductivity
    _mask = np.ones((n_des, n_des), bool)
    _mask[PORT_IJ[0] - DESIGN[0].start, PORT_IJ[1] - DESIGN[1].start] = False
    design_mask = jnp.asarray(_mask)

    def simulate_design(rho, dft_freqs, n_steps=N_STEPS):
        """Run FDTD for design density *rho* and return the result."""
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
        """P_rad(f0) / P_avail(f0): the figure of merit."""
        res = simulate_design(rho, (F0,))
        p_rad = poynting_flux_box_2d(res.dft_ez, res.dft_hx, res.dft_hy, grid, FLUX_BOX)[0]
        return p_rad / p_avail_f0

    transform = DesignTransform(radius_cells=FILTER_RADIUS)
    sched = beta_schedule(BETAS, ITERS_PER_BETA)
    n_iters = ITERS_PER_BETA * len(BETAS)

    theta = jnp.zeros((n_des, n_des))
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
        return theta, opt_state, loss, rho

    print(
        f"[opt] {n_des}x{n_des} design, {n_iters} iters, "
        f"betas={BETAS}, lr={LEARNING_RATE}, {N_STEPS} FDTD steps"
    )
    objective_vals: list[float] = []
    density_frames: list[list[float]] = []
    t_start = time.perf_counter()

    for i in range(n_iters):
        beta = sched(i)
        theta, opt_state, loss, rho = step(theta, opt_state, jnp.asarray(beta))
        p = float(-loss)
        # Store every FRAME_STRIDE-th frame (plus the last) to bound the JSON.
        if i % FRAME_STRIDE == 0 or i == n_iters - 1:
            objective_vals.append(_r4(p))
            rho_np = np.asarray(rho)
            density_frames.append(_r4_list(rho_np.ravel()))
        if i % 10 == 0 or i == n_iters - 1:
            print(
                f"[opt] iter {i:3d}  beta {beta:4.0f}  P_rad/P_avail={p:.4f}  "
                f"[{time.perf_counter() - t_start:5.1f} s]"
            )

    rho_final = transform(theta, BETAS[-1])
    print(
        f"[opt] done: P_rad/P_avail  {objective_vals[0]:.4f} -> {objective_vals[-1]:.4f}  "
        f"({time.perf_counter() - t_start:.1f} s)"
    )

    # Extent in mm: design region physical size
    w_mm = n_des * DX * 1e3
    h_mm = n_des * DX * 1e3

    return {
        "kind": "optimization",
        "nx": n_des,
        "ny": n_des,
        "extent_mm": [_r4(w_mm), _r4(h_mm)],
        "objective_label": "radiated power (arb.)",
        "objective": objective_vals,
        "frames": density_frames,
    }


# ---------------------------------------------------------------------------
# 2. farfield3d.json + 3. s11.json  (shared 3D patch run)
# ---------------------------------------------------------------------------


def run_patch_3d():
    """Run the benchmark patch and return (freqs, zin, res, grid, ntff_margin)."""
    # Re-uses the exact same function from tests/test_openems_refs.py.
    # Import it dynamically so we never modify the tests/ directory.
    import importlib.util
    import math

    import jax.numpy as jnp
    import numpy as np

    from gradenna import CPMLSpec, alpha_max_for_fmin
    from gradenna.constants import EPS0
    from gradenna.designs import patch_design
    from gradenna.fdtd3d import Grid3D, port_impedance, simulate_3d
    from gradenna.sparams import gaussian_pulse_for_band

    import geometry as geo  # noqa: E402 (benchmarks/openems_refs on sys.path)

    mesh = geo.GRADENNA_MESH
    f0 = geo.F0
    dxy, dz = mesh.dxy, mesh.dz
    n_gap = mesh.n_gap

    w, length, _ = patch_design(f0, geo.EPS_FR4, geo.H_SUB)
    npx = round(w / dxy)
    npy = round(length / dxy)

    margin = mesh.m_gnd + mesh.m_air + mesh.n_pml
    nx = npx + 2 * margin
    ny = npy + 2 * margin
    kg = mesh.n_pml + 3
    k_patch = kg + n_gap
    nz = k_patch + 1 + 9 + mesh.n_pml
    grid = Grid3D(nx=nx, ny=ny, nz=nz, dx=dxy, dy=dxy, dz=dz)

    eps_r_model = geo.EPS_FR4 * (n_gap - 1) / n_gap
    sig_sub = 2.0 * math.pi * f0 * EPS0 * eps_r_model * geo.TAN_D

    i0 = (nx - npx) // 2
    j0 = (ny - npy) // 2
    gx0, gx1 = i0 - mesh.m_gnd, i0 + npx + mesh.m_gnd
    gy0, gy1 = j0 - mesh.m_gnd, j0 + npy + mesh.m_gnd

    eps_r = np.ones(grid.shape)
    sigma = np.zeros(grid.shape)
    eps_r[gx0:gx1, gy0:gy1, kg + 1 : k_patch] = eps_r_model
    sigma[gx0:gx1, gy0:gy1, kg + 1 : k_patch] = sig_sub
    sigma[gx0:gx1, gy0:gy1, kg] = mesh.sig_metal
    sigma[i0 : i0 + npx, j0 : j0 + npy, k_patch] = mesh.sig_metal
    pi, pj = i0 + npx // 2, j0 + 1
    sigma[pi, pj, kg + 1 : k_patch - 1] = mesh.sig_metal
    port_ijk = (pi, pj, k_patch - 1)

    pulse = gaussian_pulse_for_band(*geo.PULSE_BAND)
    t = (jnp.arange(mesh.n_steps) + 0.5) * grid.dt
    print(f"[3d] grid {nx}x{ny}x{nz}, {mesh.n_steps} steps …")
    t_start = time.perf_counter()
    res = simulate_3d(
        grid,
        eps_r=jnp.asarray(eps_r),
        sigma=jnp.asarray(sigma),
        port_ijk=port_ijk,
        port_voltage=pulse(t),
        port_resistance=geo.R_PORT,
        cpml=CPMLSpec(thickness=mesh.n_pml, alpha_max=alpha_max_for_fmin(1.0e9)),
        dft_freqs=(f0,),
    )
    print(f"[3d] FDTD done ({time.perf_counter() - t_start:.1f} s)")

    freqs = np.asarray(jnp.linspace(geo.SWEEP_BAND[0], geo.SWEEP_BAND[1], geo.N_SWEEP))
    zin = np.asarray(port_impedance(res, grid, freqs, eps_r_port=eps_r_model))
    return freqs, zin, res, grid, mesh.n_pml + 1


def build_farfield3d(res, grid, ntff_margin) -> dict:
    """Compute 25 x 49 directivity grid and return JSON object."""
    import numpy as np

    from gradenna.ntff import directivity_3d, ntff_3d

    import geometry as geo

    f0 = geo.F0
    n_theta, n_phi = 25, 49
    thetas = np.linspace(0.0, math.pi, n_theta)
    phis = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)

    print(f"[ff3d] computing NTFF {n_theta}x{n_phi} grid …")
    t_start = time.perf_counter()
    e_far = ntff_3d(res.dft, grid, ntff_margin, (f0,), thetas, phis)
    # e_far: (1, n_theta, n_phi, 2)
    e_theta = e_far[0, :, :, 0]
    e_phi = e_far[0, :, :, 1]
    d = directivity_3d(e_theta[None, :, :], e_phi[None, :, :], thetas, phis)
    d_np = np.asarray(d[0])  # (n_theta, n_phi)
    print(f"[ff3d] done ({time.perf_counter() - t_start:.1f} s), peak D={d_np.max():.3f}")

    # Find peak
    peak_idx = np.unravel_index(np.argmax(d_np), d_np.shape)
    peak_theta = float(thetas[peak_idx[0]])
    peak_phi = float(phis[peak_idx[1]])
    peak_d = float(d_np[peak_idx])

    # Directivity as list-of-lists [n_theta][n_phi], 4 sig figs
    directivity_2d = [[_r4(d_np[it, ip]) for ip in range(n_phi)] for it in range(n_theta)]

    return {
        "kind": "farfield3d",
        "freq_hz": _r4(f0),
        "thetas_rad": _r4_list(thetas),
        "phis_rad": _r4_list(phis),
        "directivity": directivity_2d,
        "peak": {"theta": _r4(peak_theta), "phi": _r4(peak_phi), "d": _r4(peak_d)},
    }


def build_s11(freqs, zin, outdir: Path) -> dict:
    """Build S11 JSON object from Zin sweep, with optional openEMS reference."""
    import numpy as np

    import geometry as geo

    z0 = geo.R_PORT
    s11 = (zin - z0) / (zin + z0)
    s11_db = 20.0 * np.log10(np.maximum(np.abs(s11), 1e-30))

    freq_list = _r4_list(freqs)
    s11_db_list = _r4_list(s11_db)

    obj: dict = {
        "kind": "s11",
        "freq_hz": freq_list,
        "s11_db_gradenna": s11_db_list,
        "label": "2.45 GHz FR-4 patch",
    }

    # Optionally append interpolated openEMS reference
    s11_csv = _REFS_DIR / "s11.csv"
    if s11_csv.is_file():
        print("[s11] interpolating openEMS reference onto gradenna frequency grid …")
        # Skip comment lines; header is the last comment-block line
        n_skip = 0
        with open(s11_csv) as fh:
            for line in fh:
                if line.startswith("#"):
                    n_skip += 1
                else:
                    break
        import numpy as _np
        ref = _np.genfromtxt(
            s11_csv, delimiter=",", names=True, comments=None, skip_header=n_skip
        )
        ref_db_interp = _np.interp(freqs, ref["freq_Hz"], ref["S11_dB"])
        obj["s11_db_openems"] = _r4_list(ref_db_interp)

    dip_idx = int(np.argmin(s11_db))
    dip_freq = float(freqs[dip_idx])
    print(f"[s11] gradenna dip: {s11_db[dip_idx]:.2f} dB at {dip_freq/1e9:.3f} GHz")
    return obj


# ---------------------------------------------------------------------------
# Self-checks
# ---------------------------------------------------------------------------


def selfcheck_optimization(path: Path):
    """Assert optimization JSON schema, finite values, and near-monotone objective."""
    obj = json.loads(path.read_text())
    assert obj["kind"] == "optimization"
    n, m = obj["nx"], obj["ny"]
    assert n > 0 and m > 0
    objective = obj["objective"]
    frames = obj["frames"]
    assert len(objective) == len(frames), "objective and frames length mismatch"
    assert all(math.isfinite(v) for v in objective), "NaN/Inf in objective"
    for i, frame in enumerate(frames):
        assert len(frame) == n * m, f"frame {i}: wrong length"
        assert all(math.isfinite(v) for v in frame), f"NaN/Inf in frame {i}"
        assert all(0.0 <= v <= 1.0 for v in frame), f"density out of [0,1] in frame {i}"
    # Objective should be overall increasing (last half > first half on average)
    mid = len(objective) // 2
    assert sum(objective[mid:]) >= sum(objective[:mid]) * 0.9, (
        "objective not broadly increasing: check optimization convergence"
    )
    print(f"[check] optimization.json OK  ({len(frames)} frames, {n}x{m} density)")


def selfcheck_farfield3d(path: Path):
    """Assert farfield3d JSON schema, finite values, and broadside peak."""
    obj = json.loads(path.read_text())
    assert obj["kind"] == "farfield3d"
    d_flat = [v for row in obj["directivity"] for v in row]
    assert all(math.isfinite(v) and v >= 0 for v in d_flat), "NaN/Inf/negative in directivity"
    peak_theta = obj["peak"]["theta"]
    # Broadside for a patch above ground plane: theta near 0 (upper hemisphere)
    assert peak_theta < math.pi / 2, (
        f"peak theta={math.degrees(peak_theta):.1f} deg is not in upper hemisphere"
    )
    print(
        f"[check] farfield3d.json OK  peak theta={math.degrees(peak_theta):.1f} deg, "
        f"phi={math.degrees(obj['peak']['phi']):.1f} deg, D={obj['peak']['d']:.2f}"
    )


def selfcheck_s11(path: Path):
    """Assert s11 JSON schema, finite values, and dip near 2.39 GHz."""
    import numpy as np

    obj = json.loads(path.read_text())
    assert obj["kind"] == "s11"
    freqs = np.array(obj["freq_hz"])
    s11_db = np.array(obj["s11_db_gradenna"])
    assert all(math.isfinite(v) for v in s11_db), "NaN/Inf in s11_db_gradenna"
    dip_idx = int(np.argmin(s11_db))
    dip_freq = float(freqs[dip_idx])
    # Accept dip anywhere in 2.2-2.7 GHz (the FDTD grid is coarse)
    assert 2.2e9 <= dip_freq <= 2.7e9, (
        f"S11 dip at {dip_freq/1e9:.3f} GHz is outside [2.2, 2.7] GHz"
    )
    print(f"[check] s11.json OK  dip {s11_db[dip_idx]:.2f} dB at {dip_freq/1e9:.3f} GHz")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--outdir",
        default=str(_REPO_ROOT / "web" / "app" / "public" / "data"),
        help="output directory for the JSON files (default: web/app/public/data)",
    )
    p.add_argument(
        "--skip-opt", action="store_true",
        help="skip the 2D optimization (use for quick re-runs of farfield/s11)",
    )
    p.add_argument(
        "--skip-3d", action="store_true",
        help="skip the 3D patch run (use for quick re-runs of optimization only)",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {outdir}")

    total_bytes = 0

    # --- 1. optimization ---------------------------------------------------
    if not args.skip_opt:
        print("\n=== 1/3  2D topology optimization ===")
        opt_obj = run_optimization(outdir)
        opt_path = outdir / "optimization.json"
        nb = _write_json(opt_path, opt_obj)
        total_bytes += nb
        print(f"[write] {opt_path}  ({nb/1024:.1f} kB)")
        selfcheck_optimization(opt_path)
    else:
        print("[skip] optimization.json")

    # --- 2+3. 3D patch run (farfield + s11 share one run) ------------------
    if not args.skip_3d:
        print("\n=== 2+3/3  3D patch FDTD (farfield + S11) ===")
        freqs, zin, res, grid, ntff_margin = run_patch_3d()

        ff_obj = build_farfield3d(res, grid, ntff_margin)
        ff_path = outdir / "farfield3d.json"
        nb = _write_json(ff_path, ff_obj)
        total_bytes += nb
        print(f"[write] {ff_path}  ({nb/1024:.1f} kB)")
        selfcheck_farfield3d(ff_path)

        s11_obj = build_s11(freqs, zin, outdir)
        s11_path = outdir / "s11.json"
        nb = _write_json(s11_path, s11_obj)
        total_bytes += nb
        print(f"[write] {s11_path}  ({nb/1024:.1f} kB)")
        selfcheck_s11(s11_path)
    else:
        print("[skip] farfield3d.json + s11.json")

    print(f"\nAll done. Total written: {total_bytes/1024:.1f} kB")
    assert total_bytes < 10 * 1024 * 1024, f"total size {total_bytes} B exceeds 10 MB limit"
    return 0


if __name__ == "__main__":
    sys.exit(main())
