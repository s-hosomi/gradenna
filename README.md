# gradenna

[![CI](https://github.com/s-hosomi/gradenna/actions/workflows/ci.yml/badge.svg)](https://github.com/s-hosomi/gradenna/actions/workflows/ci.yml)

**grad**ient + ant**enna** — differentiable FDTD antenna inverse design in JAX. Grow RF antennas by gradient descent.

[日本語版 README はこちら](README.ja.md)

<p align="center">
  <img src="assets/optimization.gif" width="460" alt="A 2.45 GHz antenna growing from uniform gray by gradient descent"/>
</p>
<p align="center">
  <em>An antenna grows out of uniform gray. 200 Adam iterations through a differentiable
  Maxwell solver on a 104×104-pixel design region (1 mm cells), maximizing radiated power
  at 2.45 GHz — the objective ends up <strong>58,000×</strong> above the starting design.
  No geometry was drawn by hand.</em>
</p>

gradenna is a fully differentiable electromagnetic (FDTD) solver and topology-optimization toolkit for RF / microwave antenna design. The entire simulation — Yee update, CPML absorbing boundaries, lumped 50 Ω ports, running-DFT S-parameters, near-to-far-field transform — is a single JAX computation graph, so `jax.grad` gives you the exact adjoint gradient of any objective (S11, radiated power, directivity, gain) with respect to **every pixel of the design at once**:

```python
import jax, jax.numpy as jnp
from gradenna import (Grid2D, CPMLSpec, Port, simulate_tm,
                      gaussian_pulse_for_band, sigma_from_density,
                      poynting_flux_box_2d)

grid = Grid2D(nx=140, ny=140, dx=2e-3, dy=2e-3)
pulse = gaussian_pulse_for_band(2.0e9, 3.0e9)
t = (jnp.arange(3500) + 0.5) * grid.dt

def neg_radiated_power(rho):                      # rho: 0 = air, 1 = copper
    sigma = jnp.zeros(grid.shape).at[44:96, 60:112].set(sigma_from_density(rho))
    res = simulate_tm(grid, sigma=sigma, dft_freqs=(2.45e9,),
                      ports=(Port(ij=(70, 55), resistance=50.0, voltage=pulse(t)),),
                      cpml=CPMLSpec(thickness=10))
    return -poynting_flux_box_2d(res.dft_ez, res.dft_hx, res.dft_hy, grid,
                                 box=(20, 120, 20, 120))[0]

grad = jax.grad(neg_radiated_power)(0.5 * jnp.ones((52, 52)))  # one backward pass
```

## Interactive visualizer

`web/` hosts a three.js viewer for everything above — including a **live 2D FDTD
solver running in your browser** through a 26 kB Rust→wasm kernel. Data is
pre-generated and committed: `cd web/app && npm install && npm run dev`.

<table>
  <tr>
    <td width="50%">
      <img src="assets/viewer_optimization.png" alt="Topology optimization replay"/>
      <p align="center"><sub><b>Optimization</b> — the growth animation above, replayed with
      GPU bicubic resampling, scrubbing and a log-scale convergence chart</sub></p>
    </td>
    <td width="50%">
      <img src="assets/viewer_live_fdtd.png" alt="Live FDTD in the browser"/>
      <p align="center"><sub><b>Live FDTD</b> — the wasm kernel driving the optimized design
      live; time-averaged intensity reveals its directional beam. Paint copper, move the
      source, double-slit and mirror scenes</sub></p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <img src="assets/viewer_antenna3d.png" alt="Antenna 3D: geometry, near field and far field"/>
      <p align="center"><sub><b>Antenna 3D</b> — the real patch geometry, the glowing |E| near
      field of the 3D run on draggable slice planes (fringing fields at the radiating edges),
      and the far-field lobe above — the whole story in one orbitable scene</sub></p>
    </td>
    <td width="50%">
      <img src="assets/viewer_s11.png" alt="S11 vs openEMS overlay"/>
      <p align="center"><sub><b>S11</b> — gradenna vs the independent openEMS solver on the
      same antenna: dips 0.83% apart, RMS difference 0.51 dB</sub></p>
    </td>
  </tr>
</table>

`scripts/export_viz.py` regenerates the data JSONs from the solvers; see
`web/app/README.md` for the wasm build and GitHub Pages deployment.

## Features

- **Differentiable 2D TM and full 3D FDTD cores** (`jax.lax.scan`, jit-able, float32/float64), CPML/CFS absorbing boundaries, multi-source, √N gradient checkpointing for memory-bounded adjoints
- **Lumped RVS ports and S-parameters**: semi-implicit resistive-voltage-source ports, exact-phase running DFT, power-wave S11, discrete gap-susceptance de-embedding
- **Differentiable near-to-far-field transform** (2D and 3D): radiated power, directivity and gain as optimization objectives
- **Topology optimization toolkit**: conic density filter, tanh projection with β continuation, log-conductivity metal interpolation, connectivity and minimum-feature-size checks
- **Fabrication pipeline**: density map → polygons → RS-274X Gerber with JLCPCB design-rule checks (`pip install gradenna[fab]`); a ready-to-order package for the benchmark patch lives in `fab_campaign/`
- **Measurement loop**: Touchstone I/O, simulated-vs-measured S11 comparison, NanoVNA capture script (`pip install gradenna[measure]`)
- **Differentiable thin-wire MoM backend** (`gradenna.mom`): piecewise-sinusoidal Galerkin EFIE for free-space PEC wires — a fast surrogate with `jax.grad` through geometry (length/radius); layered-media Green's functions for substrates are future work

## Fast on consumer hardware — GPU and Apple Silicon

Most published differentiable-RF-FDTD work assumes datacenter GPUs (~90 GB per
GPU for a mid-resolution 3D antenna). gradenna is engineered so the same
problems run on hardware you already own:

| | |
|---|---|
| 3D patch topology optimization, full resolution | peaks at **7.7 GB** (float64) / **~3.9 GB** (float32) — fits a 24 GB consumer GPU |
| CPML auxiliary (ψ) memory in 3D | **−74%** via PML-slab storage |
| Adjoint gradient memory | **O(design cells × frequencies)** residuals — no time tape (frequency-domain adjoint) |
| 2D fused Rust kernel, Apple M1 Pro | **5,040 Mcell-steps/s** at 1024² float32 — **8.4×** over XLA CPU |
| 3D fused Rust kernel, Apple M1 Pro | **796 Mcell-steps/s** at 96³ — 2.4×; DFT-heavy gradients another **2.35×** via region-limited DFT |

- **Memory-bounded 3D adjoints**: PML-slab ψ storage plus √N gradient checkpointing. `gradenna.fdtd3d_memory_estimate` predicts the budget before you launch; the `gpu-24gb` preset of `examples/optimize_3d_patch.py` prints and asserts it.
- **float32 end to end**: topology optimization runs in plain float32 (complex64 DFT) — the native precision of consumer GPUs. For extreme attenuation between source and monitor, `dft_dtype=jnp.complex128` promotes only the DFT accumulators.
- **Frequency-domain adjoint (2D and 3D)**: when the objective depends only on frequency-domain quantities (S11, flux, far field), `simulate_tm_freq` / `simulate_3d_freq` compute the gradient from **two forward simulations** — no time tape at all. Validated against the full-AD oracle to cosine ≥ 0.9999997 in both dimensions, including **NTFF directivity objectives in 3D** — differentiable gain optimization on real 3D antennas with O(design) memory. The magnetic-cotangent coupling constant is derived in closed form (−ε₀/μ₀, the Yee symplectic metric ratio).
- **Design-region-limited DFT monitors (3D)**: `simulate_3d(dft_regions=...)` accumulates the running DFT only on static per-component slabs, and `freq_adjoint_gradient_3d(objective_kind="port" | "ntff_box" | "field")` derives those slabs automatically — design-region E components for the gradient contraction plus exactly the cells the objective reads. The gradient matches the full-grid path (cos ≥ 1−1e−12) while the DFT carry shrinks from six full-grid components to the slabs.
- **Fused Rust kernels (optional, ARM-tuned)**: the 2D and 3D time loops compile to cache-friendly multithreaded native kernels (cargo build on first use, clean fallback without a Rust toolchain); thread counts and tiling are tuned on Apple-Silicon cores. The frequency-domain adjoint runs both its forward and adjoint passes on the kernel (`backend="native"`, gradient parity tested). `scripts/benchmark.py --backend native` reproduces the numbers.

## Validation

Every physics component is tested against analytic solutions, textbook references and an independent solver in CI (175+ tests):

| Benchmark | Result |
|---|---|
| Cylindrical wave of a line current vs `H0^(2)(kρ)` (Harrington) | < 2.5% profile error, 2nd-order grid convergence |
| CPML reflection vs enlarged-domain reference | −92 dB (spec −60 dB) |
| 2D line-current radiation resistance vs ωμ0/4 | 1.3–2.4% |
| Infinitesimal dipole radiation resistance vs 80π²(l/λ)² | 0.34% after gap de-embedding |
| Dipole directivity via NTFF vs D₀ = 1.5 | 0.14% |
| 2.45 GHz FR-4 patch resonance vs Balanis design equations | −2.5% |
| 2.45 GHz patch vs openEMS (committed reference data) | resonance 0.83%, \|S11\| RMS 0.51 dB, pattern corr ≥ 0.999 |
| Thin-wire MoM dipole resonance vs textbook (0.47–0.48 λ, ~72 Ω) | L_res = 0.476 λ, Re Zin = 71.7 Ω |
| Beam-steering main lobe vs array-factor theory | within ±5° at −30°/0°/+30° |
| `jax.grad` vs finite differences (all parameter classes) | ≤ 1e-4 relative |
| Checkpointed vs plain adjoint | bit-identical outputs |

## Demos

| Script | What it shows |
|---|---|
| `examples/optimize_2d_antenna.py` | The growth demo: radiated-energy maximization at 2.45 GHz, fully binary final design (the hero GIF is this problem at 1 mm resolution, via `scripts/export_viz.py`) |
| `examples/optimize_directivity.py` | Beam shaping through the far-field transform: D(0°) 0.31 → 4.47, front-to-back ratio 16.8 dB |
| `examples/optimize_multiband.py` | Worst-band (softmin) radiated power across 2.0 + 3.0 GHz simultaneously |
| `examples/optimize_beamsteering.py` | **Beam steering**: complex feed weights of a 4-element λ/2 array optimized through the far-field transform |
| `examples/optimize_3d_patch.py` | **3D topology optimization**: copper density on a real FR-4 patch stackup, checkpointed adjoint; `--preset cpu-demo` (39× radiated power in ~2.5 min) or `--preset gpu-24gb` |
| `examples/patch_to_gerber.py` | Balanis patch design → density map → DRC checks → Gerber |

## Quick start

```bash
git clone https://github.com/s-hosomi/gradenna && cd gradenna
uv sync                                  # or: pip install -e ".[fab,measure]"
uv run pytest -m "not slow" -q           # fast verification suite (~1-2 min on CPU)
uv run python examples/optimize_2d_antenna.py
```

Runs on CPU out of the box (all demos finish in minutes); JAX GPU/TPU backends work unchanged.

## Why differentiable FDTD?

A 50×50 design region has 2500 degrees of freedom. Gradient-free methods (GA, pixel flipping) need thousands of simulations per generation; the adjoint method — which reverse-mode AD performs automatically and exactly for a leapfrog Maxwell solver — gets the full gradient for the cost of about two simulations, regardless of the number of parameters. gradenna applies machinery proven in photonics inverse design to the RF band, where conductor loss, lumped feeds and fabrication constraints change the problem.

## Roadmap

- [x] Phase 1 — differentiable 2D TM FDTD core, CPML, analytic + gradient verification
- [x] Phase 2 — lumped ports, S11, running-DFT monitors
- [x] Phase 3 — 2D topology optimization (density method, β continuation)
- [x] Phase 4 — 3D core, patch benchmark, Gerber export, measurement tooling
- [x] Phase 5 — far-field directivity and multiband objectives
- [x] GPU memory optimization (PML-slab ψ storage, √N checkpointing, float32 objectives), 3D topology optimization
- [x] Frequency-domain adjoint (gradient = two forward runs, no time tape) and fused Rust CPU kernels, both in 2D and 3D
- [x] Design-region-limited DFT monitors (unlocks kernel speedup for DFT-heavy 3D gradients)
- [x] openEMS cross-check reference data (committed CSVs, compared in CI)
- [x] Phase 5 extensions — array beam-steering demo, differentiable thin-wire MoM backend (free space)
- [x] Web visualizer — three.js viewer + Rust→wasm in-browser FDTD kernel
- [ ] PCB fabrication + NanoVNA measurement campaign — ready-to-order Gerber/drill package and runbook in `fab_campaign/` (probe-fed benchmark patch, JLCPCB DRC clean); physical ordering + measurement pending

## License

MIT
