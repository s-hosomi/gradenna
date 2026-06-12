# gradenna

[![CI](https://github.com/s-hosomi/gradenna/actions/workflows/ci.yml/badge.svg)](https://github.com/s-hosomi/gradenna/actions/workflows/ci.yml)

**grad**ient + ant**enna** — differentiable FDTD antenna inverse design in JAX. Grow RF antennas by gradient descent.

[日本語版 README はこちら](README.ja.md)

gradenna is a fully differentiable electromagnetic (FDTD) solver and topology-optimization toolkit for RF / microwave antenna design. The entire simulation — Yee update, CPML absorbing boundaries, lumped 50 Ω ports, running-DFT S-parameters, near-to-far-field transform — is a single JAX computation graph, so `jax.grad` gives you the exact adjoint gradient of any objective (S11, radiated power, directivity, gain) with respect to **every pixel of the design at once**.

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

## Features

- **Differentiable 2D TM and full 3D FDTD cores** (`jax.lax.scan`, jit-able, float32/float64), CPML/CFS absorbing boundaries, multi-source, √N gradient checkpointing for memory-bounded adjoints
- **Lumped RVS ports and S-parameters**: semi-implicit resistive-voltage-source ports, exact-phase running DFT, power-wave S11, discrete gap-susceptance de-embedding
- **Differentiable near-to-far-field transform** (2D and 3D): radiated power, directivity and gain as optimization objectives
- **Topology optimization toolkit**: conic density filter, tanh projection with β continuation, log-conductivity metal interpolation, connectivity and minimum-feature-size checks
- **Fabrication pipeline**: density map → polygons → RS-274X Gerber with JLCPCB design-rule checks (`pip install gradenna[fab]`)
- **Measurement loop**: Touchstone I/O, simulated-vs-measured S11 comparison, NanoVNA capture script (`pip install gradenna[measure]`)

## Validation

Every physics component is tested against analytic solutions and textbook references in CI (94 tests):

| Benchmark | Result |
|---|---|
| Cylindrical wave of a line current vs `H0^(2)(kρ)` (Harrington) | < 2.5% profile error, 2nd-order grid convergence |
| CPML reflection vs enlarged-domain reference | −92 dB (spec −60 dB) |
| 2D line-current radiation resistance vs ωμ0/4 | 1.3–2.4% |
| Infinitesimal dipole radiation resistance vs 80π²(l/λ)² | 0.34% after gap de-embedding |
| Dipole directivity via NTFF vs D₀ = 1.5 | 0.14% |
| 2.45 GHz FR-4 patch resonance vs Balanis design equations | −2.5% |
| `jax.grad` vs finite differences (all parameter classes) | ≤ 1e-4 relative |
| Checkpointed vs plain adjoint | bit-identical outputs |

## Demos

| Script | What it shows |
|---|---|
| `examples/optimize_2d_antenna.py` | An antenna grows from uniform gray: radiated-energy maximization at 2.45 GHz, 4× over an empty-box baseline, fully binary final design |
| `examples/optimize_directivity.py` | Beam shaping through the far-field transform: D(0°) 0.31 → 4.47, front-to-back ratio 16.8 dB |
| `examples/optimize_multiband.py` | Worst-band (softmin) radiated power across 2.0 + 3.0 GHz simultaneously |
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
- [ ] openEMS cross-check reference data, PCB fabrication + NanoVNA measurement campaign
- [ ] GPU memory optimization (PML-strip ψ storage, design-region DFT adjoints), 3D topology optimization

## License

MIT
