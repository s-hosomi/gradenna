# gradenna-wasm — browser 2D FDTD kernel

An educational, single-threaded, `f32` **2D TM-mode FDTD** solver (Ez, Hx, Hy)
with **CPML** absorbing boundaries, compiled from Rust to WebAssembly for use by
the in-browser three.js viewer.

It mirrors the Yee discretization and CPML formulation of the gradenna reference
solver (`src/gradenna/fdtd2d.py`, `src/gradenna/cpml.py`) closely enough that the
wave looks physically correct — isotropic cylindrical propagation, no visible
reflection at the absorbing borders, and reflection from conductors — but it is
**not** the performance kernel and makes no attempt at bit-for-bit parity.

## Relationship to the main gradenna kernel (`rust/`)

This crate is **independent** of the high-performance, multi-threaded,
auto-differentiable kernel in `rust/` (which uses `rayon` and is loaded by Python
via FFI). This one is a deliberately simple single-threaded `f32` port whose only
job is to drive a live browser demo. The two share no code and serve different
purposes:

| | `rust/` (`gradenna_kernel`) | `web/wasm-kernel/` (`gradenna-wasm`) |
|---|---|---|
| Target | native, Python FFI | `wasm32-unknown-unknown` (browser) |
| Threads | multi-threaded (rayon) | single-threaded |
| Precision | f32 / f64 | f32 |
| Gradients | yes (frequency-domain adjoint) | no — forward simulation only |
| Goal | speed & differentiability | clear, lightweight visual demo |

## Physics

TM-mode Yee staggering, row-major flat index `i * ny + j`:

```
Ez(i, j)       integer grid, shape (nx, ny)   -- outer ring held at 0 (PEC)
Hx(i, j+1/2)   half grid in y, shape (nx, ny-1)
Hy(i+1/2, j)   half grid in x, shape (nx-1, ny)
```

Update (Schneider, *Understanding the FDTD Method*, Ch. 8/11):

```
Hx^{n+1/2} = Hx - (dt/mu) [ dEz/dy / kappa_y + psi_Hx,y ]
Hy^{n+1/2} = Hy + (dt/mu) [ dEz/dx / kappa_x + psi_Hy,x ]
Ez^{n+1}   = Ca Ez + Cb [ dHy/dx / kappa_x + psi_Ez,x
                          - dHx/dy / kappa_y - psi_Ez,y ]
```

with `Ca = (1 - sigma·dt/2eps)/(1 + sigma·dt/2eps)` and
`Cb = (dt/eps)/(1 + sigma·dt/2eps)`. The time step on a square cell is

```
dt = 0.99 · dx / (c · √2)
```

CPML coefficients follow Roden & Gedney (2000) with the Taflove reference
grading (`m = 3`, `kappa_max = 5`, `sigma_factor = 0.75`, `alpha_max = 0`),
matching `CPMLSpec` defaults in `src/gradenna/cpml.py`.

## Build

Prerequisites (one-time):

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-pack        # or: brew install wasm-pack
```

Build the browser package (outputs `pkg/gradenna_wasm.js` and
`pkg/gradenna_wasm_bg.wasm`):

```sh
wasm-pack build --target web --out-dir pkg --release
```

`pkg/` is committed so the viewer can load it without a Rust toolchain.

## JavaScript API

The exported `Fdtd2D` class (signature frozen):

```js
import init, { Fdtd2D } from "./pkg/gradenna_wasm.js";

await init();                              // load + instantiate the wasm module
const nx = 256, ny = 256;
const sim = new Fdtd2D(nx, ny, 1e-3, 12);  // (nx, ny, dx_metres, npml_cells)

// Optional: per-cell conductivity map (S/m), row-major idx = i*ny + j.
// const sigma = new Float32Array(nx * ny); ...; sim.set_sigma(sigma);
// sim.clear_sigma();                      // back to vacuum

const dt = sim.dt_seconds();               // time step (seconds)

// Map a Float32Array view straight onto the Ez field in wasm memory.
// Re-create the view after any wasm allocation that could grow memory.
function ezView() {
  return new Float32Array(memory.buffer, sim.ez_ptr(), sim.nx() * sim.ny());
}
// `memory` is the wasm instance memory; with `--target web` it is the
// `memory` export available after `init()` (see the generated glue).

let t = 0;
function frame() {
  // Soft Gaussian/Ricker pulse on a center Ez cell, for example:
  const src = /* your waveform */ 0.0;
  sim.step(nx >> 1, ny >> 1, src);         // advance one step, soft source on Ez
  // ... upload ezView() to a three.js DataTexture and render ...
  t += dt;
}

sim.reset();                               // zero the fields (keeps sigma)
```

### Methods

| Method | Description |
|---|---|
| `new Fdtd2D(nx, ny, dx_m, npml)` | Vacuum-initialized solver, square cells. |
| `set_sigma(Float32Array)` | Conductivity map (S/m), length `nx*ny`, row-major `i*ny + j`. |
| `clear_sigma()` | Reset material to vacuum. |
| `step(src_i, src_j, src_val)` | Advance one step; adds `src_val` onto `Ez[src_i, src_j]`. |
| `ez_ptr()` | Pointer into wasm memory to the row-major `nx*ny` Ez array. |
| `nx()`, `ny()` | Grid dimensions. |
| `dt_seconds()` | Time step in seconds. |
| `reset()` | Zero all fields and CPML state (keeps `sigma`). |

## Tests

Physics is validated with native (non-wasm) unit tests; the wasm-bindgen layer is
`cfg`-gated so the core compiles for the host target:

```sh
cargo test --release
```

Covers: long-run stability (1000 steps, bounded `|Ez|`), isotropy of the
cylindrical wavefront, CPML absorption (residual below −40 dB of the peak), and
reflection / attenuation by a high-conductivity region.
