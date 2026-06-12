# Contributing to gradenna

Thanks for your interest! Issues and pull requests are welcome.

## Development setup

```bash
git clone https://github.com/s-hosomi/gradenna && cd gradenna
uv sync                                   # or: pip install -e ".[fab,measure]"
```

Optional extras:

- **Native kernels**: a Rust toolchain (`cargo`) — the kernels build on first
  use and fall back cleanly without one.
- **Web visualizer**: Node 18+ (`cd web/app && npm install && npm run dev`),
  and `wasm-pack` only if you change `web/wasm-kernel/`.

## Tests

```bash
JAX_ENABLE_X64=1 uv run pytest tests/ -m "not slow" -q   # fast tier (~1-2 min)
JAX_ENABLE_X64=1 uv run pytest tests/ -q                 # + slow convergence tier
```

The fast tier must stay green. Native-kernel tests skip automatically when
`cargo` is unavailable; openEMS cross-checks compare against committed CSVs
and never require openEMS itself.

## What we look for

- **Physics changes need physics evidence.** Anything touching the solvers,
  ports, NTFF, adjoints or kernels must come with a test against an analytic
  solution, a textbook value, an independent solver, or gradient checks vs
  finite differences (see `tests/` for the house style and thresholds).
- **Gradients are the product.** If your change can affect differentiability
  or gradient values, add or extend a `jax.grad`-vs-finite-difference check.
- **Backward compatibility** for default arguments: new solver capabilities
  should be opt-in and leave existing code paths bit-identical when off.
- Docstrings and comments in English; README changes go to both `README.md`
  and `README.ja.md`.

## Repository layout

| Path | Contents |
|---|---|
| `src/gradenna/` | the JAX solvers, adjoints, optimization toolkit, fab/measure I/O |
| `rust/` | the multithreaded native CPU kernels (optional acceleration) |
| `web/` | the three.js visualizer and the wasm demo kernel |
| `tests/` | the verification suite (analytic, cross-solver, gradient) |
| `benchmarks/openems_refs/` | committed openEMS reference data + generator |
| `fab_campaign/` | ready-to-order Gerber package for the benchmark patch |
