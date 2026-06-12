#!/usr/bin/env python
"""FDTD throughput benchmark (Mcell-steps/s) for the gradenna solvers.

Runs the standard 2D (256^2 x 1000 steps) and 3D (64^3 x 300 steps) cases
in float32 and float64 on the current default JAX backend and prints a
table of sustained throughput (timed on the second and later calls of the
jitted simulation, i.e. compilation excluded).

By default the per-platform environment preset from
:mod:`gradenna.platform` is applied before jax is imported; pass
``--env-off`` to benchmark the stock environment instead and compare.

Examples::

    .venv/bin/python scripts/benchmark.py            # full run, preset on
    .venv/bin/python scripts/benchmark.py --env-off  # stock environment
    .venv/bin/python scripts/benchmark.py --quick    # small grids, 1 repeat
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

# Make the repo's src/ importable when running from a source checkout.
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, _SRC)


def _load_platform_module():
    """Load gradenna/platform.py without importing the gradenna package.

    `import gradenna.platform` would run gradenna/__init__.py, which imports
    jax — and the whole point of the env preset is to be applied *before*
    jax is imported.
    """
    path = os.path.join(_SRC, "gradenna", "platform.py")
    spec = importlib.util.spec_from_file_location("_gradenna_platform", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--quick", action="store_true",
                   help="small grids and a single timed repeat (CI smoke test)")
    p.add_argument("--env-off", action="store_true",
                   help="do not apply gradenna.platform.recommended_env")
    p.add_argument("--platform", default="auto",
                   choices=("auto", "cpu", "apple-silicon", "cuda"),
                   help="preset passed to recommended_env (default: auto)")
    p.add_argument("--repeats", type=int, default=None,
                   help="timed repeats per case (default: 3, --quick: 1)")
    p.add_argument("--backend", default="xla", choices=("xla", "native"),
                   help="2D solver backend: xla (default) or native (Rust "
                        "fused kernel; skipped with a message if unavailable)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # Environment must be settled before the first `import jax`.
    platform_mod = _load_platform_module()
    applied = {} if args.env_off else platform_mod.apply_recommended_env(args.platform)
    os.environ.setdefault("JAX_ENABLE_X64", "1")  # needed for the f64 rows

    import jax
    import jax.numpy as jnp
    import numpy as np

    from gradenna.fdtd2d import simulate_tm
    from gradenna.fdtd3d import simulate_3d
    from gradenna.grid import Grid2D, Grid3D
    from gradenna.sources import gaussian_derivative

    repeats = args.repeats if args.repeats is not None else (1 if args.quick else 3)

    def bench(make_fn, src, n_cells, n_steps):
        """Best wall time of `repeats` calls after one compile/warmup call."""
        jfn = jax.jit(make_fn)
        jax.block_until_ready(jfn(src))  # compile + first run
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            jax.block_until_ready(jfn(src))
            best = min(best, time.perf_counter() - t0)
        return n_cells * n_steps / best / 1e6, best

    def bench_native(run_fn, n_cells, n_steps):
        """Best wall time of `repeats` native (ctypes) calls after a warmup."""
        run_fn()  # warmup (build + load on first ever call)
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            run_fn()
            best = min(best, time.perf_counter() - t0)
        return n_cells * n_steps / best / 1e6, best

    def waveform(grid, n_steps, dtype):
        t = (np.arange(n_steps) + 0.5) * grid.dt
        return jnp.asarray(gaussian_derivative(t, 20 * grid.dt, 6 * grid.dt), dtype)

    def case_2d(n, n_steps, dtype):
        g = Grid2D(n, n, 1e-3, 1e-3)
        src = waveform(g, n_steps, dtype)

        if args.backend == "native":
            import numpy as _np

            from gradenna import native
            if not native.is_available():
                return None, None
            src_np = _np.asarray(src)

            def run():
                native.simulate_tm_native(
                    g, source_ij=(n // 2, n // 2), source_current=src_np,
                    probe_ij=((n // 4, n // 4),), dtype=_np.dtype(dtype.__name__))

            return bench_native(run, n * n, n_steps)

        eps = jnp.ones((n, n), dtype)
        sig = jnp.zeros((n, n), dtype)

        def fn(s):
            r = simulate_tm(g, source_ij=(n // 2, n // 2), source_current=s,
                            eps_r=eps, sigma=sig, probe_ij=((n // 4, n // 4),))
            return r.probe_ez

        return bench(fn, src, n * n, n_steps)

    def case_3d(n, n_steps, dtype):
        g = Grid3D(n, n, n, 1e-3, 1e-3, 1e-3)
        src = waveform(g, n_steps, dtype)

        if args.backend == "native":
            import numpy as _np

            from gradenna import native3d
            if not native3d.is_available():
                return None, None
            src_np = _np.asarray(src)

            def run():
                native3d.simulate_3d_native(
                    g, source_ijk=(n // 2, n // 2, n // 2), source_current=src_np,
                    probe_ijk=((n // 4, n // 4, n // 4),),
                    dtype=_np.dtype(dtype.__name__))

            return bench_native(run, n * n * n, n_steps)

        eps = jnp.ones((n, n, n), dtype)
        sig = jnp.zeros((n, n, n), dtype)

        def fn(s):
            r = simulate_3d(g, source_ijk=(n // 2, n // 2, n // 2), source_current=s,
                            eps_r=eps, sigma=sig, probe_ijk=((n // 4, n // 4, n // 4),))
            return r.probe_ez

        return bench(fn, src, n * n * n, n_steps)

    if args.quick:
        cases = [("2D 128^2 x 200", case_2d, 128, 200), ("3D 32^3 x 100", case_3d, 32, 100)]
    else:
        cases = [("2D 256^2 x 1000", case_2d, 256, 1000), ("3D 64^3 x 300", case_3d, 64, 300)]

    print(f"backend: {jax.default_backend()}  devices: {jax.devices()}  solver: {args.backend}")
    print(f"jax {jax.__version__}, repeats={repeats}, "
          f"env preset: {'OFF' if args.env_off else applied or '(nothing to set)'}")
    if args.backend == "native":
        from gradenna import native
        if not native.is_available():
            print("native solver unavailable (cargo missing / build failed); "
                  "run scripts/build_kernel.sh. Skipping.")
            return 0
    header = f"{'case':<18} {'dtype':<8} {'Mcell-steps/s':>14} {'best time':>11}"
    print(header)
    print("-" * len(header))
    for name, runner, n, n_steps in cases:
        for dtype in (jnp.float32, jnp.float64):
            mcs, sec = runner(n, n_steps, dtype)
            if mcs is None:
                print(f"{name:<18} {jnp.dtype(dtype).name:<8} {'(unavailable)':>14}")
                continue
            print(f"{name:<18} {jnp.dtype(dtype).name:<8} {mcs:>14.1f} {sec * 1e3:>9.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
