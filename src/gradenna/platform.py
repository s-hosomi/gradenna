"""Per-platform JAX environment presets (Apple Silicon CPU, CUDA).

Why CPU and not jax-metal on Apple Silicon: jax-metal is effectively
abandoned (the JAX project closed all jax-metal issues as "no development"
in 2025-12, and it does not work with current macOS/jaxlib), and the
experimental MLX-based community backends are not trustworthy for a project
whose value rests on the numerical correctness of reverse-mode AD.  The
native arm64 CPU backend of stock ``jax`` is stable and fast enough for
2D work and small 3D smoke
tests, so it is the supported local backend.

Measured on an Apple M1 Pro (6P+2E cores, jax/jaxlib 0.10.1), the stock
CPU backend already runs the gradenna FDTD solvers at full speed and no
globally beneficial XLA flag was found, so the Apple Silicon preset is
intentionally minimal.  What was measured (scripts/benchmark.py
reproduces it):

- ``--xla_cpu_multi_thread_eigen=false``, ``--xla_cpu_enable_fast_math=
  true``, ``--xla_cpu_use_onednn=true``: all within run-to-run noise
  (the first one ~15% slower for 3D f32).
- Intra-op thread count (env var ``NPROC``, honored by this jaxlib;
  the classic ``XLA_FLAGS=intra_op_parallelism_threads=N`` no longer
  parses): ``NPROC=2`` is ~20% faster for small 2D grids (256^2) but
  ~25% *slower* for 3D (64^3); ``NPROC=6`` (P-cores only) is within
  noise of the default.  Because the optimum is workload-dependent and
  the default is never far from the best, no thread setting is part of
  the preset; set ``NPROC=2`` yourself for small-2D-only workloads.

Typical use, before the first ``import jax``::

    from gradenna.platform import apply_recommended_env
    apply_recommended_env()          # 'auto' -> cpu preset on a Mac
    import jax
"""

from __future__ import annotations

import os
import sys
import warnings

__all__ = ["recommended_env", "apply_recommended_env"]

#: Apple Silicon (and generic) CPU preset.
#:
#: - ``JAX_PLATFORMS=cpu``: pin the backend explicitly so behaviour does not
#:   change if an experimental Metal/MLX PJRT plugin happens to be installed
#:   (research note 08: such plugins must never be picked up silently).
#:
#: Thread-count and XLA codegen flags are deliberately absent: on an M1 Pro
#: none of them was a global win for the 2D/3D FDTD kernels (see the module
#: docstring and ``scripts/benchmark.py``).  XLA's default intra-op thread
#: pool (one thread per logical core) is already at/near the optimum.
_CPU_ENV: dict[str, str] = {
    "JAX_PLATFORMS": "cpu",
}

#: CUDA preset (for RunPod/Colab/Kaggle class machines; research note 08).
#:
#: - ``XLA_PYTHON_CLIENT_MEM_FRACTION=0.85``: leave headroom for the CUDA
#:   context and host-side buffers instead of the default 75%-then-grow
#:   behaviour; FDTD checkpointed AD is memory-bound, so an explicit
#:   fraction makes OOM behaviour predictable.
#: - ``XLA_PYTHON_CLIENT_PREALLOCATE=true``: keep preallocation (the
#:   default) explicit; one large arena avoids fragmentation during the
#:   optimizer loop.
_CUDA_ENV: dict[str, str] = {
    "JAX_PLATFORMS": "cuda,cpu",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
}

_PRESETS = ("auto", "cpu", "apple-silicon", "cuda")


def _resolve(platform: str) -> str:
    if platform not in _PRESETS:
        raise ValueError(f"unknown platform {platform!r}; expected one of {_PRESETS}")
    if platform == "auto":
        # Darwin/arm64 (and any machine without an obvious GPU stack) -> cpu.
        # We only auto-select 'cuda' when an NVIDIA driver is clearly present.
        if sys.platform != "darwin" and (
            os.path.exists("/proc/driver/nvidia/version")
            or os.environ.get("NVIDIA_VISIBLE_DEVICES") not in (None, "", "void")
        ):
            return "cuda"
        return "cpu"
    if platform == "apple-silicon":
        return "cpu"
    return platform


def recommended_env(platform: str = "auto") -> dict[str, str]:
    """Return the recommended JAX environment variables for `platform`.

    Args:
        platform: ``'auto'`` (detect; Darwin always maps to the CPU preset),
            ``'cpu'`` / ``'apple-silicon'`` (Apple Silicon / generic CPU
            preset) or ``'cuda'``.

    Returns:
        A fresh ``dict[str, str]`` of environment variables to set *before*
        importing jax.  The CPU preset is minimal on purpose: on Apple
        Silicon every extra XLA flag we benchmarked was neutral or harmful
        (see the module docstring).
    """
    preset = _CPU_ENV if _resolve(platform) == "cpu" else _CUDA_ENV
    return dict(preset)


def apply_recommended_env(platform: str = "auto", *, override: bool = False) -> dict[str, str]:
    """Write :func:`recommended_env` into ``os.environ``.

    Must be called before the first ``import jax``: JAX reads these
    variables at import/backend-initialization time, so changes made
    afterwards are silently ignored (a ``RuntimeWarning`` is emitted in
    that case, and the variables are still set for child processes).

    Args:
        platform: passed to :func:`recommended_env`.
        override: if False (default), variables the user already set are
            left untouched; if True, the preset wins.

    Returns:
        The dict of variables that were actually written.
    """
    if "jax" in sys.modules:
        warnings.warn(
            "apply_recommended_env() called after jax was imported; "
            "the new environment variables will not affect this process",
            RuntimeWarning,
            stacklevel=2,
        )
    applied: dict[str, str] = {}
    for key, value in recommended_env(platform).items():
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied
