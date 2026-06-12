"""Tests for gradenna.platform presets and the benchmark CLI."""

import json
import os
import subprocess
import sys

import pytest

from gradenna.platform import apply_recommended_env, recommended_env

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")

#: Every key any preset is allowed to touch.  Extend deliberately.
KNOWN_KEYS = {
    "JAX_PLATFORMS",
    "XLA_FLAGS",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "NPROC",
}

# Snippet that loads gradenna.platform via importlib so that gradenna's
# __init__.py (which eagerly imports jax) is never executed first.  This
# allows subprocess tests to verify "apply before jax import" behaviour.
# The absolute path to platform.py is injected at test-collection time so
# the snippet works under `python -c` (where __file__ is undefined).
_PLATFORM_PY = os.path.join(SRC, "gradenna", "platform.py")
_LOAD_PLATFORM = (
    "import importlib.util\n"
    "_spec = importlib.util.spec_from_file_location(\n"
    f"    '_gradenna_platform', {_PLATFORM_PY!r},\n"
    ")\n"
    "_mod = importlib.util.module_from_spec(_spec)\n"
    "_spec.loader.exec_module(_mod)\n"
    "apply_recommended_env = _mod.apply_recommended_env\n"
    "recommended_env = _mod.recommended_env\n"
)


def run_py(code, env_extra=None, env_remove=()):
    env = {**os.environ, "PYTHONPATH": SRC, **(env_extra or {})}
    for k in env_remove:
        env.pop(k, None)
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )


@pytest.mark.parametrize("platform", ["auto", "cpu", "apple-silicon", "cuda"])
def test_recommended_env_returns_str_dict_with_known_keys(platform):
    env = recommended_env(platform)
    assert isinstance(env, dict) and env
    for k, v in env.items():
        assert isinstance(k, str) and isinstance(v, str)
    assert set(env) <= KNOWN_KEYS


def test_recommended_env_cpu_pins_backend():
    assert recommended_env("cpu")["JAX_PLATFORMS"] == "cpu"
    assert recommended_env("apple-silicon") == recommended_env("cpu")


def test_recommended_env_cuda_has_mem_fraction():
    env = recommended_env("cuda")
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION" in env
    assert 0.0 < float(env["XLA_PYTHON_CLIENT_MEM_FRACTION"]) <= 1.0


def test_recommended_env_rejects_unknown_platform():
    with pytest.raises(ValueError, match="unknown platform"):
        recommended_env("tpu")


def test_recommended_env_returns_fresh_copy():
    a = recommended_env("cpu")
    a["JAX_PLATFORMS"] = "mutated"
    assert recommended_env("cpu")["JAX_PLATFORMS"] == "cpu"


def test_apply_sets_os_environ_before_jax_import():
    # Fresh interpreter: apply *before* jax is imported, then verify both
    # os.environ and that jax actually selects the CPU backend.
    # gradenna.platform is loaded via importlib so that gradenna/__init__.py
    # (which eagerly imports jax) never runs before apply_recommended_env().
    code = (
        "import json, os, sys\n"
        + _LOAD_PLATFORM
        + "applied = apply_recommended_env('cpu')\n"
        "import jax\n"
        "print(json.dumps({'applied': applied,"
        " 'env': {k: os.environ.get(k) for k in applied},"
        " 'backend': jax.default_backend()}))\n"
    )
    # Strip JAX_PLATFORMS from the inherited env (conftest sets it as a
    # default) so the subprocess starts with a clean slate and
    # apply_recommended_env() has something to actually write.
    proc = run_py(code, env_remove=("JAX_PLATFORMS",))
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["env"]["JAX_PLATFORMS"] == "cpu"
    assert out["applied"]["JAX_PLATFORMS"] == "cpu"
    assert out["backend"] == "cpu"


def test_apply_does_not_override_user_env_by_default():
    code = (
        "import os\n"
        "from gradenna.platform import apply_recommended_env\n"
        "applied = apply_recommended_env('cpu')\n"
        "print(os.environ['JAX_PLATFORMS'], sorted(applied))\n"
    )
    proc = run_py(code, env_extra={"JAX_PLATFORMS": "cpu,user"})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split()[0] == "cpu,user"


def test_apply_warns_when_jax_already_imported():
    code = (
        "import warnings, jax\n"
        "from gradenna.platform import apply_recommended_env\n"
        "with warnings.catch_warnings(record=True) as w:\n"
        "    warnings.simplefilter('always')\n"
        "    apply_recommended_env('cpu')\n"
        "assert any(issubclass(x.category, RuntimeWarning) for x in w), w\n"
        "print('WARNED')\n"
    )
    proc = run_py(code, env_extra={"JAX_PLATFORMS": "cpu"})
    assert proc.returncode == 0, proc.stderr
    assert "WARNED" in proc.stdout


def test_apply_no_warning_when_jax_not_imported():
    # gradenna.platform is loaded via importlib so that gradenna/__init__.py
    # (which eagerly imports jax) never runs before apply_recommended_env().
    code = (
        "import warnings, sys\n"
        + _LOAD_PLATFORM
        + "with warnings.catch_warnings(record=True) as w:\n"
        "    warnings.simplefilter('always')\n"
        "    apply_recommended_env('cpu')\n"
        "assert not w, [str(x.message) for x in w]\n"
        "print('CLEAN')\n"
    )
    proc = run_py(code)
    assert proc.returncode == 0, proc.stderr
    assert "CLEAN" in proc.stdout


def test_benchmark_quick_runs():
    script = os.path.join(REPO, "scripts", "benchmark.py")
    proc = subprocess.run(
        [sys.executable, script, "--quick"],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": SRC},
    )
    assert proc.returncode == 0, proc.stderr
    assert "Mcell-steps/s" in proc.stdout
    # Four data rows: {2D, 3D} x {float32, float64}.
    assert proc.stdout.count("float32") == 2
    assert proc.stdout.count("float64") == 2
