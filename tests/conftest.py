import os

# Must be set before jax initializes. Numerical validation tests need float64.
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
