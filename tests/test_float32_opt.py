"""float32 topology-optimization regression (underflow rescue).

Consumer GPUs run float32 by default and Apple-Silicon CPU is fastest in
float32, so topology optimization must survive ``JAX_ENABLE_X64=0``. The
historical failure (recorded in examples/optimize_2d_antenna.py) was that the
radiated Poynting flux through a gray rho=0.5 absorber underflowed float32 to
exactly zero, killing the gradient and freezing the optimizer at iteration 0.

Two distinct underflow sites and their fixes are exercised here, both in
*subprocesses* with ``JAX_ENABLE_X64`` controlled explicitly (the suite-wide
conftest forces x64 on, which we must override to actually test float32):

1. ``test_shrunk_to_optimizes_in_float32``: the real shrunk TO problem of
   tests/test_optimization.py, run in plain float32 (x64 off, complex64 DFT).
   With the slab-psi storage and the float64-built exact-phase DFT tables this
   already works -- fields stay well above the float32 floor and the
   ``P_avail`` normalization keeps ``P_rad/P_avail`` at order 1e-7..1e-3 -- so
   we assert finite *non-zero* gradients and a real improvement. This is the
   acceptance criterion: float32 TO is not frozen.

2. ``test_dft_dtype_rescues_phasor_underflow``: a deliberately extreme thick
   absorbing wall drives the *accumulated DFT phasor* below the complex64
   normal minimum (~1e-38) while the time-domain fields are still alive. Here
   the default complex64 accumulator underflows the flux to exactly 0.0, and
   ``dft_dtype=complex128`` recovers a finite positive flux. complex128 only
   takes effect under ``JAX_ENABLE_X64=1`` (JAX silently truncates complex128
   to complex64 otherwise), so this subprocess keeps the fields float32 but
   enables x64 -- the realistic GPU-with-headroom configuration.
"""

import os
import subprocess
import sys
import textwrap

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")


def run_py(code, x64):
    """Run a Python snippet in a subprocess with a chosen JAX_ENABLE_X64."""
    env = {
        **os.environ,
        "PYTHONPATH": SRC,
        "JAX_ENABLE_X64": "1" if x64 else "0",
        "JAX_PLATFORMS": "cpu",
    }
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"subprocess failed (rc={proc.returncode})\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return proc.stdout


# Shrunk problem shared by both snippets (matches tests/test_optimization.py).
_PROBLEM = """
    import jax, jax.numpy as jnp, numpy as np
    from gradenna import (
        CPMLSpec, Grid2D, Port, alpha_max_for_fmin, gaussian_pulse_for_band,
        half_step_dft, poynting_flux_box_2d, sigma_from_density, simulate_tm,
    )
    from gradenna.topopt import DesignTransform, beta_schedule, gray_indicator

    DX = 2e-3
    NX = NY = 80
    F0 = 2.45e9
    F_MIN, F_MAX = 1.5e9, 3.5e9
    RS = 50.0
    N_STEPS = 1400
    PORT_IJ = (40, 36)
    DESIGN = (slice(32, 48), slice(32, 48))
    FLUX_BOX = (13, 66, 13, 66)
    SIGMA_MAX, SIGMA_MIN = 1e5, 1e-4

    grid = Grid2D(nx=NX, ny=NY, dx=DX, dy=DX)
    cpml = CPMLSpec(thickness=10, alpha_max=alpha_max_for_fmin(F_MIN))
    pulse = gaussian_pulse_for_band(F_MIN, F_MAX)
    t = (jnp.arange(N_STEPS) + 0.5) * grid.dt
    vs = pulse(t)
    # P_avail computed in float64 numpy so it is exact regardless of x64.
    vs64 = np.asarray(pulse(t), np.float64)
    n = np.arange(N_STEPS)
    ker = np.exp(-2j * np.pi * F0 * (n + 0.5) * float(grid.dt)) * float(grid.dt)
    p_avail = abs(ker @ vs64) ** 2 / (8.0 * RS)

    mask = np.ones((16, 16), bool)
    mask[PORT_IJ[0] - 32, PORT_IJ[1] - 32] = False
    mask = jnp.asarray(mask)
"""


def test_shrunk_to_optimizes_in_float32():
    """Plain float32 (complex64 DFT): non-zero gradient and >=2x improvement."""
    out = run_py(
        _PROBLEM
        + """
    assert jnp.zeros(1).dtype == jnp.float32, jnp.zeros(1).dtype
    import optax

    def radiated_fraction(rho):
        sig_design = jnp.where(mask, sigma_from_density(rho, SIGMA_MIN, SIGMA_MAX), 0.0)
        sigma = jnp.zeros(grid.shape).at[DESIGN].set(sig_design)
        res = simulate_tm(
            grid, ports=(Port(ij=PORT_IJ, resistance=RS, voltage=vs),),
            sigma=sigma, dft_freqs=(F0,), cpml=cpml,
        )
        p_rad = poynting_flux_box_2d(res.dft_ez, res.dft_hx, res.dft_hy, grid, FLUX_BOX)[0]
        return p_rad / p_avail

    transform = DesignTransform(radius_cells=2.0)
    theta = jnp.zeros((16, 16))
    p_initial = float(radiated_fraction(transform(theta, 8.0)))

    # The gradient at the gray start must be finite and non-zero (not frozen).
    g0 = jax.grad(lambda th: radiated_fraction(transform(th, 8.0)))(theta)
    g0n = float(jnp.linalg.norm(g0))
    assert bool(jnp.all(jnp.isfinite(g0))), "non-finite start gradient in float32"
    assert g0n > 0.0, "float32 start gradient underflowed to exactly zero (frozen)"

    opt = optax.adam(0.1)
    opt_state = opt.init(theta)

    def objective(th, beta):
        return -radiated_fraction(transform(th, beta))

    @jax.jit
    def step(th, st, beta):
        loss, grads = jax.value_and_grad(objective)(th, beta)
        updates, st = opt.update(grads, st, th)
        th = optax.apply_updates(th, updates)
        return th, st, loss, grads

    schedule = beta_schedule((8.0, 32.0), 15)
    last_gn = None
    for i in range(30):
        theta, opt_state, loss, grads = step(theta, opt_state, jnp.asarray(schedule(i)))
        assert bool(jnp.all(jnp.isfinite(grads))), f"non-finite gradient at iter {i}"
        last_gn = float(jnp.linalg.norm(grads))

    p_final = float(radiated_fraction(transform(theta, 32.0)))
    print(f"FLOAT32 p_initial={p_initial:.6e} p_final={p_final:.6e} "
          f"ratio={p_final / p_initial:.3f} g0_norm={g0n:.6e} gN_norm={last_gn:.6e} "
          f"dft={jnp.zeros(1, jnp.complex64).dtype}")
    """,
        x64=False,
    )
    line = next(ln for ln in out.splitlines() if ln.startswith("FLOAT32"))
    fields = dict(tok.split("=") for tok in line.split()[1:])
    p_initial, p_final = float(fields["p_initial"]), float(fields["p_final"])
    ratio, g0n = float(fields["ratio"]), float(fields["g0_norm"])
    assert p_initial > 0.0
    assert g0n > 0.0
    # Same acceptance bar as the float64 regression: >=2x normalized improvement.
    assert ratio >= 2.0, f"float32 improvement only {ratio:.2f}x"


def test_dft_dtype_rescues_phasor_underflow():
    """A thick absorber underflows the complex64 DFT phasor; complex128 saves it.

    Demonstrates the ``dft_dtype`` lever: with the field state float32 a thick
    high-sigma wall pushes the accumulated flux below the complex64 floor to
    exactly 0.0, while complex128 keeps a finite positive value. complex128 is
    real only under x64 (else JAX truncates it), so the subprocess enables x64
    but pins every field-affecting array to float32 -- fields stay float32,
    only the DFT carry is high precision.
    """
    out = run_py(
        _PROBLEM
        + """
    # x64 is ON here so complex128 is a real dtype; keep all field inputs f32.
    vs32 = pulse(t).astype(jnp.float32)

    def flux(dft_dtype, wall_sigma, wall_cells):
        rho = jnp.zeros((16, 16), jnp.float32)  # gray start
        sig_design = jnp.where(mask, sigma_from_density(rho, SIGMA_MIN, SIGMA_MAX), 0.0)
        sigma = jnp.zeros(grid.shape, jnp.float32).at[DESIGN].set(sig_design)
        # Thick absorbing frame around the source, inside the flux contour.
        for d in range(wall_cells):
            a, b = 20 + d, 59 - d
            sigma = (sigma.at[20:60, a].set(wall_sigma).at[20:60, b].set(wall_sigma)
                             .at[a, 20:60].set(wall_sigma).at[b, 20:60].set(wall_sigma))
        res = simulate_tm(
            grid, ports=(Port(ij=PORT_IJ, resistance=RS, voltage=vs32),),
            sigma=sigma, dft_freqs=(F0,), dft_dtype=dft_dtype, cpml=cpml,
        )
        assert res.ez.dtype == jnp.float32, res.ez.dtype  # fields stay f32
        assert res.dft_ez.dtype == dft_dtype, res.dft_ez.dtype
        p = poynting_flux_box_2d(res.dft_ez, res.dft_hx, res.dft_hy, grid, FLUX_BOX)[0]
        return float(p)

    # Wall tuned so the complex64 phasor underflows but fields are still alive.
    WS, WC = 1e6, 2
    p_c64 = flux(jnp.complex64, WS, WC)
    p_c128 = flux(jnp.complex128, WS, WC)
    print(f"RESCUE p_c64={p_c64:.6e} p_c128={p_c128:.6e}")
    """,
        x64=True,
    )
    line = next(ln for ln in out.splitlines() if ln.startswith("RESCUE"))
    fields = dict(tok.split("=") for tok in line.split()[1:])
    p_c64, p_c128 = float(fields["p_c64"]), float(fields["p_c128"])
    # complex64 underflows this flux to exactly zero; complex128 recovers it.
    assert p_c64 == 0.0, f"expected complex64 underflow to 0, got {p_c64:e}"
    assert p_c128 != 0.0 and abs(p_c128) > 0.0, "complex128 did not rescue the flux"


def test_log_radiated_fraction_matches_ratio_and_grad():
    """``log_radiated_fraction`` is monotone in the ratio with scale-free grad."""
    out = run_py(
        """
    import jax, jax.numpy as jnp
    from gradenna import log_radiated_fraction
    p_avail = 4.4e-22
    # value: log P_rad - log P_avail == log(P_rad / P_avail)
    for p_rad in (1e-26, 1e-30, 3.6e-29):
        lhs = float(log_radiated_fraction(p_rad, p_avail))
        rhs = float(jnp.log(p_rad / p_avail))
        assert abs(lhs - rhs) < 1e-6, (lhs, rhs)
    # gradient wrt p_rad is 1/p_rad (scale-invariant), not (1/p_avail).
    g = float(jax.grad(lambda x: log_radiated_fraction(x, p_avail))(1e-26))
    assert abs(g - 1e26) / 1e26 < 1e-5, g
    print("LOGOK")
    """,
        x64=True,
    )
    assert "LOGOK" in out
