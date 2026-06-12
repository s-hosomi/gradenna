"""Acceptance tests for the differentiable thin-wire MoM backend.

Covers the four required checks for the piecewise-sinusoidal (PWS) Galerkin
thin-wire dipole solver in :mod:`gradenna.mom`:

  1. Textbook value -- half-wave dipole input impedance, validated via the
     classic resonance near ``0.47-0.48 lambda`` (Im(Zin) -> 0, Re ~ 70 ohm)
     and a loose bound on the 0.5 lambda value.  The PWS-Galerkin thin-wire
     dipole at a/lambda = 1e-3 settles around Re ~ 85 ohm at exactly 0.5 lambda
     (vs. the 73 ohm of the idealized infinitely-thin sinusoidal-current
     model); the resonant-length crossing is the robust textbook anchor and is
     what the spec recommends using.
  2. Convergence -- doubling the number of modes shrinks the change in Zin.
  3. Gradient -- jax.grad of Re(Zin) vs. central finite difference, rel err
     <= 1e-4 (matches the project's existing gradient-check style).
  4. Reciprocity / symmetry -- the impedance matrix is symmetric.

Run with::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/test_mom.py -q
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from gradenna.constants import C0  # noqa: E402
from gradenna.mom import (  # noqa: E402
    wire_dipole_impedance,
    wire_impedance_matrix,
)

LAM = 1.0  # work in wavelengths; frequency follows
F0 = C0 / LAM
A_OVER_LAM = 1e-3  # thin wire, ~ lambda/1000


# ---------------------------------------------------------------------------
# 1. Textbook value: half-wave dipole and resonant length.
# ---------------------------------------------------------------------------
def test_halfwave_dipole_impedance_order_of_magnitude():
    """0.5 lambda dipole: Re ~ 70-95 ohm, Im positive (inductive, near resonance)."""
    zin = complex(wire_dipole_impedance(0.5 * LAM, A_OVER_LAM * LAM, F0, n_modes=41))
    # Thin-wire PWS-Galerkin settles ~85 ohm; classic idealized value is 73 ohm.
    assert 65.0 < zin.real < 95.0, f"Re(Zin)={zin.real}"
    # At 0.5 lambda the dipole is slightly above resonance -> inductive (Im > 0),
    # textbook ~ +42.5 ohm for the idealized case.
    assert 20.0 < zin.imag < 65.0, f"Im(Zin)={zin.imag}"


def test_resonant_length_imag_crossing():
    """Im(Zin) changes sign through zero in the classic 0.47-0.48 lambda window."""
    lo = complex(wire_dipole_impedance(0.47 * LAM, A_OVER_LAM * LAM, F0, n_modes=41))
    hi = complex(wire_dipole_impedance(0.48 * LAM, A_OVER_LAM * LAM, F0, n_modes=41))
    assert lo.imag < 0.0, f"Im at 0.47 lambda should be capacitive, got {lo.imag}"
    assert hi.imag > 0.0, f"Im at 0.48 lambda should be inductive, got {hi.imag}"

    # Bisect for the resonant length and check it lands in 0.46-0.49 lambda with
    # a physical resistance (~70 ohm) there.
    a, b = 0.47, 0.48
    for _ in range(25):
        m = 0.5 * (a + b)
        zm = complex(wire_dipole_impedance(m * LAM, A_OVER_LAM * LAM, F0, n_modes=41))
        if zm.imag > 0.0:
            b = m
        else:
            a = m
    l_res = 0.5 * (a + b)
    assert 0.46 < l_res < 0.49, f"resonant length {l_res} lambda out of range"
    zr = complex(wire_dipole_impedance(l_res * LAM, A_OVER_LAM * LAM, F0, n_modes=41))
    assert abs(zr.imag) < 1.0, f"Im at resonance not ~0: {zr.imag}"
    assert 55.0 < zr.real < 85.0, f"Re at resonance {zr.real} ohm unphysical"


# ---------------------------------------------------------------------------
# 2. Convergence in the number of modes.
# ---------------------------------------------------------------------------
def test_convergence_with_n_modes():
    """Doubling the mode count shrinks the change in Zin (Cauchy convergence)."""
    z = [
        complex(wire_dipole_impedance(0.5 * LAM, A_OVER_LAM * LAM, F0, n_modes=n))
        for n in (21, 41, 81)
    ]
    d1 = abs(z[1] - z[0])
    d2 = abs(z[2] - z[1])
    assert d2 < d1, f"not converging: |dZ| 21->41 = {d1}, 41->81 = {d2}"


# ---------------------------------------------------------------------------
# 3. Gradient vs. finite difference.
# ---------------------------------------------------------------------------
def test_gradient_re_zin_wrt_length():
    """jax.grad of Re(Zin) w.r.t. length matches central finite difference."""
    n_seg = 41

    def re_zin(length):
        return jnp.real(
            wire_dipole_impedance(length, A_OVER_LAM * LAM, F0, n_modes=n_seg)
        )

    l0 = 0.5 * LAM
    g = float(jax.grad(re_zin)(l0))
    h = 1e-7 * LAM
    fd = float((re_zin(l0 + h) - re_zin(l0 - h)) / (2.0 * h))
    rel = abs(g - fd) / abs(fd)
    assert rel <= 1e-4, f"grad {g} vs fd {fd}, rel err {rel}"


def test_gradient_re_zin_wrt_radius():
    """jax.grad of Re(Zin) w.r.t. radius matches central finite difference."""
    n_seg = 41

    def re_zin(radius):
        return jnp.real(
            wire_dipole_impedance(0.5 * LAM, radius, F0, n_modes=n_seg)
        )

    a0 = A_OVER_LAM * LAM
    g = float(jax.grad(re_zin)(a0))
    h = 1e-7 * a0
    fd = float((re_zin(a0 + h) - re_zin(a0 - h)) / (2.0 * h))
    rel = abs(g - fd) / abs(fd)
    assert rel <= 1e-4, f"grad {g} vs fd {fd}, rel err {rel}"


# ---------------------------------------------------------------------------
# 4. Reciprocity / symmetry of the impedance matrix.
# ---------------------------------------------------------------------------
def test_impedance_matrix_symmetric():
    """Z = Z^T (reciprocity) to numerical precision."""
    z_mat = np.asarray(
        wire_impedance_matrix(0.5 * LAM, A_OVER_LAM * LAM, F0, n_modes=25)
    )
    asym = np.max(np.abs(z_mat - z_mat.T))
    scale = np.max(np.abs(z_mat))
    assert asym / scale < 1e-10, f"matrix not symmetric: asym/scale = {asym / scale}"


# ---------------------------------------------------------------------------
# Misc API checks.
# ---------------------------------------------------------------------------
def test_scalar_vs_array_freqs():
    """Scalar freq returns a scalar; array freq returns one Zin per frequency."""
    zs = wire_dipole_impedance(0.5 * LAM, A_OVER_LAM * LAM, F0, n_modes=21)
    assert jnp.ndim(zs) == 0
    za = wire_dipole_impedance(
        0.5 * LAM, A_OVER_LAM * LAM, jnp.array([F0, 1.1 * F0]), n_modes=21
    )
    assert za.shape == (2,)
    assert complex(za[0]) == complex(zs)


def test_even_modes_rejected():
    """Even mode counts (no mode at the feed) are rejected."""
    with pytest.raises(ValueError):
        wire_dipole_impedance(0.5 * LAM, A_OVER_LAM * LAM, F0, n_modes=40)


def test_x64_guard_fires():
    """RuntimeError is raised when jax_enable_x64 is False."""
    # jax.config.jax_enable_x64 is a property; toggle via jax.config.update.
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="JAX_ENABLE_X64"):
            wire_dipole_impedance(0.5 * LAM, A_OVER_LAM * LAM, F0, n_modes=21)
    finally:
        jax.config.update("jax_enable_x64", True)
