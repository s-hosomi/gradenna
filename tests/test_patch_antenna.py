"""B9 benchmark: rectangular microstrip patch antenna.

Two-stage validation of the full 3D antenna stack — finite ground plane,
lossy FR-4 substrate, PEC patch sheet and a probe (RVS) feed:

1. ``patch_design`` implements the Balanis transmission-line design
   equations (4th ed., Sec. 14.2: Eqs. 14-1, 14-2 Hammerstad, 14-6, 14-7)
   and is unit-checked against textbook Example 14.1
   (fr = 10 GHz, eps_r = 2.2, h = 1.588 mm -> W = 11.86 mm, L = 9.06 mm).
2. A 2.45 GHz patch on FR-4 (eps_r = 4.3, h = 1.6 mm, tan d = 0.02)
   designed with those equations is modelled in `simulate_3d`; the
   simulated resonance (both the |S11| minimum and the Im{Zin} = 0
   antiresonance) must fall within +-5% of the design frequency. The
   budget absorbs the patch quantization to whole cells (~1%), the
   design equations' own accuracy (a few %), and the feed/sheet
   discretization systematics below.

FR-4 material model: constant eps_r plus a constant conductivity
converted from the loss tangent at the design frequency,
sigma = 2 pi f0 eps0 eps_r_model tan_d.

Thin-sheet metal and the pin-layer compensation
-----------------------------------------------
Materials are cell-based: cell (i, j, k) is applied to the three E edges
emanating from node (i, j, k), so a one-cell conductor sheet at plane k
unavoidably also shorts the *vertical* edges Ez(i, j, k+1/2) directly
above it. For a ground plane below a substrate this creates a layer of
vertical shorting pins inside the gap: with n_free free Ez layers out of
n_gap = n_free + 1 cell layers between the ground and patch planes, the
quasi-TEM line capacitance sees only h_C = n_free dz while the inductance
(return current flows in the ground sheet itself) sees the full
h_L = n_gap dz. The wave is slowed by sqrt(h_L / h_C) — measured -5.85%
on a sigma-sheet TM110 cavity (sqrt(9/8) predicts -5.7%) and -22% on a
2-cell patch gap (sqrt(3/2) predicts -18%), independent of the sheet
conductivity.

The principled fix used here: span the physical substrate height with
n_gap = 3 cells (the pin layer included) and scale the substrate
permittivity by n_free / n_gap = 2/3, which restores *both* L' and C' of
the discrete line to the physical microstrip values (the substrate
conductivity scales by the same factor through eps_r_model, preserving
tan d). With this compensation the simulated resonance lands within
~2.5% of the Balanis design frequency.
"""

import functools
import math

import jax.numpy as jnp
import numpy as np
import pytest

from gradenna import CPMLSpec, alpha_max_for_fmin
from gradenna.constants import EPS0
from gradenna.designs import patch_design
from gradenna.fdtd3d import Grid3D, port_impedance, simulate_3d
from gradenna.sparams import gaussian_pulse_for_band

# Design target: 2.45 GHz patch on FR-4.
F0 = 2.45e9
EPS_FR4 = 4.3
H_SUB = 1.6e-3
TAN_D = 0.02

# Discretization.
DXY = 1.2e-3       # in-plane cell (~ lambda0 / 100 at 2.45 GHz)
N_GAP = 3          # cells between ground plane and patch plane (= h / dz)
DZ = H_SUB / N_GAP
M_GND = 8          # ground/substrate margin beyond the patch [cells] (= 6 h)
M_AIR = 5          # air gap between ground edge and CPML [cells]
N_PML = 8
N_STEPS = 6400     # ~9.5 ns: pulse (2.9 ns) + ringdown of the lossy patch
SIG_METAL = 1.0e7  # thin-sheet PEC surrogate [S/m]
R_PORT = 50.0


@functools.lru_cache(maxsize=2)
def _patch_run(eps_r_sub: float):
    """Simulate the 2.45 GHz FR-4-designed patch with a given substrate eps_r.

    The geometry is always the FR-4 design (so the sensitivity test varies
    only the material). Returns (freqs [Hz], Zin(f)) as numpy arrays.
    """
    w, length, _ = patch_design(F0, EPS_FR4, H_SUB)
    npx = round(w / DXY)   # patch width W along x
    npy = round(length / DXY)  # resonant length L along y

    margin = M_GND + M_AIR + N_PML
    nx = npx + 2 * margin
    ny = npy + 2 * margin
    kg = N_PML + 3                  # ground sheet cell layer
    k_patch = kg + N_GAP            # patch sheet cell layer
    nz = k_patch + 1 + 9 + N_PML    # dragged pin layer + air + CPML above
    grid = Grid3D(nx=nx, ny=ny, nz=nz, dx=DXY, dy=DXY, dz=DZ)

    # Pin-layer compensation (see module docstring): n_free of n_gap free
    # Ez layers -> scale the substrate eps (and sigma, via eps_r_model).
    eps_r_model = eps_r_sub * (N_GAP - 1) / N_GAP
    sig_sub = 2.0 * math.pi * F0 * EPS0 * eps_r_model * TAN_D

    i0 = (nx - npx) // 2
    j0 = (ny - npy) // 2
    gx0, gx1 = i0 - M_GND, i0 + npx + M_GND
    gy0, gy1 = j0 - M_GND, j0 + npy + M_GND

    eps_r = np.ones(grid.shape)
    sigma = np.zeros(grid.shape)
    # Substrate: cell layers kg+1 .. k_patch-1 (the kg layer is the ground
    # sheet whose dragged pins fill the lowest gap layer).
    eps_r[gx0:gx1, gy0:gy1, kg + 1 : k_patch] = eps_r_model
    sigma[gx0:gx1, gy0:gy1, kg + 1 : k_patch] = sig_sub
    # Ground sheet and patch sheet (thin PEC surrogates).
    sigma[gx0:gx1, gy0:gy1, kg] = SIG_METAL
    sigma[i0 : i0 + npx, j0 : j0 + npy, k_patch] = SIG_METAL
    # Probe feed at the radiating edge, centered along W: a vertical PEC
    # pin up to the last free Ez edge, which is the 1-cell RVS port gap.
    pi, pj = i0 + npx // 2, j0 + 1
    sigma[pi, pj, kg + 1 : k_patch - 1] = SIG_METAL
    port_ijk = (pi, pj, k_patch - 1)

    pulse = gaussian_pulse_for_band(1.5e9, 3.5e9)
    t = (jnp.arange(N_STEPS) + 0.5) * grid.dt
    res = simulate_3d(
        grid,
        eps_r=jnp.asarray(eps_r),
        sigma=jnp.asarray(sigma),
        port_ijk=port_ijk,
        port_voltage=pulse(t),
        port_resistance=R_PORT,
        cpml=CPMLSpec(thickness=N_PML, alpha_max=alpha_max_for_fmin(1.0e9)),
    )

    freqs = jnp.linspace(1.7e9, 3.2e9, 301)
    zin = port_impedance(res, grid, freqs, eps_r_port=eps_r_model)
    return np.asarray(freqs), np.asarray(zin)


def _s11_dip(freqs, zin, z0=R_PORT):
    """Frequency and depth [dB] of the |S11| minimum."""
    s11 = np.abs((zin - z0) / (zin + z0))
    k = int(np.argmin(s11))
    return float(freqs[k]), float(20.0 * np.log10(s11[k]))


def _antiresonances(freqs, zin, re_min=50.0):
    """Im{Zin} = 0 crossings with Re{Zin} above re_min (parallel resonances)."""
    im, re = zin.imag, zin.real
    out = []
    for k in np.nonzero(np.diff(np.sign(im)) != 0)[0]:
        fz = freqs[k] + (freqs[k + 1] - freqs[k]) * (0.0 - im[k]) / (im[k + 1] - im[k])
        if max(re[k], re[k + 1]) >= re_min:
            out.append(float(fz))
    return out


# ---------------------------------------------------------------------------
# Design equations vs Balanis Example 14.1
# ---------------------------------------------------------------------------


def test_patch_design_matches_balanis_example_14_1():
    """fr = 10 GHz, eps_r = 2.2, h = 1.588 mm -> W = 11.86 mm, L = 9.06 mm."""
    w, length, eps_reff = patch_design(10.0e9, 2.2, 1.588e-3)
    assert w == pytest.approx(11.86e-3, rel=5e-3)
    assert length == pytest.approx(9.06e-3, rel=5e-3)
    assert eps_reff == pytest.approx(1.972, abs=5e-3)  # Balanis intermediate value


def test_patch_design_2g45_fr4_dimensions():
    """The benchmark geometry itself: ~37.6 x 29.1 mm on FR-4."""
    w, length, eps_reff = patch_design(F0, EPS_FR4, H_SUB)
    assert w == pytest.approx(37.58e-3, rel=1e-2)
    assert length == pytest.approx(29.14e-3, rel=1e-2)
    assert 3.9 < eps_reff < 4.1


# ---------------------------------------------------------------------------
# B9: FDTD resonance vs the design frequency
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_patch_resonance_within_5pct_of_design():
    """|S11| dip and Im{Zin} = 0 antiresonance both within +-5% of 2.45 GHz.

    Measured (float64, 73 x 66 x 32 grid, 6400 steps): dip at 2.390 GHz
    (-2.45%, about -10 dB) and antiresonance at 2.359 GHz (-3.7%).
    """
    freqs, zin = _patch_run(EPS_FR4)
    f_dip, dip_db = _s11_dip(freqs, zin)
    assert dip_db <= -6.0, f"no clear S11 dip: min |S11| = {dip_db:.1f} dB"
    assert abs(f_dip - F0) / F0 <= 0.05, (
        f"S11 dip at {f_dip / 1e9:.3f} GHz, {100 * (f_dip - F0) / F0:+.1f}% from design"
    )
    anti = _antiresonances(freqs, zin)
    assert anti, "no Im(Zin)=0 antiresonance found in the sweep band"
    f_anti = min(anti, key=lambda f: abs(f - F0))
    assert abs(f_anti - F0) / F0 <= 0.05, (
        f"antiresonance at {f_anti / 1e9:.3f} GHz, "
        f"{100 * (f_anti - F0) / F0:+.1f}% from design"
    )


@pytest.mark.slow
def test_resonance_shifts_down_with_substrate_eps():
    """Raising the substrate eps_r 4.3 -> 5.5 must lower the resonance ~1/sqrt(eps).

    Material-assignment regression guard: the expected shift is
    sqrt(eps_reff(5.5)/eps_reff(4.3)) ~ 1.13, far above the assert margin.
    """
    f_43, _ = _s11_dip(*_patch_run(EPS_FR4))
    f_55, _ = _s11_dip(*_patch_run(5.5))
    assert f_55 < 0.93 * f_43, (
        f"resonance did not shift down: {f_43 / 1e9:.3f} -> {f_55 / 1e9:.3f} GHz"
    )
    assert f_55 > 0.75 * f_43, "implausibly large shift; material model broken?"
