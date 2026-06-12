"""Cross-check gradenna against committed openEMS reference data (B9).

openEMS is *not* a CI dependency (no pip/conda package; source build only).
Instead ``benchmarks/openems_refs/generate_patch_refs.py`` is run locally and
its small CSV outputs (s11.csv, zin.csv, farfield_e.csv, farfield_h.csv) are
committed under ``benchmarks/openems_refs/``. This test loads those CSVs, runs
gradenna on the *same* physical geometry (the shared ``geometry`` module), and
compares:

    * resonance frequency (|S11| dip)      within +-2 %
    * |S11| dB curve RMS difference        <= 2 dB outside the deep null
    * E/H-plane far-field pattern shape     correlation >= 0.99

If the reference CSVs are absent (the default state until openEMS data is
committed), every test ``pytest.skip``s -- so CI stays green without openEMS.

The whole module is marked ``slow``: the gradenna patch run is the same
~73x66x32, 6400-step FDTD as ``tests/test_patch_antenna.py`` (a couple of
minutes on CPU).

Geometry / convention mapping between the two solvers is documented in
``benchmarks/openems_refs/generate_patch_refs.py``; the salient points:
  - both S11 curves are referenced to 50 ohm at the de-embedded feed plane;
  - far-field angles share the (theta from +z, phi from +x) convention, so
    the E-plane is phi = 90 deg and the H-plane is phi = 0 deg, matching
    gradenna's ``ntff_3d``.
"""

from __future__ import annotations

import functools
import math
import os
import sys

import numpy as np
import pytest

# --- locate and import the shared geometry / reference CSVs ------------------
_REFS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir, "benchmarks", "openems_refs")
)
if _REFS_DIR not in sys.path:
    sys.path.insert(0, _REFS_DIR)

# The shared geometry module lives alongside the reference data; it has no
# heavy dependencies, so importing it is always safe.
import geometry as geo  # noqa: E402

_S11_CSV = os.path.join(_REFS_DIR, "s11.csv")
_FF_E_CSV = os.path.join(_REFS_DIR, "farfield_e.csv")
_FF_H_CSV = os.path.join(_REFS_DIR, "farfield_h.csv")

# Acceptance tolerances (docs/research/07 Sec. 4).
RESONANCE_TOL = 0.02       # +-2 % on the |S11| dip frequency
S11_RMS_TOL_DB = 2.0       # RMS |S11| dB difference outside the deep null
NULL_GUARD_DB = -10.0      # samples below this (deeper null) excluded from RMS
PATTERN_CORR_MIN = 0.99    # E/H-plane pattern correlation


def _has_refs() -> bool:
    return os.path.isfile(_S11_CSV)


def _load_csv(path: str):
    """Load a ``#``-commented reference CSV into a structured array + metadata."""
    meta = {}
    n_comment = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                n_comment += 1
                if ":" in line:
                    k, _, v = line[1:].partition(":")
                    meta[k.strip()] = v.strip()
                continue
            break
    # Skip the comment block explicitly; the next line is the column header.
    data = np.genfromtxt(
        path, delimiter=",", names=True, comments=None, skip_header=n_comment
    )
    return data, meta


# ---------------------------------------------------------------------------
# gradenna side: reproduce the test_patch_antenna.py run on the shared geometry
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _gradenna_patch():
    """Run the gradenna patch and return (freqs, Zin) plus the grid/monitor.

    Mirrors ``tests/test_patch_antenna.py::_patch_run`` (same grid, pin-layer
    compensation, probe feed and port) but additionally requests an f0 DFT
    monitor so we can form the far-field cuts with ``ntff_3d``.
    """
    import jax.numpy as jnp

    from gradenna import CPMLSpec, alpha_max_for_fmin
    from gradenna.constants import EPS0
    from gradenna.designs import patch_design
    from gradenna.fdtd3d import Grid3D, port_impedance, simulate_3d
    from gradenna.sparams import gaussian_pulse_for_band

    mesh = geo.GRADENNA_MESH
    f0 = geo.F0
    dxy, dz = mesh.dxy, mesh.dz
    n_gap = mesh.n_gap

    w, length, _ = patch_design(f0, geo.EPS_FR4, geo.H_SUB)
    npx = round(w / dxy)
    npy = round(length / dxy)

    margin = mesh.m_gnd + mesh.m_air + mesh.n_pml
    nx = npx + 2 * margin
    ny = npy + 2 * margin
    kg = mesh.n_pml + 3
    k_patch = kg + n_gap
    nz = k_patch + 1 + 9 + mesh.n_pml
    grid = Grid3D(nx=nx, ny=ny, nz=nz, dx=dxy, dy=dxy, dz=dz)

    # Pin-layer compensation: scale substrate eps (and sigma via eps_r_model).
    eps_r_model = geo.EPS_FR4 * (n_gap - 1) / n_gap
    sig_sub = 2.0 * math.pi * f0 * EPS0 * eps_r_model * geo.TAN_D

    i0 = (nx - npx) // 2
    j0 = (ny - npy) // 2
    gx0, gx1 = i0 - mesh.m_gnd, i0 + npx + mesh.m_gnd
    gy0, gy1 = j0 - mesh.m_gnd, j0 + npy + mesh.m_gnd

    eps_r = np.ones(grid.shape)
    sigma = np.zeros(grid.shape)
    eps_r[gx0:gx1, gy0:gy1, kg + 1 : k_patch] = eps_r_model
    sigma[gx0:gx1, gy0:gy1, kg + 1 : k_patch] = sig_sub
    sigma[gx0:gx1, gy0:gy1, kg] = mesh.sig_metal
    sigma[i0 : i0 + npx, j0 : j0 + npy, k_patch] = mesh.sig_metal
    pi, pj = i0 + npx // 2, j0 + 1
    sigma[pi, pj, kg + 1 : k_patch - 1] = mesh.sig_metal
    port_ijk = (pi, pj, k_patch - 1)

    pulse = gaussian_pulse_for_band(*geo.PULSE_BAND)
    t = (jnp.arange(mesh.n_steps) + 0.5) * grid.dt
    res = simulate_3d(
        grid,
        eps_r=jnp.asarray(eps_r),
        sigma=jnp.asarray(sigma),
        port_ijk=port_ijk,
        port_voltage=pulse(t),
        port_resistance=geo.R_PORT,
        cpml=CPMLSpec(thickness=mesh.n_pml, alpha_max=alpha_max_for_fmin(1.0e9)),
        dft_freqs=(f0,),
    )

    freqs = np.asarray(jnp.linspace(geo.SWEEP_BAND[0], geo.SWEEP_BAND[1], geo.N_SWEEP))
    zin = np.asarray(port_impedance(res, grid, freqs, eps_r_port=eps_r_model))
    return freqs, zin, res, grid, mesh.n_pml + 1


def _s11_from_zin(zin, z0=geo.R_PORT):
    return (zin - z0) / (zin + z0)


def _dip_freq(freqs, s11):
    return float(freqs[int(np.argmin(np.abs(s11)))])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_geometry_matches_gradenna_design():
    """Sanity: the standalone geometry.patch_design equals gradenna's."""
    from gradenna.designs import patch_design as gd_patch_design

    a = geo.patch_design(geo.F0, geo.EPS_FR4, geo.H_SUB)
    b = gd_patch_design(geo.F0, geo.EPS_FR4, geo.H_SUB)
    assert a == pytest.approx(b, rel=1e-12)


@pytest.mark.slow
def test_s11_resonance_matches_openems():
    """|S11| dip frequency within +-2 % of the openEMS reference."""
    if not _has_refs():
        pytest.skip(f"no openEMS reference data at {_S11_CSV}; run generate_patch_refs.py")

    ref, meta = _load_csv(_S11_CSV)
    f_ref = _dip_freq(ref["freq_Hz"], ref["S11_re"] + 1j * ref["S11_im"])

    freqs, zin, *_ = _gradenna_patch()
    f_grad = _dip_freq(freqs, _s11_from_zin(zin))

    rel = abs(f_grad - f_ref) / f_ref
    assert rel <= RESONANCE_TOL, (
        f"resonance mismatch: gradenna {f_grad/1e9:.3f} GHz vs "
        f"openEMS {f_ref/1e9:.3f} GHz ({100*rel:.1f}% > {100*RESONANCE_TOL:.0f}%)"
    )


@pytest.mark.slow
def test_s11_curve_rms_matches_openems():
    """|S11| dB curve RMS difference <= 2 dB outside the deep null."""
    if not _has_refs():
        pytest.skip(f"no openEMS reference data at {_S11_CSV}; run generate_patch_refs.py")

    ref, meta = _load_csv(_S11_CSV)
    f_ref = ref["freq_Hz"]
    s11_ref_db = ref["S11_dB"]

    freqs, zin, *_ = _gradenna_patch()
    s11_grad_db = 20.0 * np.log10(np.maximum(np.abs(_s11_from_zin(zin)), 1e-30))
    # Interpolate gradenna's curve onto the reference frequency samples.
    s11_grad_on_ref = np.interp(f_ref, freqs, s11_grad_db)

    # Exclude the deep-null region (where small frequency shifts cause huge dB
    # swings) and stay inside the trusted comparison band.
    lo, hi = geo.COMPARE_BAND
    keep = (f_ref >= lo) & (f_ref <= hi)
    keep &= (s11_ref_db > NULL_GUARD_DB) & (s11_grad_on_ref > NULL_GUARD_DB)
    assert keep.sum() >= 10, "too few comparison points outside the null"

    rms = float(np.sqrt(np.mean((s11_grad_on_ref[keep] - s11_ref_db[keep]) ** 2)))
    assert rms <= S11_RMS_TOL_DB, (
        f"|S11| dB RMS difference {rms:.2f} dB > {S11_RMS_TOL_DB} dB"
    )


def _pattern_correlation(ref_db, grad_db):
    """Pearson correlation of two dB patterns on the same theta grid.

    gradenna's NTFF is evaluated directly on the reference CSV's theta
    samples, so no resampling is needed.
    """
    a = ref_db - ref_db.mean()
    b = grad_db - grad_db.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / denom) if denom > 0 else 0.0


@pytest.mark.slow
@pytest.mark.parametrize(
    "csv_path, phi_deg, plane",
    [(_FF_E_CSV, 90.0, "E"), (_FF_H_CSV, 0.0, "H")],
)
def test_farfield_pattern_correlation(csv_path, phi_deg, plane):
    """E/H-plane far-field pattern correlation >= 0.99 with openEMS."""
    if not os.path.isfile(csv_path):
        pytest.skip(f"no openEMS far-field reference at {csv_path}")

    import jax.numpy as jnp

    from gradenna.ntff import ntff_3d

    ref, meta = _load_csv(csv_path)
    theta_deg = ref["theta_deg"]
    ref_db = ref["E_dB_norm"]

    _, _, res, grid, ntff_margin = _gradenna_patch()
    thetas = np.deg2rad(theta_deg)
    phis = np.deg2rad([phi_deg])
    e_far = ntff_3d(res.dft, grid, ntff_margin, (geo.F0,), thetas, phis)
    e_theta = np.asarray(e_far[0, :, 0, 0])
    e_phi = np.asarray(e_far[0, :, 0, 1])
    mag = np.sqrt(np.abs(e_theta) ** 2 + np.abs(e_phi) ** 2)
    grad_db = 20.0 * np.log10(np.maximum(mag, 1e-30) / np.max(mag))

    corr = _pattern_correlation(ref_db, grad_db)
    assert corr >= PATTERN_CORR_MIN, (
        f"{plane}-plane pattern correlation {corr:.4f} < {PATTERN_CORR_MIN}"
    )
