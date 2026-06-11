"""Tests for gradenna.measure and scripts/nanovna_capture.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from gradenna.measure import (
    S11Comparison,
    compare_s11,
    load_touchstone,
    plot_s11_comparison,
    save_touchstone,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_SCRIPT = REPO_ROOT / "scripts" / "nanovna_capture.py"


def rlc_s11(f, f0, r=10.0, l=5e-9, z0=50.0):
    """S11 of a series RLC load: resonates (|S11| minimum) at f0."""
    f = np.asarray(f, dtype=float)
    w = 2.0 * np.pi * f
    c = 1.0 / ((2.0 * np.pi * f0) ** 2 * l)
    z = r + 1j * (w * l - 1.0 / (w * c))
    return (z - z0) / (z + z0)


def test_touchstone_roundtrip_identical_traces(tmp_path):
    """Save -> load -> compare against itself: zero shift, ~zero RMS."""
    f0 = 2.45e9
    f = np.linspace(2.0e9, 3.0e9, 401)  # f0 lies exactly on this grid
    s11 = rlc_s11(f, f0)

    path = tmp_path / "meas.s1p"
    save_touchstone(path, f, s11)
    assert path.exists()

    ntwk = load_touchstone(path)
    np.testing.assert_allclose(ntwk.f, f, rtol=1e-9)
    np.testing.assert_allclose(ntwk.s[:, 0, 0], s11, atol=1e-9)

    result = compare_s11(f, s11, ntwk)
    assert isinstance(result, S11Comparison)
    assert result.f_res_shift_pct == pytest.approx(0.0, abs=1e-6)
    assert result.rms_diff_db == pytest.approx(0.0, abs=1e-5)
    assert result.s11_res_diff_db == pytest.approx(0.0, abs=1e-5)
    # The refined resonance should sit at the synthetic f0.
    assert result.f_res_sim == pytest.approx(f0, rel=1e-3)


def test_resonance_shift_on_mismatched_grids(tmp_path):
    """Shifted resonance on a different frequency grid is recovered to 0.1%."""
    f0_sim = 2.45e9
    shift = 1.01  # measurement resonates 1% higher
    f0_meas = f0_sim * shift

    f_sim = np.linspace(2.0e9, 3.0e9, 401)
    s11_sim = rlc_s11(f_sim, f0_sim)

    # Different point count and span, nodes not aligned with the sim grid.
    f_meas = np.linspace(1.9e9, 3.1e9, 357)
    s11_meas = rlc_s11(f_meas, f0_meas)
    path = tmp_path / "meas_shifted.s1p"
    save_touchstone(path, f_meas, s11_meas)

    result = compare_s11(f_sim, s11_sim, load_touchstone(path), band=(2.1e9, 2.9e9))
    expected_pct = 100.0 * (f0_sim - f0_meas) / f0_meas  # about -0.990%
    assert result.f_res_shift_pct == pytest.approx(expected_pct, abs=0.1)
    assert result.f_res_meas == pytest.approx(f0_meas, rel=1e-3)
    # The traces differ, so the band RMS must be nonzero.
    assert result.rms_diff_db > 0.0
    assert result.band[0] >= 2.1e9 and result.band[1] <= 2.9e9


def test_plot_s11_comparison_writes_file(tmp_path):
    f = np.linspace(2.0e9, 3.0e9, 201)
    s11_sim = rlc_s11(f, 2.45e9)
    path_s1p = tmp_path / "meas.s1p"
    ntwk = save_touchstone(path_s1p, f, rlc_s11(f, 2.5e9))

    out = tmp_path / "comparison.png"
    returned = plot_s11_comparison(f, s11_sim, ntwk, out)
    assert Path(returned) == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_nanovna_capture_importable_without_pynanovna():
    """The capture script must import cleanly even when pynanovna is absent."""
    spec = importlib.util.spec_from_file_location("nanovna_capture", CAPTURE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # must not raise

    assert callable(mod.main)
    assert callable(mod.parse_args)

    args = mod.parse_args(["--start", "2.0e9", "--stop", "3.0e9", "--points", "11", "--out", "x.s1p"])
    assert args.start == pytest.approx(2.0e9)
    assert args.points == 11

    if mod.pynanovna is None:
        # Without the hardware library, main() prints usage and exits nonzero.
        rc = mod.main(["--start", "2.0e9", "--stop", "3.0e9", "--out", "x.s1p"])
        assert rc != 0
