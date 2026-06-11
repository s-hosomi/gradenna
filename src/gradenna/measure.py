"""Measurement I/O and sim-vs-measurement S11 comparison (scikit-rf based).

Workflow (Phase 4): export simulated S11 with :func:`save_touchstone`,
capture the fabricated antenna with a NanoVNA (``scripts/nanovna_capture.py``),
then quantify the agreement with :func:`compare_s11` and visualize it with
:func:`plot_s11_comparison`.

scikit-rf is a dev-group dependency; this module is not imported by the
simulation core.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import skrf


def load_touchstone(path: str | Path) -> skrf.Network:
    """Load a Touchstone file (.s1p/.s2p/...) as an ``skrf.Network``."""
    return skrf.Network(str(path))


def save_touchstone(path: str | Path, f, s11, z0: float = 50.0) -> skrf.Network:
    """Save a one-port S11 trace as a Touchstone .s1p file.

    Parameters
    ----------
    path:
        Output file path (``.s1p`` extension recommended; scikit-rf appends
        the proper extension if missing).
    f:
        Frequencies [Hz], shape (n,).
    s11:
        Complex reflection coefficient, shape (n,).
    z0:
        Reference impedance [ohm].

    Returns
    -------
    skrf.Network
        The network that was written (handy for chaining/plotting).
    """
    f = np.asarray(f, dtype=float)
    s11 = np.asarray(s11, dtype=complex)
    if f.ndim != 1 or f.shape != s11.shape:
        raise ValueError(f"f and s11 must be 1-D with equal length, got {f.shape} and {s11.shape}")
    freq = skrf.Frequency.from_f(f, unit="Hz")
    ntwk = skrf.Network(frequency=freq, s=s11.reshape(-1, 1, 1), z0=z0)
    # 'ri' (real/imag) preserves full precision better than mag/angle forms.
    ntwk.write_touchstone(str(path), form="ri")
    return ntwk


@dataclass(frozen=True)
class S11Comparison:
    """Result of :func:`compare_s11`.

    Attributes
    ----------
    f_res_sim, f_res_meas:
        Resonant frequencies [Hz] (location of the |S11| minimum, refined by
        parabolic interpolation around the discrete minimum).
    f_res_shift_pct:
        Relative resonance shift (f_res_sim - f_res_meas) / f_res_meas * 100 [%].
    rms_diff_db:
        RMS of (|S11|_sim - |S11|_meas) in dB over the comparison band, with
        the measurement interpolated onto the simulation grid.
    s11_res_sim_db, s11_res_meas_db:
        |S11| [dB] at the respective resonances.
    s11_res_diff_db:
        s11_res_sim_db - s11_res_meas_db [dB].
    band:
        (f_lo, f_hi) [Hz] actually used for the comparison.
    """

    f_res_sim: float
    f_res_meas: float
    f_res_shift_pct: float
    rms_diff_db: float
    s11_res_sim_db: float
    s11_res_meas_db: float
    s11_res_diff_db: float
    band: tuple[float, float]


def _db(s) -> np.ndarray:
    """|S| in dB, floored to avoid -inf at exact zeros."""
    return 20.0 * np.log10(np.maximum(np.abs(s), 1e-30))


def _refine_minimum(f: np.ndarray, y_db: np.ndarray) -> tuple[float, float]:
    """Locate the minimum of y_db(f) with sub-grid parabolic refinement.

    Fits a parabola through the discrete minimum and its two neighbours
    (in a locally centered coordinate for conditioning). Falls back to the
    grid point at band edges or for degenerate fits.
    """
    i = int(np.argmin(y_db))
    if i == 0 or i == len(y_db) - 1:
        return float(f[i]), float(y_db[i])
    x = f[i - 1 : i + 2] - f[i]
    a, b, c = np.polyfit(x, y_db[i - 1 : i + 2], 2)
    if a <= 0.0:  # not convex -> no refinement
        return float(f[i]), float(y_db[i])
    xm = float(np.clip(-b / (2.0 * a), x[0], x[2]))
    return float(f[i] + xm), float(np.polyval([a, b, c], xm))


def compare_s11(
    f_sim,
    s11_sim,
    network_meas: skrf.Network,
    band: tuple[float, float] | None = None,
) -> S11Comparison:
    """Compare a simulated S11 trace against a measured one-port network.

    The measurement is interpolated (linearly, on real/imag parts) onto the
    simulation frequency grid restricted to the overlapping band (further
    restricted to ``band`` if given). Resonances are found independently on
    each trace's native grid to avoid interpolation bias.

    Parameters
    ----------
    f_sim:
        Simulation frequencies [Hz], shape (n,).
    s11_sim:
        Simulated complex S11, shape (n,).
    network_meas:
        Measured one-port ``skrf.Network`` (e.g. from :func:`load_touchstone`).
    band:
        Optional (f_lo, f_hi) [Hz] restriction of the comparison band.

    Returns
    -------
    S11Comparison
    """
    f_sim = np.asarray(f_sim, dtype=float)
    s11_sim = np.asarray(s11_sim, dtype=complex)
    if network_meas.nports != 1:
        raise ValueError(f"expected a one-port network, got {network_meas.nports} ports")
    f_meas = np.asarray(network_meas.f, dtype=float)
    s11_meas = np.asarray(network_meas.s[:, 0, 0], dtype=complex)

    lo = max(f_sim.min(), f_meas.min())
    hi = min(f_sim.max(), f_meas.max())
    if band is not None:
        lo = max(lo, float(band[0]))
        hi = min(hi, float(band[1]))
    if lo >= hi:
        raise ValueError("simulation, measurement and band frequency ranges do not overlap")

    m_sim = (f_sim >= lo) & (f_sim <= hi)
    m_meas = (f_meas >= lo) & (f_meas <= hi)
    if m_sim.sum() < 3 or m_meas.sum() < 3:
        raise ValueError("need at least 3 frequency points from each trace inside the band")

    fg = f_sim[m_sim]
    sim_db = _db(s11_sim[m_sim])
    # Interpolate the complex measurement onto the simulation grid.
    meas_on_sim = np.interp(fg, f_meas, s11_meas.real) + 1j * np.interp(fg, f_meas, s11_meas.imag)
    rms_diff_db = float(np.sqrt(np.mean((sim_db - _db(meas_on_sim)) ** 2)))

    f_res_sim, s11_res_sim_db = _refine_minimum(fg, sim_db)
    f_res_meas, s11_res_meas_db = _refine_minimum(f_meas[m_meas], _db(s11_meas[m_meas]))

    return S11Comparison(
        f_res_sim=f_res_sim,
        f_res_meas=f_res_meas,
        f_res_shift_pct=100.0 * (f_res_sim - f_res_meas) / f_res_meas,
        rms_diff_db=rms_diff_db,
        s11_res_sim_db=s11_res_sim_db,
        s11_res_meas_db=s11_res_meas_db,
        s11_res_diff_db=s11_res_sim_db - s11_res_meas_db,
        band=(float(lo), float(hi)),
    )


def plot_s11_comparison(
    f_sim,
    s11_sim,
    network_meas: skrf.Network,
    path: str | Path,
) -> Path:
    """Save a |S11| [dB] simulation-vs-measurement comparison plot.

    Uses a standalone matplotlib Figure (no pyplot global state, backend
    independent), so it is safe in headless environments and tests.

    Returns the output path.
    """
    from matplotlib.figure import Figure

    f_sim = np.asarray(f_sim, dtype=float)
    s11_sim = np.asarray(s11_sim, dtype=complex)
    f_meas = np.asarray(network_meas.f, dtype=float)
    s11_meas = np.asarray(network_meas.s[:, 0, 0], dtype=complex)

    fig = Figure(figsize=(7.0, 4.5))
    ax = fig.subplots()
    ax.plot(f_sim / 1e9, _db(s11_sim), label="simulation", color="C0")
    ax.plot(f_meas / 1e9, _db(s11_meas), label="measurement", color="C1", ls="--")
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("|S11| [dB]")
    ax.set_title("S11: simulation vs measurement")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=150)
    return path
