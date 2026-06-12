#!/usr/bin/env python3
"""Generate openEMS reference data for the 2.45 GHz FR-4 patch benchmark.

Run this *locally* (or in the community Docker image) after building openEMS
with its Python interface; the produced CSV files are then committed under
``benchmarks/openems_refs/`` and consumed by ``tests/test_openems_refs.py``
in CI (which never imports openEMS itself).

The model is the project port of the upstream ``Simple_Patch_Antenna.py``
tutorial, rebuilt from the shared :mod:`geometry` definition so that gradenna
and openEMS describe the same physical antenna. See ``README.md`` and
docs/research/07 Sec. 3-4 for the workflow.

Usage
-----
    python generate_patch_refs.py --outdir .                # default 30 cells/lambda mesh
    python generate_patch_refs.py --outdir . --resolution 40
    python generate_patch_refs.py --sim-dir /tmp/openems_patch  # keep raw FDTD data

Outputs (CSV with ``#``-comment metadata header lines)
    s11.csv        : freq_Hz, S11_dB, S11_re, S11_im
    zin.csv        : freq_Hz, Zin_re, Zin_im
    farfield_e.csv : theta_deg, E_dB_norm        (E-plane cut at f0, phi = 90 deg)
    farfield_h.csv : theta_deg, E_dB_norm        (H-plane cut at f0, phi =  0 deg)

Conventions / how gradenna maps onto these references
-----------------------------------------------------
* **Reference plane & de-embedding.** openEMS' lumped/MSL port returns S11
  referenced to ``geometry.R_PORT`` (50 ohm) at the port plane it defines,
  with the port's own de-embedding applied (``port.CalcPort`` shifts the
  reference plane to the feed). gradenna's ``port_impedance(..., deembed_gap
  =True)`` removes the discrete 1-cell gap susceptance, yielding Zin at the
  feed edge; S11 is then formed with the SAME 50 ohm reference. So both
  curves are "Zin at the feed, normalised to 50 ohm" -- directly comparable.
* **Excitation / normalisation.** openEMS excites with a Gaussian
  (set_GaussExcite(f0, fc)); gradenna uses gaussian_pulse_for_band over the
  same band. S11 = (Zin - Z0)/(Zin + Z0) is excitation-independent, so only
  the band and the reference impedance must match (they do, via geometry).
* **Far-field angles.** openEMS' nf2ff returns E_theta/E_phi on a
  (theta, phi) grid with theta measured from +z and phi from +x -- the SAME
  convention as gradenna's ``ntff_3d``. We export the broadside-centred
  E-plane (phi = 90 deg, the y-z plane containing L) and H-plane
  (phi = 0 deg, the x-z plane) cuts, each normalised to its own peak in dB,
  so the comparison is a pattern-shape correlation independent of absolute
  gain calibration.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import math
import os
import sys

import numpy as np

# Shared, solver-independent geometry (import works without openEMS).
try:
    from geometry import (
        COMPARE_BAND,
        F0,
        N_SWEEP,
        PATCH,
        PULSE_BAND,
        R_PORT,
        SWEEP_BAND,
    )
except ImportError:  # allow `python benchmarks/openems_refs/generate_patch_refs.py`
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from geometry import (  # type: ignore  # noqa: E402
        COMPARE_BAND,
        F0,
        N_SWEEP,
        PATCH,
        PULSE_BAND,
        R_PORT,
        SWEEP_BAND,
    )


def _import_openems():
    """Import the openEMS Python interface or exit with a clear message.

    openEMS has no pip/conda package -- the Cython CSXCAD/openEMS modules come
    from a source build (see README.md). We fail loudly and actionably rather
    than with a bare ImportError so the "not installed yet" state is obvious.
    """
    try:
        from CSXCAD import ContinuousStructure  # noqa: F401
        from openEMS import openEMS  # noqa: F401
        from openEMS.physical_constants import C0, EPS0  # noqa: F401

        import CSXCAD
        import openEMS as openems_mod

        return CSXCAD, openems_mod
    except ImportError as exc:  # pragma: no cover - depends on local build
        sys.stderr.write(
            "ERROR: the openEMS Python interface (CSXCAD / openEMS) is not "
            "importable.\n"
            f"  underlying import error: {exc}\n\n"
            "openEMS is NOT a pip/conda package; build it from source and add "
            "its Python\nmodules to PYTHONPATH. On macOS (community tap; the "
            "upstream thliebig tap is gone):\n"
            "    brew install vinn-ie/openems/fparser\n"
            "    brew install vinn-ie/openems/csxcad vinn-ie/openems/openems "
            "--without-gui\n"
            "    pip install <CSXCAD-src>/python <openEMS-src>/python "
            "--no-build-isolation\n"
            "On Linux: clone openEMS-Project and run ./update_openEMS.sh --python\n"
            "Or use the community Docker image. See benchmarks/openems_refs/README.md.\n"
        )
        raise SystemExit(2) from exc


def _openems_version(openems_mod) -> str:
    """Best-effort openEMS version string for the CSV metadata header."""
    for attr in ("__version__", "_version", "version"):
        v = getattr(openems_mod, attr, None)
        if isinstance(v, str):
            return v
    try:  # pragma: no cover - depends on local build
        from openEMS import openEMS

        return str(getattr(openEMS, "__version__", "unknown"))
    except Exception:
        return "unknown"


def build_and_run(sim_dir: str, resolution: int, end_criteria: float = 1e-4):
    """Build the patch model from ``geometry.PATCH`` and run openEMS.

    Args:
        sim_dir: working directory for the FDTD run (HDF5 dumps etc.).
        resolution: target mesh density in cells per wavelength at the upper
            band edge inside the substrate (a typical openEMS patch mesh uses
            ~30-40). Also used to set the third-rule fine mesh near edges.
        end_criteria: openEMS energy decay end criterion (-40 dB = 1e-4).

    Returns a dict with the frequency sweep, S-parameters, input impedance and
    the two principal far-field cuts at f0, plus mesh metadata. The body uses
    the openEMS Python API exactly as in Simple_Patch_Antenna.py.
    """
    CSXCAD, openems_mod = _import_openems()
    from CSXCAD import ContinuousStructure
    from openEMS import openEMS
    from openEMS.physical_constants import C0 as OEMS_C0

    geo = PATCH
    f0 = geo.f0
    f_start, f_stop = SWEEP_BAND

    # --- openEMS FDTD object ------------------------------------------------
    fdtd = openEMS(NrTS=60000, EndCriteria=end_criteria)
    fdtd.SetGaussExcite(0.5 * (f_start + f_stop), 0.5 * (f_stop - f_start))
    fdtd.SetBoundaryCond(["MUR"] * 6)  # simple absorbing box; swap to PML if built

    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(1.0)  # work in metres

    # Convenience dimensions from the shared geometry (all in metres).
    W, L = geo.patch_w, geo.patch_l
    h = geo.h_sub
    sub_w, sub_l = geo.sub_w, geo.sub_l
    feed_inset = geo.feed_offset_y()

    # --- materials & primitives (Simple_Patch_Antenna.py layout) -----------
    # Patch in the z = h plane, centred in x, offset so its near edge (-y) is
    # the feed edge; substrate 0..h; ground at z = 0.
    patch = csx.AddMetal("patch")
    patch.AddBox(priority=10, start=[-W / 2, -L / 2, h], stop=[W / 2, L / 2, h])

    substrate = csx.AddMaterial(
        "substrate", epsilon=geo.eps_r, kappa=geo.conductivity_sub
    )
    substrate.AddBox(
        priority=0, start=[-sub_w / 2, -sub_l / 2, 0], stop=[sub_w / 2, sub_l / 2, h]
    )

    gnd = csx.AddMetal("gnd")
    gnd.AddBox(priority=10, start=[-sub_w / 2, -sub_l / 2, 0], stop=[sub_w / 2, sub_l / 2, 0])

    # Probe feed: a lumped port from ground to patch at (x = 0, y = feed edge
    # + inset). openEMS' AddLumpedPort handles excitation + de-embedded S11.
    feed_y = -L / 2 + feed_inset
    port = fdtd.AddLumpedPort(
        port_nr=1,
        R=R_PORT,
        start=[0.0, feed_y, 0.0],
        stop=[0.0, feed_y, h],
        p_dir="z",
        excite=1.0,
        priority=5,
    )

    # --- mesh (thirds rule around metal edges) ------------------------------
    lambda0_sub = OEMS_C0 / f_stop / math.sqrt(geo.eps_r)
    res = lambda0_sub / resolution
    third = res / 3.0
    mesh.AddLine("x", [-sub_w / 2, sub_w / 2, -W / 2, W / 2, 0.0])
    mesh.AddLine("y", [-sub_l / 2, sub_l / 2, -L / 2, L / 2, feed_y])
    mesh.AddLine("z", [0.0, h])
    # Air box and absorbing-boundary padding ~ lambda0/4 around the structure.
    air = OEMS_C0 / f0 / 4.0
    mesh.AddLine("x", [-sub_w / 2 - air, sub_w / 2 + air])
    mesh.AddLine("y", [-sub_l / 2 - air, sub_l / 2 + air])
    mesh.AddLine("z", [-air, h + air])
    mesh.SmoothMeshLines("all", res)
    mesh.SmoothMeshLines("z", min(third, h / 3.0))  # resolve the thin substrate

    # --- near-field-to-far-field box ---------------------------------------
    nf2ff = fdtd.CreateNF2FFBox()

    # --- run ----------------------------------------------------------------
    os.makedirs(sim_dir, exist_ok=True)
    fdtd.Run(sim_dir, verbose=1, cleanup=False)

    # --- post-process -------------------------------------------------------
    freqs = np.linspace(f_start, f_stop, N_SWEEP)
    port.CalcPort(sim_dir, freqs)  # de-embeds the reference plane to the feed
    zin = port.uf_tot / port.if_tot          # complex input impedance
    s11 = port.uf_ref / port.uf_inc          # de-embedded reflection coefficient

    # Far-field cuts at f0: E-plane (phi = 90 deg, y-z) and H-plane (phi = 0).
    theta = np.linspace(-90.0, 90.0, 181)
    ff_e = nf2ff.CalcNF2FF(sim_dir, f0, theta, [90.0], center=[0, 0, h / 2])
    ff_h = nf2ff.CalcNF2FF(sim_dir, f0, theta, [0.0], center=[0, 0, h / 2])
    e_e = np.abs(np.asarray(ff_e.E_norm[0]).reshape(-1))
    e_h = np.abs(np.asarray(ff_h.E_norm[0]).reshape(-1))
    e_e_db = 20.0 * np.log10(np.maximum(e_e, 1e-30) / np.max(e_e))
    e_h_db = 20.0 * np.log10(np.maximum(e_h, 1e-30) / np.max(e_h))

    n_cells = (mesh.GetQtyLines("x") * mesh.GetQtyLines("y") * mesh.GetQtyLines("z"))
    meta = {
        "openems_version": _openems_version(openems_mod),
        "generated_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "resolution_cells_per_lambda": resolution,
        "mesh_cells": int(n_cells),
        "nx": int(mesh.GetQtyLines("x")),
        "ny": int(mesh.GetQtyLines("y")),
        "nz": int(mesh.GetQtyLines("z")),
        "f0_Hz": f0,
        "patch_W_m": W,
        "patch_L_m": L,
        "h_sub_m": h,
        "eps_r": geo.eps_r,
        "tan_d": geo.tan_d,
        "kappa_sub_S_per_m": geo.conductivity_sub,
        "R_port_ohm": R_PORT,
        "compare_band_Hz": COMPARE_BAND,
        "boundary": "MUR",
    }
    return {
        "freqs": np.asarray(freqs),
        "s11": np.asarray(s11),
        "zin": np.asarray(zin),
        "theta_deg": theta,
        "ff_e_db": e_e_db,
        "ff_h_db": e_h_db,
        "meta": meta,
    }


def _write_csv(path: str, meta: dict, header: list[str], rows) -> None:
    """Write a CSV with ``# key: value`` metadata comment lines on top."""
    with open(path, "w", newline="") as fh:
        fh.write("# openEMS patch antenna reference data\n")
        for key, val in meta.items():
            fh.write(f"# {key}: {val}\n")
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def write_all(outdir: str, result: dict) -> list[str]:
    """Write s11/zin/farfield CSVs and return the list of written paths."""
    os.makedirs(outdir, exist_ok=True)
    meta = result["meta"]
    freqs = result["freqs"]
    s11 = result["s11"]
    zin = result["zin"]
    s11_db = 20.0 * np.log10(np.maximum(np.abs(s11), 1e-30))

    written = []

    p = os.path.join(outdir, "s11.csv")
    _write_csv(
        p, meta, ["freq_Hz", "S11_dB", "S11_re", "S11_im"],
        zip(freqs, s11_db, s11.real, s11.imag),
    )
    written.append(p)

    p = os.path.join(outdir, "zin.csv")
    _write_csv(
        p, meta, ["freq_Hz", "Zin_re", "Zin_im"], zip(freqs, zin.real, zin.imag)
    )
    written.append(p)

    p = os.path.join(outdir, "farfield_e.csv")
    _write_csv(
        p, meta, ["theta_deg", "E_dB_norm"],
        zip(result["theta_deg"], result["ff_e_db"]),
    )
    written.append(p)

    p = os.path.join(outdir, "farfield_h.csv")
    _write_csv(
        p, meta, ["theta_deg", "E_dB_norm"],
        zip(result["theta_deg"], result["ff_h_db"]),
    )
    written.append(p)

    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir", default=os.path.dirname(os.path.abspath(__file__)),
        help="directory for the committed CSV files (default: this folder)",
    )
    parser.add_argument(
        "--sim-dir", default=None,
        help="working dir for the raw openEMS run (default: a temp dir)",
    )
    parser.add_argument(
        "--resolution", type=int, default=30,
        help="target mesh density [cells per wavelength in substrate]",
    )
    parser.add_argument(
        "--end-criteria", type=float, default=1e-4,
        help="openEMS energy decay end criterion (1e-4 = -40 dB)",
    )
    args = parser.parse_args(argv)

    import tempfile

    sim_dir = args.sim_dir or tempfile.mkdtemp(prefix="openems_patch_")
    result = build_and_run(sim_dir, args.resolution, args.end_criteria)
    written = write_all(args.outdir, result)
    print("Wrote openEMS reference CSVs:")
    for p in written:
        print("  " + p)
    print(f"openEMS version: {result['meta']['openems_version']}")
    print(f"mesh: {result['meta']['mesh_cells']} cells "
          f"({result['meta']['nx']}x{result['meta']['ny']}x{result['meta']['nz']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
