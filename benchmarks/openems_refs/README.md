# openEMS cross-check references

openEMS has no pip/conda package — the Python interface requires a source
build, so it is **not** part of CI. On macOS the upstream tap
(`thliebig/openems`) is gone; the community tap works on Apple Silicon:

```sh
brew install vinn-ie/openems/fparser
brew install vinn-ie/openems/csxcad vinn-ie/openems/openems --without-gui
# Python bindings: pip install the python/ subdirs of the CSXCAD and openEMS
# sources (set CSXCAD_INSTALL_PATH/OPENEMS_INSTALL_PATH to the brew prefixes,
# use --no-build-isolation), ideally into a dedicated venv.
```

On Linux use `update_openEMS.sh --python`, or use the community Docker image. Cross-checks follow the workflow from the project
research notes (docs/research/07, Sec. 3–4):

1. Define the benchmark geometry once and generate both the gradenna and the
   openEMS model from it. The single source of truth is `geometry.py`
   (the 2.45 GHz FR-4 patch of `tests/test_patch_antenna.py`, matching the
   openEMS `Simple_Patch_Antenna.py` tutorial). Both solvers import it.
2. Run openEMS locally / in Docker and commit the results here as small static
   CSV files (`s11.csv`, `zin.csv`, `farfield_e.csv`, `farfield_h.csv`)
   together with the openEMS version and mesh metadata (written as `#` comment
   lines at the top of each CSV).
3. CI (`tests/test_openems_refs.py`) compares gradenna's output against the
   committed reference data only — openEMS itself is never imported. Until the
   CSVs exist, those comparisons `pytest.skip`.

## Files

- `geometry.py` — solver-independent benchmark definition (`PatchGeometry`,
  `GradennaMesh`, standalone `patch_design`). Dependency-light so the openEMS
  side (no gradenna on its PYTHONPATH) and the gradenna test both consume it.
- `generate_patch_refs.py` — builds the openEMS model from `geometry.py`
  (port of `Simple_Patch_Antenna.py`), runs the FDTD, and writes the CSVs.
- `tests/test_openems_refs.py` — the CI comparison test (skips without CSVs).

## Generating the reference data (openEMS required)

After building openEMS with its Python interface:

```sh
# from this directory (benchmarks/openems_refs/)
python generate_patch_refs.py --outdir . --resolution 30
```

Useful flags:

- `--resolution N` — mesh density [cells per wavelength in the substrate]
  (30–40 typical for a patch).
- `--sim-dir DIR` — keep the raw openEMS run (HDF5 dumps); default is a temp dir.
- `--end-criteria E` — energy-decay stop criterion (`1e-4` = −40 dB).

If openEMS is not importable the script prints actionable build instructions
and exits with status 2 (it never writes partial CSVs).

Commit the four CSVs once they look reasonable; they are a few tens of KB and
embed the openEMS version + mesh size in their header comments.

## Running the comparison test

```sh
# from the repo root
JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/test_openems_refs.py -q
```

- With no committed CSVs: the four `slow` comparison tests skip; only the
  geometry sanity check (`geometry.patch_design` == `gradenna.designs`) runs.
- With CSVs present: gradenna runs the same 73×66×32, 6400-step patch as
  `tests/test_patch_antenna.py` (a couple of minutes on CPU) and asserts
  - resonance (|S11| dip) within ±2 %,
  - |S11| dB curve RMS difference ≤ 2 dB outside the deep null,
  - E/H-plane far-field pattern correlation ≥ 0.99.

The comparison tests are marked `@pytest.mark.slow`, so the default fast suite
(`-m "not slow"`) skips them regardless.

## Solver convention mapping

The two solvers differ in excitation, normalisation and reference plane; the
comparison is made apples-to-apples as follows (details in the docstrings of
`generate_patch_refs.py`):

- **S11 / Zin** — both referenced to 50 Ω at the *de-embedded feed plane*.
  openEMS uses `port.CalcPort` (shifts the reference plane to the feed);
  gradenna uses `port_impedance(..., deembed_gap=True)` (removes the discrete
  1-cell gap susceptance). S11 = (Zin − Z₀)/(Zin + Z₀) is excitation-
  independent, so only the band and Z₀ must match (they do, via `geometry`).
- **FR-4 loss** — `geometry.PatchGeometry.conductivity_sub` is the physical
  bulk σ = 2πf₀ε₀εᵣ·tanδ used by openEMS. gradenna additionally scales εᵣ/σ
  by the pin-layer factor (n_free/n_gap); that is an internal discretisation
  correction to its thin-sheet metal model, not a change to the physics.
- **Far-field angles** — both use θ from +z and φ from +x. The E-plane cut is
  φ = 90° (y–z plane, containing L) and the H-plane cut is φ = 0° (x–z plane),
  matching gradenna's `ntff_3d`. Patterns are peak-normalised in dB, so the
  test checks shape correlation, not absolute gain calibration.

Until reference data is committed, the analytic benchmarks (cylindrical wave,
infinitesimal dipole, Balanis patch design equations) remain the primary
validation; see `tests/`.
