# openEMS cross-check references

openEMS has no pip/conda package — the Python interface requires a source
build (`brew tap thliebig/openems && brew install --HEAD openems` on macOS,
`update_openEMS.sh --python` on Linux, or the community Docker image), so it
is **not** part of CI. Cross-checks follow the workflow from the project
research notes (docs/research/07, Sec. 4):

1. Define the benchmark geometry once and generate both the gradenna and the
   openEMS model from it (the `Simple_Patch_Antenna.py` and `RCS_Sphere.py`
   openEMS tutorials map directly onto `tests/test_patch_antenna.py` and the
   future Mie benchmark).
2. Run openEMS locally/in Docker and commit the results here as small static
   CSV files (S11(f), Zin(f), far-field cuts) together with the openEMS
   version and mesh metadata.
3. CI then only compares gradenna's output against the committed reference
   data (suggested tolerances: resonance ±2%, |S11| RMS ≤ 2 dB outside the
   deep null, pattern correlation ≥ 0.99).

Until reference data is committed, the analytic benchmarks (cylindrical
wave, infinitesimal dipole, Balanis patch design equations) serve as the
primary validation; see `tests/`.
