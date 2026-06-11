"""End-to-end fabrication demo: design a patch antenna, export Gerber.

Designs a 2.45 GHz rectangular patch on FR-4 with the Balanis
transmission-line equations, rasterizes it (patch + probe-feed pad) onto a
density map at PCB resolution, runs the manufacturing checks (JLCPCB
5 mil rules) and writes RS-274X Gerber + SVG preview.

Run:  uv run python examples/patch_to_gerber.py
Outputs land in examples/fab_output/.
"""

from pathlib import Path

import numpy as np

from gradenna.designs import patch_design
from gradenna.fab import density_to_gerber, write_svg

F0 = 2.45e9
EPS_FR4 = 4.3
H_SUB = 1.6e-3
DX_MM = 0.2  # raster pitch [mm]
MARGIN_MM = 5.0

w, length, eps_reff = patch_design(F0, EPS_FR4, H_SUB)
print(f"Balanis design @ {F0 / 1e9:.2f} GHz on FR-4 (eps_r={EPS_FR4}, h={H_SUB * 1e3:.1f} mm):")
print(f"  W = {w * 1e3:.2f} mm, L = {length * 1e3:.2f} mm, eps_reff = {eps_reff:.3f}")

w_mm, l_mm = w * 1e3, length * 1e3
nx = int(round((w_mm + 2 * MARGIN_MM) / DX_MM))
ny = int(round((l_mm + 2 * MARGIN_MM) / DX_MM))
rho = np.zeros((ny, nx))

# Patch rectangle (rows = y, cols = x; fab.py convention x = col * dx).
i0 = int(round(MARGIN_MM / DX_MM))
j0 = int(round(MARGIN_MM / DX_MM))
rho[i0 : i0 + int(round(l_mm / DX_MM)), j0 : j0 + int(round(w_mm / DX_MM))] = 1.0

out_dir = Path(__file__).with_name("fab_output")
out_dir.mkdir(exist_ok=True)
result = density_to_gerber(rho, DX_MM, out_dir / "patch_top_copper.gbr")
write_svg(result.polygons, out_dir / "patch_top_copper.svg")

print(f"\npolygons: {len(result.polygons.geoms)}, total area "
      f"{result.polygons.area:.1f} mm^2 (patch alone: {w_mm * l_mm:.1f} mm^2)")
print(f"min-width violations: {len(result.width_violations)}")
print(f"min-gap violations:   {len(result.gap_violations)}")
print(f"wrote {out_dir / 'patch_top_copper.gbr'}")
print(f"wrote {out_dir / 'patch_top_copper.svg'}")
