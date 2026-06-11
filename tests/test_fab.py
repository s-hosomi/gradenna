"""Tests for the fabrication pipeline (density map -> polygons -> Gerber)."""

import re

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from shapely.geometry import MultiPolygon

from gradenna.fab import (
    check_min_gap,
    check_min_width,
    density_to_gerber,
    density_to_polygons,
    write_gerber,
    write_svg,
)

DX = 0.1  # mm


def smooth_disk(shape, center_rc, radius_mm, dx_mm):
    """Anti-aliased disk density: the 0.5 iso-contour sits exactly at radius."""
    rows, cols = np.indices(shape)
    d = np.hypot(rows - center_rc[0], cols - center_rc[1]) * dx_mm
    return np.clip(0.5 + (radius_mm - d) / dx_mm, 0.0, 1.0)


# ---------------------------------------------------------------------------
# density_to_polygons
# ---------------------------------------------------------------------------


class TestDensityToPolygons:
    def test_rectangle_area_and_coords(self):
        # 40 x 20 pixel rectangle at dx=0.5mm -> 20mm x 10mm = 200 mm^2.
        dx = 0.5
        rho = np.zeros((50, 60))
        rho[10:30, 5:45] = 1.0
        polys = density_to_polygons(rho, dx)

        assert isinstance(polys, MultiPolygon)
        assert len(polys.geoms) == 1
        area = polys.area
        assert area == pytest.approx(20.0 * 10.0, rel=0.02)

        # Pixel footprint convention: edges at (index - 0.5) * dx.
        minx, miny, maxx, maxy = polys.bounds
        assert minx == pytest.approx((5 - 0.5) * dx, abs=0.01)
        assert maxx == pytest.approx((45 - 0.5) * dx, abs=0.01)
        assert miny == pytest.approx((10 - 0.5) * dx, abs=0.01)
        assert maxy == pytest.approx((30 - 0.5) * dx, abs=0.01)

    def test_ring_shell_and_hole(self):
        # Annulus: outer R=6mm, inner r=3mm on a 0.1mm grid.
        R, r = 6.0, 3.0
        outer = smooth_disk((160, 160), (80, 80), R, DX)
        inner = smooth_disk((160, 160), (80, 80), r, DX)
        rho = outer * (1.0 - inner)
        polys = density_to_polygons(rho, DX)

        assert len(polys.geoms) == 1
        poly = polys.geoms[0]
        assert len(poly.interiors) == 1
        expected = np.pi * (R**2 - r**2)
        assert poly.area == pytest.approx(expected, rel=0.02)
        # Hole area on its own.
        hole_area = np.pi * r**2
        from shapely.geometry import Polygon

        assert Polygon(poly.interiors[0]).area == pytest.approx(hole_area, rel=0.02)

    def test_empty_density(self):
        polys = density_to_polygons(np.zeros((20, 20)), DX)
        assert polys.is_empty

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            density_to_polygons(np.zeros(10), DX)
        with pytest.raises(ValueError):
            density_to_polygons(np.zeros((4, 4)), DX, threshold=1.5)
        with pytest.raises(ValueError):
            density_to_polygons(np.zeros((4, 4)), dx_mm=0.0)


# ---------------------------------------------------------------------------
# DRC checks
# ---------------------------------------------------------------------------


def pads_with_bridge(bridge_cells):
    """Two 20x20-cell pads joined by a bridge of given width in cells."""
    rho = np.zeros((40, 80))
    rho[10:30, 5:25] = 1.0
    rho[10:30, 55:75] = 1.0
    mid = 20
    half = bridge_cells // 2
    rho[mid - half : mid - half + bridge_cells, 25:55] = 1.0
    return rho


class TestCheckMinWidth:
    def test_thin_bridge_is_flagged(self):
        # 1-cell bridge at dx=0.1mm -> 0.1mm wide < 0.127mm minimum.
        polys = density_to_polygons(pads_with_bridge(1), DX)
        violations = check_min_width(polys, min_width_mm=0.127)
        assert len(violations) >= 1
        # The violation should sit on the bridge, between the pads.
        xs = [v.location[0] for v in violations]
        assert any(2.5 < x < 5.5 for x in xs)

    def test_thick_pattern_is_clean(self):
        # 5-cell bridge -> 0.5mm wide, comfortably above the minimum.
        polys = density_to_polygons(pads_with_bridge(5), DX)
        assert check_min_width(polys, min_width_mm=0.127) == []

    def test_plain_rectangle_is_clean(self):
        rho = np.zeros((30, 30))
        rho[5:25, 5:25] = 1.0
        polys = density_to_polygons(rho, DX)
        assert check_min_width(polys, min_width_mm=0.127) == []


class TestCheckMinGap:
    def test_narrow_gap_is_flagged(self):
        # Two pads separated by 1 cell -> 0.1mm gap < 0.127mm minimum.
        rho = np.zeros((30, 60))
        rho[5:25, 5:25] = 1.0
        rho[5:25, 26:46] = 1.0
        polys = density_to_polygons(rho, DX)
        assert len(polys.geoms) == 2
        violations = check_min_gap(polys, min_gap_mm=0.127)
        assert len(violations) == 1
        assert violations[0].distance_mm < 0.127

    def test_wide_gap_is_clean(self):
        # 10-cell separation -> 1.0mm gap.
        rho = np.zeros((30, 60))
        rho[5:25, 5:20] = 1.0
        rho[5:25, 30:45] = 1.0
        polys = density_to_polygons(rho, DX)
        assert len(polys.geoms) == 2
        assert check_min_gap(polys, min_gap_mm=0.127) == []


# ---------------------------------------------------------------------------
# Gerber / SVG output
# ---------------------------------------------------------------------------


def parse_gerber_region_points(text):
    """Minimal RS-274X reparse: coordinates inside G36/G37 region blocks."""
    points = []
    in_region = False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("G36"):
            in_region = True
        elif line.startswith("G37"):
            in_region = False
        elif in_region:
            m = re.match(r"X(-?\d+)Y(-?\d+)D0[12]\*", line)
            if m:
                points.append((int(m.group(1)), int(m.group(2))))
    return points


class TestWriteGerber:
    def test_gerber_file_contents(self, tmp_path):
        R, r = 6.0, 3.0
        outer = smooth_disk((160, 160), (80, 80), R, DX)
        inner = smooth_disk((160, 160), (80, 80), r, DX)
        polys = density_to_polygons(outer * (1.0 - inner), DX)

        out = tmp_path / "top.gbr"
        write_gerber(polys, str(out), layer="copper_top")

        assert out.exists()
        text = out.read_text()
        # RS-274X mandatory header: coordinate format and mm units.
        assert "%FSLAX" in text
        assert "%MOMM*%" in text
        assert "Copper,L1,Top" in text
        # Region statements present and balanced.
        assert text.count("G36*") >= 1
        assert text.count("G36*") == text.count("G37*")
        assert text.rstrip().endswith("M02*")

        # Reparse: region must contain a nontrivial number of vertices.
        points = parse_gerber_region_points(text)
        assert len(points) > 10

    def test_layer_alias_bottom(self, tmp_path):
        rho = np.zeros((20, 20))
        rho[5:15, 5:15] = 1.0
        polys = density_to_polygons(rho, DX)
        out = tmp_path / "bot.gbr"
        write_gerber(polys, str(out), layer="copper_bottom")
        assert "Copper,L2,Bot" in out.read_text()


class TestSmoothRandomDensity:
    def test_pipeline_end_to_end(self, tmp_path):
        rng = np.random.default_rng(42)
        rho = gaussian_filter(rng.random((100, 100)), sigma=4.0)
        rho = (rho - rho.min()) / (rho.max() - rho.min())

        gbr = tmp_path / "random.gbr"
        svg = tmp_path / "random.svg"
        result = density_to_gerber(rho, DX, str(gbr), svg_path=str(svg))

        assert not result.polygons.is_empty
        assert result.polygons.area > 0.0
        assert gbr.exists() and gbr.stat().st_size > 0
        assert svg.exists()
        svg_text = svg.read_text()
        assert "<svg" in svg_text and "<path" in svg_text

    def test_write_svg_empty_geometry(self, tmp_path):
        out = tmp_path / "empty.svg"
        write_svg(MultiPolygon([]), str(out))
        assert "<svg" in out.read_text()
