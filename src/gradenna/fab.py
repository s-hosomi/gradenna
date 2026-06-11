"""Fabrication pipeline: density map -> polygons -> Gerber (RS-274X).

Non-differentiable post-processing. Converts an optimized density map
(values in [0, 1] on a uniform grid) into manufacturable copper geometry:

1. Vectorize with marching squares (``skimage.measure.find_contours``).
2. Assemble shapely polygons with holes via even-odd nesting.
3. Run simple DRC checks (minimum width / minimum gap, JLCPCB 5/5 mil).
4. Emit a Gerber file using regions (G36/G37 contours) via ``gerber_writer``,
   plus an SVG preview fallback.

Coordinate convention: ``rho[row, col]`` maps to ``y = row * dx_mm``,
``x = col * dx_mm`` (pixel centers); each pixel occupies a ``dx_mm`` square
footprint centered on its sample point. All output coordinates are in mm.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
from gerber_writer import DataLayer, Path as GerberPath, set_generation_software
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, unary_union
from skimage.measure import find_contours

__all__ = [
    "FabResult",
    "GapViolation",
    "WidthViolation",
    "check_min_gap",
    "check_min_width",
    "density_to_gerber",
    "density_to_polygons",
    "write_gerber",
    "write_svg",
]

# JLCPCB 2-layer 1 oz capability: 5 mil minimum trace width and clearance.
MIN_WIDTH_MM_JLCPCB = 0.127
MIN_GAP_MM_JLCPCB = 0.127

# Known layer aliases -> Gerber .FileFunction values (Gerber spec 2023.08).
_LAYER_FUNCTIONS = {
    "copper_top": "Copper,L1,Top",
    "copper_bottom": "Copper,L2,Bot",
}

PolygonsLike = Union[Polygon, MultiPolygon, Sequence[Polygon]]


class WidthViolation(NamedTuple):
    """A region of copper thinner than the minimum manufacturable width."""

    location: Tuple[float, float]  # representative point (x, y) in mm
    area_mm2: float
    geometry: Polygon


class GapViolation(NamedTuple):
    """A pair of copper islands closer than the minimum clearance."""

    indices: Tuple[int, int]  # indices into the polygon part list
    distance_mm: float
    location: Tuple[float, float]  # midpoint of the closing segment, mm


class FabResult(NamedTuple):
    """Return value of :func:`density_to_gerber`."""

    polygons: MultiPolygon
    width_violations: List[WidthViolation]
    gap_violations: List[GapViolation]
    gerber_path: str
    svg_path: Optional[str]


def _iter_parts(polygons: PolygonsLike) -> List[Polygon]:
    """Normalize Polygon / MultiPolygon / sequence input to a list of parts."""
    if isinstance(polygons, BaseGeometry):
        if polygons.is_empty:
            return []
        if isinstance(polygons, Polygon):
            return [polygons]
        # MultiPolygon or GeometryCollection: keep polygonal parts only.
        return [g for g in polygons.geoms if isinstance(g, Polygon) and not g.is_empty]
    parts: List[Polygon] = []
    for geom in polygons:
        parts.extend(_iter_parts(geom))
    return parts


def _as_multipolygon(geom) -> MultiPolygon:
    return MultiPolygon(_iter_parts(geom))


def density_to_polygons(
    rho: np.ndarray,
    dx_mm: float,
    threshold: float = 0.5,
    simplify_tol_mm: float = 0.02,
) -> MultiPolygon:
    """Vectorize a density map into shapely polygons (coordinates in mm).

    Marching squares (sub-pixel, hole-aware) extracts iso-contours at
    ``threshold``; contours are assembled into polygons with holes using
    even-odd containment nesting, then Douglas-Peucker simplified.

    Args:
        rho: 2D density array, values in [0, 1], indexed ``[row, col]``.
        dx_mm: Grid pitch in mm.
        threshold: Iso-level separating copper (>=) from void (<).
        simplify_tol_mm: Douglas-Peucker tolerance in mm (~1/4 of the
            minimum feature width is a good default).

    Returns:
        MultiPolygon in mm. Empty MultiPolygon if no copper.
    """
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 2:
        raise ValueError(f"rho must be 2D, got shape {rho.shape}")
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")
    if dx_mm <= 0.0:
        raise ValueError(f"dx_mm must be positive, got {dx_mm}")

    # Pad with void so every contour closes inside the array.
    padded = np.pad(rho, 1, mode="constant", constant_values=0.0)
    contours = find_contours(padded, level=threshold)

    rings: List[Polygon] = []
    for contour in contours:
        if len(contour) < 4:  # cannot form a ring
            continue
        # (row, col) in padded indices -> (x, y) in mm; remove 1-sample pad.
        x = (contour[:, 1] - 1.0) * dx_mm
        y = (contour[:, 0] - 1.0) * dx_mm
        ring = Polygon(np.column_stack((x, y)))
        if not ring.is_valid:
            ring = ring.buffer(0)
        for part in _iter_parts(ring):
            if part.area > 0.0:
                rings.append(part)

    if not rings:
        return MultiPolygon([])

    # Even-odd nesting: rings contained in an even number of other rings are
    # shells; odd-depth rings are holes of their smallest containing shell.
    order = sorted(range(len(rings)), key=lambda i: rings[i].area, reverse=True)
    depths = []
    parents = []
    for i in order:
        pt = rings[i].representative_point()
        # Contours never cross, so "strictly larger ring containing a point of
        # this ring" is equivalent to full containment (and avoids the trap of
        # a representative point of an outer ring landing inside its hole).
        containers = [
            j
            for j in order
            if j != i and rings[j].area > rings[i].area and rings[j].contains(pt)
        ]
        depths.append(len(containers))
        # Smallest containing ring is the direct parent (rings are nested).
        parent = min(containers, key=lambda j: rings[j].area, default=None)
        parents.append(parent)

    holes_of = {i: [] for i in order}
    for idx, i in enumerate(order):
        if depths[idx] % 2 == 1 and parents[idx] is not None:
            holes_of[parents[idx]].append(rings[i].exterior)

    polys = []
    for idx, i in enumerate(order):
        if depths[idx] % 2 == 0:
            poly = Polygon(rings[i].exterior, holes_of[i])
            if not poly.is_valid:
                poly = poly.buffer(0)
            polys.append(poly)

    merged = unary_union(polys)
    if simplify_tol_mm > 0.0:
        merged = merged.simplify(simplify_tol_mm, preserve_topology=True)
    return _as_multipolygon(merged)


def check_min_width(
    polygons: PolygonsLike,
    min_width_mm: float = MIN_WIDTH_MM_JLCPCB,
    area_tol_mm2: Optional[float] = None,
) -> List[WidthViolation]:
    """Detect copper features thinner than ``min_width_mm``.

    Morphological opening (erode by half the minimum width, then dilate
    back) removes any feature too thin to manufacture; whatever the opening
    fails to recover is reported as a violation. Mitre joins preserve sharp
    corners so plain rectangles do not trigger false positives.

    Args:
        polygons: Geometry in mm.
        min_width_mm: Minimum manufacturable width (JLCPCB 5 mil default).
        area_tol_mm2: Ignore residues smaller than this (numerical noise).
            Defaults to ``min_width_mm**2 / 4``.

    Returns:
        List of :class:`WidthViolation`, empty if the geometry is clean.
    """
    if area_tol_mm2 is None:
        area_tol_mm2 = min_width_mm**2 / 4.0
    radius = min_width_mm / 2.0
    violations: List[WidthViolation] = []
    for poly in _iter_parts(polygons):
        eroded = poly.buffer(-radius, join_style="mitre", mitre_limit=5.0)
        opened = eroded.buffer(radius, join_style="mitre", mitre_limit=5.0)
        # Tiny outward epsilon absorbs floating-point slivers along edges.
        residue = poly.difference(opened.buffer(1e-6))
        for part in _iter_parts(residue):
            if part.area > area_tol_mm2:
                pt = part.representative_point()
                violations.append(
                    WidthViolation(location=(pt.x, pt.y), area_mm2=part.area, geometry=part)
                )
    return violations


def check_min_gap(
    polygons: PolygonsLike,
    min_gap_mm: float = MIN_GAP_MM_JLCPCB,
) -> List[GapViolation]:
    """Detect pairs of copper islands closer than ``min_gap_mm``.

    Each disjoint part is dilated by half the minimum gap; parts whose
    dilations merge are closer than the manufacturable clearance.

    Args:
        polygons: Geometry in mm.
        min_gap_mm: Minimum clearance (JLCPCB 5 mil default).

    Returns:
        List of :class:`GapViolation`, empty if the geometry is clean.
    """
    parts = _iter_parts(polygons)
    radius = min_gap_mm / 2.0
    dilated = [p.buffer(radius) for p in parts]
    violations: List[GapViolation] = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            if not dilated[i].intersects(dilated[j]):
                continue
            a, b = nearest_points(parts[i], parts[j])
            violations.append(
                GapViolation(
                    indices=(i, j),
                    distance_mm=parts[i].distance(parts[j]),
                    location=((a.x + b.x) / 2.0, (a.y + b.y) / 2.0),
                )
            )
    return violations


def _add_ring(path: GerberPath, coords) -> None:
    """Append one closed contour (subpath) to a gerber_writer Path."""
    coords = list(coords)
    path.moveto(tuple(coords[0]))
    for pt in coords[1:]:
        path.lineto(tuple(pt))
    if tuple(coords[-1]) != tuple(coords[0]):
        path.lineto(tuple(coords[0]))


def write_gerber(polygons: PolygonsLike, path: str, layer: str = "copper_top") -> str:
    """Write polygons as Gerber regions (G36/G37), RS-274X / Gerber X2.

    Each polygon part becomes one region; holes are written as additional
    closed subpaths of the same region (nested contours, even-odd fill per
    the Gerber Layer Format specification, section 4.10).

    Args:
        polygons: Geometry in mm.
        path: Output file path.
        layer: Either a known alias ('copper_top', 'copper_bottom') or a
            raw Gerber .FileFunction string such as 'Copper,L1,Top'.

    Returns:
        The output path.
    """
    function = _LAYER_FUNCTIONS.get(layer, layer)
    set_generation_software("gradenna", "gradenna.fab", "0.1")
    data = DataLayer(function)
    for poly in _iter_parts(polygons):
        region = GerberPath()
        _add_ring(region, poly.exterior.coords)
        for interior in poly.interiors:
            _add_ring(region, interior.coords)
        data.add_region(region, "Conductor")
    with open(path, "w") as fh:
        data.dump_gerber(fh)
    return path


def write_svg(polygons: PolygonsLike, path: str, margin_mm: float = 1.0) -> str:
    """Write a simple SVG preview (mm units, y axis flipped, even-odd fill).

    Args:
        polygons: Geometry in mm.
        path: Output file path.
        margin_mm: Whitespace margin around the geometry.

    Returns:
        The output path.
    """
    parts = _iter_parts(polygons)
    if parts:
        minx, miny, maxx, maxy = MultiPolygon(parts).bounds
    else:
        minx = miny = 0.0
        maxx = maxy = 1.0
    minx -= margin_mm
    miny -= margin_mm
    maxx += margin_mm
    maxy += margin_mm
    width = maxx - minx
    height = maxy - miny

    def ring_d(coords) -> str:
        # Flip y so +y in mm points up in the rendered image.
        pts = [f"{x - minx:.4f},{maxy - y:.4f}" for x, y in coords]
        return "M " + " L ".join(pts) + " Z"

    d_parts = []
    for poly in parts:
        d_parts.append(ring_d(poly.exterior.coords))
        for interior in poly.interiors:
            d_parts.append(ring_d(interior.coords))
    d = " ".join(d_parts)

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width:.4f}mm" height="{height:.4f}mm" '
        f'viewBox="0 0 {width:.4f} {height:.4f}">\n'
        f'<rect width="100%" height="100%" fill="white"/>\n'
        f'<path d="{d}" fill="#b87333" fill-rule="evenodd" stroke="none"/>\n'
        f"</svg>\n"
    )
    with open(path, "w") as fh:
        fh.write(svg)
    return path


def density_to_gerber(
    rho: np.ndarray,
    dx_mm: float,
    path: str,
    threshold: float = 0.5,
    simplify_tol_mm: float = 0.02,
    min_width_mm: float = MIN_WIDTH_MM_JLCPCB,
    min_gap_mm: float = MIN_GAP_MM_JLCPCB,
    layer: str = "copper_top",
    svg_path: Optional[str] = None,
) -> FabResult:
    """One-shot pipeline: density map -> polygons -> DRC -> Gerber (+ SVG).

    Args:
        rho: 2D density array in [0, 1].
        dx_mm: Grid pitch in mm.
        path: Output Gerber file path.
        threshold: Binarization iso-level.
        simplify_tol_mm: Polygon simplification tolerance in mm.
        min_width_mm: Minimum width for DRC report.
        min_gap_mm: Minimum clearance for DRC report.
        layer: Gerber layer alias or .FileFunction string.
        svg_path: Optional SVG preview output path.

    Returns:
        :class:`FabResult` with the polygons and the violation reports.
        Violations are reported, not fixed; the Gerber is written regardless.
    """
    polygons = density_to_polygons(
        rho, dx_mm, threshold=threshold, simplify_tol_mm=simplify_tol_mm
    )
    width_violations = check_min_width(polygons, min_width_mm=min_width_mm)
    gap_violations = check_min_gap(polygons, min_gap_mm=min_gap_mm)
    write_gerber(polygons, path, layer=layer)
    if svg_path is not None:
        write_svg(polygons, svg_path)
    return FabResult(
        polygons=polygons,
        width_violations=width_violations,
        gap_violations=gap_violations,
        gerber_path=path,
        svg_path=svg_path,
    )
