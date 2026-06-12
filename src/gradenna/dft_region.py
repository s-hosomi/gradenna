"""Design-region-limited running-DFT monitors for the 3D solver.

Recording the full-grid running DFT of all six field components costs
``n_freq x grid`` complex numbers per component, and reverse-mode AD puts
the whole DFT carry on the tape every step. For the frequency-domain
adjoint (:mod:`gradenna.freq_adjoint`) the contraction only ever reads the
running-DFT phasors on the *design region* (forward + adjoint) and on the
*objective region* (forward only), so accumulating the DFT on those slabs
alone cuts both the steady-state carry and the residual byte budget by the
grid/slab ratio.

This module holds the data structures and pure-JAX helpers that describe
those slabs and scatter them back onto a full grid:

* :class:`FieldSlab` -- a static half-open index box ``[lo, hi)`` in one
  component's own array coordinates.
* :class:`DFTRegions` -- one optional slab per field component (``None``
  means the component is not accumulated, so it stays off the carry/tape).
* :class:`RegionDFTMonitor` -- the slab spectra returned by
  :func:`gradenna.simulate_3d` when ``dft_regions`` is given.

The objective-region helpers (:func:`port_regions`, :func:`ntff_box_regions`,
:func:`field_regions`) return slabs that *completely cover* every index the
corresponding objective reads (so that :func:`scatter_full`'s zero-fill of
the rest of the grid produces the exact same cotangent as the full-grid
path -- see the module-level comment on each helper).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class FieldSlab(NamedTuple):
    """A static half-open index box ``[lo, hi)`` in one component's coords.

    ``lo``/``hi`` are plain Python ``int`` tuples (not arrays): the slab
    bounds are static so the accumulated array has a compile-time shape.
    """

    lo: tuple[int, int, int]
    hi: tuple[int, int, int]

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.hi[0] - self.lo[0], self.hi[1] - self.lo[1], self.hi[2] - self.lo[2])

    def slices(self) -> tuple[slice, slice, slice]:
        return (
            slice(self.lo[0], self.hi[0]),
            slice(self.lo[1], self.hi[1]),
            slice(self.lo[2], self.hi[2]),
        )


class DFTRegions(NamedTuple):
    """One optional :class:`FieldSlab` per field component.

    A ``None`` entry means the component is not accumulated; it then carries
    no DFT array (so it never lands on the scan carry or the reverse tape).
    """

    ex: FieldSlab | None
    ey: FieldSlab | None
    ez: FieldSlab | None
    hx: FieldSlab | None
    hy: FieldSlab | None
    hz: FieldSlab | None


class RegionDFTMonitor(NamedTuple):
    """Running-DFT phasors recorded on per-component slabs.

    ``freqs``: (n_freq,) evaluation frequencies [Hz].
    ``regions``: the :class:`DFTRegions` that defined the slabs.
    ``ex..hz``: complex ``(n_freq, *slab_shape)`` spectra, or ``None`` for a
        component that was not accumulated. The dt-scale and phase
        conventions match :class:`gradenna.fdtd3d.DFTMonitor`
        (E at ``(n+1) dt``, H at ``(n+1/2) dt``).
    """

    freqs: jnp.ndarray
    regions: DFTRegions
    ex: jnp.ndarray | None
    ey: jnp.ndarray | None
    ez: jnp.ndarray | None
    hx: jnp.ndarray | None
    hy: jnp.ndarray | None
    hz: jnp.ndarray | None


# Full-grid component shapes (without the leading n_freq axis), in the same
# order as DFTRegions / RegionDFTMonitor / DFTMonitor.
def full_field_shapes(nx: int, ny: int, nz: int):
    """The six full-grid component shapes for an ``(nx, ny, nz)`` cell grid."""
    return (
        (nx - 1, ny, nz),  # ex
        (nx, ny - 1, nz),  # ey
        (nx, ny, nz - 1),  # ez
        (nx, ny - 1, nz - 1),  # hx
        (nx - 1, ny, nz - 1),  # hy
        (nx - 1, ny - 1, nz),  # hz
    )


def union_slab(a: FieldSlab | None, b: FieldSlab | None) -> FieldSlab | None:
    """Smallest slab covering both ``a`` and ``b`` (axis-wise min lo, max hi).

    ``None`` is the empty set: ``union_slab(None, b) == b``.
    """
    if a is None:
        return b
    if b is None:
        return a
    lo = tuple(min(a.lo[d], b.lo[d]) for d in range(3))
    hi = tuple(max(a.hi[d], b.hi[d]) for d in range(3))
    return FieldSlab(lo, hi)


def union_regions(a: DFTRegions, b: DFTRegions) -> DFTRegions:
    """Component-wise :func:`union_slab` of two :class:`DFTRegions`."""
    return DFTRegions(*(union_slab(x, y) for x, y in zip(a, b)))


def _normalize_slice(s: slice, n: int) -> tuple[int, int]:
    """Resolve a (possibly open/negative) cell slice to ``(start, stop)`` ints.

    Step must be 1 or None (the design-region / monitor-region convention).
    """
    if s.step not in (None, 1):
        raise ValueError(f"design/monitor region slices must have step 1, got {s}")
    start, stop, _ = s.indices(n)
    return int(start), int(stop)


def design_region_to_slabs(design_region, full_shapes) -> DFTRegions:
    """Map a cell-slice ``design_region`` onto Ex/Ey/Ez slabs (Hx=Hy=Hz=None).

    The frequency-domain adjoint reads the design phasors as
    ``fd.ex[(slice(None),) + design_region]`` etc. (one ``sigma``/``eps_r``
    cell drives the three E edges of its node), so the cell slice is applied
    **unchanged** to each E component -- start/stop only clamped to that
    component's own array shape. Changing this mapping would change the
    recorded phasors and break gradient parity with the full-grid path.
    """
    ex_shape, ey_shape, ez_shape = full_shapes[0], full_shapes[1], full_shapes[2]
    if len(design_region) != 3:
        raise ValueError("design_region must be a 3-tuple of slices")

    def slab_for(shape):
        bounds = [_normalize_slice(design_region[d], shape[d]) for d in range(3)]
        lo = tuple(b[0] for b in bounds)
        hi = tuple(b[1] for b in bounds)
        return FieldSlab(lo, hi)

    return DFTRegions(slab_for(ex_shape), slab_for(ey_shape), slab_for(ez_shape),
                      None, None, None)


def field_regions(monitor_region, full_shapes) -> DFTRegions:
    """Ex/Ey/Ez slabs for a cell-slice field objective (same map as design)."""
    return design_region_to_slabs(monitor_region, full_shapes)


def port_regions(port_ijk, full_shapes) -> DFTRegions:
    """Slabs covering every index ``_port_v_i_3d`` reads for a lumped RVS port.

    The 3D port reconstruction (``freq_adjoint._port_v_i_3d``) reads, at the
    port edge ``(i, j, k)``:

    * ``Ez[i, j, k]``                    -> Ez ``[i, i+1) x [j, j+1) x [k, k+1)``
    * ``Hy[i, j, k]``, ``Hy[i-1, j, k]`` -> Hy ``[i-1, i+1) x [j, j+1) x [k, k+1)``
    * ``Hx[i, j, k]``, ``Hx[i, j-1, k]`` -> Hx ``[i, i+1) x [j-1, j+1) x [k, k+1)``

    Ex/Ey/Hz are unused (``None``).
    """
    i, j, k = (int(v) for v in np.asarray(port_ijk).reshape(3))
    ez = FieldSlab((i, j, k), (i + 1, j + 1, k + 1))
    hy = FieldSlab((i - 1, j, k), (i + 1, j + 1, k + 1))
    hx = FieldSlab((i, j - 1, k), (i + 1, j + 1, k + 1))
    return DFTRegions(None, None, ez, hx, hy, None)


def ntff_box_regions(grid, box_margin, full_shapes) -> DFTRegions:
    """Per-component bounding-box slabs covering every index ``ntff_3d`` reads.

    ``ntff_3d`` (gradenna.ntff) integrates over the six faces of the box at
    ``box_margin`` cells inside the boundary, averaging the staggered Yee
    components onto the patch centers. With

        i0, i1 = m, nx-1-m;  j0, j1 = m, ny-1-m;  k0, k1 = m, nz-1-m

    and cell/shifted-cell ranges ``[i0, i1)`` / ``[i0+1, i1+1)`` (and likewise
    for j, k), the union of all per-face slices (including the +-1 cell H
    averages and the ``i-1``/``j-1``/``k-1`` neighbours) gives, per component:

        Ex: [i0, i1)   x [j0, j1+1) x [k0, k1+1)
        Ey: [i0, i1+1) x [j0, j1)   x [k0, k1+1)
        Ez: [i0, i1+1) x [j0, j1+1) x [k0, k1)
        Hx: [i0, i1+1) x [j0-1, j1+1) x [k0-1, k1+1)
        Hy: [i0-1, i1+1) x [j0, j1+1) x [k0-1, k1+1)
        Hz: [i0-1, i1+1) x [j0-1, j1+1) x [k0, k1+1)

    Each is clamped to the component's own array shape. (Derived by reading
    ntff_3d face by face; verified against the full-grid NTFF gradient.)
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    m = int(box_margin)
    i0, i1 = m, nx - 1 - m
    j0, j1 = m, ny - 1 - m
    k0, k1 = m, nz - 1 - m

    def clamp(lo, hi, shape):
        lo = tuple(max(0, lo[d]) for d in range(3))
        hi = tuple(min(shape[d], hi[d]) for d in range(3))
        return FieldSlab(lo, hi)

    ex = clamp((i0, j0, k0), (i1, j1 + 1, k1 + 1), full_shapes[0])
    ey = clamp((i0, j0, k0), (i1 + 1, j1, k1 + 1), full_shapes[1])
    ez = clamp((i0, j0, k0), (i1 + 1, j1 + 1, k1), full_shapes[2])
    hx = clamp((i0, j0 - 1, k0 - 1), (i1 + 1, j1 + 1, k1 + 1), full_shapes[3])
    hy = clamp((i0 - 1, j0, k0 - 1), (i1 + 1, j1 + 1, k1 + 1), full_shapes[4])
    hz = clamp((i0 - 1, j0 - 1, k0), (i1 + 1, j1 + 1, k1 + 1), full_shapes[5])
    return DFTRegions(ex, ey, ez, hx, hy, hz)


def scatter_full(mon: RegionDFTMonitor, full_shapes):
    """Zero-fill each slab spectrum onto its full-grid component.

    Returns a :class:`gradenna.fdtd3d.DFTMonitor` (full-grid phasors). A
    ``None`` slab becomes an all-zeros full-grid component. Written with JAX
    ops so that ``jax.grad`` folds a full-grid cotangent back onto the slab
    (the rest of the grid being a constant zero).
    """
    from gradenna.fdtd3d import DFTMonitor

    n_freq = mon.freqs.shape[0]
    regions = mon.regions

    def scatter(slab, arr, full_shape):
        if slab is None or arr is None:
            # No accumulation here: a constant zero full-grid component. dtype
            # follows the other slabs (or complex128 if every slab is None).
            dtype = arr.dtype if arr is not None else jnp.complex128
            return jnp.zeros((n_freq,) + tuple(full_shape), dtype)
        full = jnp.zeros((n_freq,) + tuple(full_shape), arr.dtype)
        return full.at[(slice(None),) + slab.slices()].set(arr)

    comps = (mon.ex, mon.ey, mon.ez, mon.hx, mon.hy, mon.hz)
    full = tuple(
        scatter(slab, arr, shape)
        for slab, arr, shape in zip(regions, comps, full_shapes)
    )
    return DFTMonitor(mon.freqs, *full)
