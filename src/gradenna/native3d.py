"""Native (Rust) fused 3D Yee FDTD time loop -- optional local accelerator.

The 3D analogue of :mod:`gradenna.native`. ``gradenna.fdtd3d.simulate_3d`` runs
the whole 3D time loop as a ``jax.lax.scan`` of many small XLA kernels per step;
on CPU the per-step dispatch + thread sync dominates and only ~6-13% of M1 Pro
memory bandwidth is reached (research note 17). This module loads the same
``rust/`` cdylib (now carrying a fused 3D time loop) and exposes
:func:`simulate_3d_native`, a drop-in for the forward-only subset of
:func:`gradenna.fdtd3d.simulate_3d` returning a ``SimResult3D``-compatible
object.

Design: coefficient parity, not arithmetic re-derivation (same as the 2D
module). Every coefficient table (Ca/Cb at the three E slices, the per-source
``cb_*`` with component-specific cross-sections, the 12 CPML slab b/c tables,
the six 1/kappa axis tables, the semi-implicit RVS port coefficients and the
exact-phase DFT tables) is built here exactly as :func:`simulate_3d` builds
them, then handed to the kernel flat. The Rust side does only the time
stepping, so the update order and operations match and results agree to f64
rel <= 1e-12 / f32 1e-5. Forward-only (no checkpointing; the freq-adjoint
backend hook needs only forward runs).
"""

from __future__ import annotations

import ctypes

import numpy as np

from gradenna.constants import EPS0, MU0
from gradenna.cpml import CPMLSpec, axis_coefficients, slab_slices
from gradenna.fdtd3d import DFTMonitor, SimResult3D
from gradenna.grid import Grid3D
from gradenna.native import _axis_np, _load

__all__ = ["is_available", "simulate_3d_native"]


def is_available() -> bool:
    """True if the native kernel can be loaded (cargo present + builds)."""
    from gradenna import native

    return native.is_available()


# ---------------------------------------------------------------------------
# ctypes structs (mirror the #[repr(C)] structs in rust/src/lib3d.rs)
# ---------------------------------------------------------------------------


class _DFTSlab3D(ctypes.Structure):
    """One component's DFT slab; mirrors ``DftSlab3D`` in rust/src/lib3d.rs.

    ``on == 0`` marks the component off (no accumulation, zero-sized buffer).
    Bounds are the half-open box ``[lo, hi)`` in that component's own array
    coordinates.
    """

    _fields_ = [
        ("on", ctypes.c_int),
        ("lo0", ctypes.c_int), ("hi0", ctypes.c_int),
        ("lo1", ctypes.c_int), ("hi1", ctypes.c_int),
        ("lo2", ctypes.c_int), ("hi2", ctypes.c_int),
    ]


class _DFTRegion3D(ctypes.Structure):
    """Per-component DFT slabs (ex, ey, ez, hx, hy, hz -- DFTRegions order).

    Mirrors ``DftRegion3D`` in rust/src/lib3d.rs. A NULL pointer to this struct
    selects the full-grid DFT (backward compatible).
    """

    _fields_ = [
        ("ex", _DFTSlab3D), ("ey", _DFTSlab3D), ("ez", _DFTSlab3D),
        ("hx", _DFTSlab3D), ("hy", _DFTSlab3D), ("hz", _DFTSlab3D),
    ]


class _SimParams3D(ctypes.Structure):
    _fields_ = [
        ("nx", ctypes.c_int), ("ny", ctypes.c_int), ("nz", ctypes.c_int),
        ("npml", ctypes.c_int), ("n_steps", ctypes.c_int),
        ("n_jx", ctypes.c_int), ("n_jy", ctypes.c_int), ("n_jz", ctypes.c_int),
        ("n_ports", ctypes.c_int),
        ("n_mx", ctypes.c_int), ("n_my", ctypes.c_int), ("n_mz", ctypes.c_int),
        ("n_probes", ctypes.c_int), ("n_freq", ctypes.c_int),
        ("record_energy", ctypes.c_int),
    ]


# Field names of the four pointer structs, in the exact order of the Rust
# struct definitions. They are filled with c_void_p so the same struct works
# for f32 and f64 (the kernel reads the pointee at the right width).
_COEFF_FIELDS = (
    "ca_ex", "cb_ex", "ca_ey", "cb_ey", "ca_ez", "cb_ez",
    "ikx_e", "iky_e", "ikz_e", "ikx_h", "iky_h", "ikz_h", "eps",
)

# 12 psi variables (E-type then H-type), each lo/hi b/c -> 48 pointers. The
# order matches the Rust CpmlSlabs struct.
_PSI_NAMES = (
    "exy", "exz", "eyz", "eyx", "ezx", "ezy",
    "hxz", "hxy", "hyx", "hyz", "hzy", "hzx",
)
_CPML_FIELDS = tuple(
    f"{name}_{bc}_{end}"
    for name in _PSI_NAMES
    for bc, end in (("b", "lo"), ("c", "lo"), ("b", "hi"), ("c", "hi"))
)

_SRC_FIELDS = (
    "jx_i", "jx_j", "jx_k", "cb_jx", "jx_cur",
    "jy_i", "jy_j", "jy_k", "cb_jy", "jy_cur",
    "jz_i", "jz_j", "jz_k", "cb_jz", "jz_cur",
    "mx_i", "mx_j", "mx_k", "mx_cur",
    "my_i", "my_j", "my_k", "my_cur",
    "mz_i", "mz_j", "mz_k", "mz_cur",
    "port_i", "port_j", "port_k", "a_port", "b_port", "c_port", "port_vs",
    "probe_i", "probe_j", "probe_k",
    "ph_e_re", "ph_e_im", "ph_h_re", "ph_h_im",
)

_FIELD_FIELDS = (
    "ex", "ey", "ez", "hx", "hy", "hz",
    "out_probe", "out_v", "out_i", "out_energy",
    "dft_ex_re", "dft_ex_im", "dft_ey_re", "dft_ey_im",
    "dft_ez_re", "dft_ez_im", "dft_hx_re", "dft_hx_im",
    "dft_hy_re", "dft_hy_im", "dft_hz_re", "dft_hz_im",
)


def _make_struct(name, fields):
    return type(name, (ctypes.Structure,), {"_fields_": [(f, ctypes.c_void_p) for f in fields]})


_CoeffTables = _make_struct("_CoeffTables", _COEFF_FIELDS)
_CpmlSlabs = _make_struct("_CpmlSlabs", _CPML_FIELDS)
_SourceBuffers = _make_struct("_SourceBuffers", _SRC_FIELDS)
_FieldBuffers = _make_struct("_FieldBuffers", _FIELD_FIELDS)


_CONFIGURED = set()


def _configure3d(lib, dtype) -> None:
    """Declare argtypes for the 3D entry point of the given width (once)."""
    name = "gradenna_3d_run_f64" if dtype == np.float64 else "gradenna_3d_run_f32"
    if name in _CONFIGURED:
        return
    fp = ctypes.c_double if dtype == np.float64 else ctypes.c_float
    fn = getattr(lib, name)
    fn.restype = None
    fn.argtypes = [
        ctypes.POINTER(_SimParams3D),
        ctypes.POINTER(_CoeffTables),
        ctypes.POINTER(_CpmlSlabs),
        ctypes.POINTER(_SourceBuffers),
        ctypes.POINTER(_FieldBuffers),
        fp, fp, fp, fp, fp, fp, fp, fp,  # dt_mu, inv_dx/dy/dz, dx, dy, dz, mu0
        ctypes.POINTER(_DFTRegion3D),  # NULL = full-grid DFT
    ]
    _CONFIGURED.add(name)


def _index_array_3d(ijk, name, grid, margin):
    idx = np.asarray(ijk, dtype=np.int32).reshape(-1, 3)
    lo = np.array([margin, margin, 0])
    hi = np.array([grid.nx - 1 - margin, grid.ny - 1 - margin, grid.nz - 2])
    if idx.size and (np.any(idx < lo) or np.any(idx > hi)):
        raise ValueError(f"{name} {idx.tolist()} outside the valid Ez range (margin {margin})")
    return idx


def simulate_3d_native(
    grid: Grid3D,
    *,
    eps_r=1.0,
    sigma=0.0,
    source_ijk=None,
    source_current=None,
    source_x_ijk=None,
    source_x_current=None,
    source_y_ijk=None,
    source_y_current=None,
    mx_ijk=None,
    mx_current=None,
    my_ijk=None,
    my_current=None,
    mz_ijk=None,
    mz_current=None,
    port_ijk=None,
    port_voltage=None,
    port_resistance: float = 50.0,
    probe_ijk=(),
    cpml: CPMLSpec = CPMLSpec(),
    dft_freqs=None,
    dft_regions=None,
    record_energy: bool = False,
    dtype=None,
) -> SimResult3D:
    """Run the 3D forward solve in the native Rust kernel.

    Drop-in for the forward-only subset of
    :func:`gradenna.fdtd3d.simulate_3d` (same argument names and semantics),
    returning a :class:`SimResult3D` of numpy arrays. ``dtype`` selects
    float32 / float64 (default: inferred from the inputs). The DFT
    accumulators are always kept in float64 internally (== complex128) and
    returned as complex128. Raises ``RuntimeError`` if the native library is
    unavailable; gate on :func:`is_available` for a fallback.

    ``dft_regions`` (optional :class:`gradenna.dft_region.DFTRegions`) limits
    the running DFT to per-component slabs: ``result.dft`` is then a
    :class:`gradenna.dft_region.RegionDFTMonitor` whose slab spectra match the
    corresponding slices of the full-grid DFT (a ``None`` component slab gives
    a ``None`` spectrum). Without it the full-grid :class:`DFTMonitor` is
    returned, bit-identical to before this option existed.
    """
    lib = _load()
    if lib is None:
        raise RuntimeError(
            "native FDTD kernel unavailable (cargo missing or build failed); "
            "use simulate_3d (XLA) or run scripts/build_kernel.sh"
        )

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    npml = cpml.thickness
    if min(nx, ny, nz) <= 2 * npml + 2:
        raise ValueError(f"grid {nx}x{ny}x{nz} is too small for CPML thickness {npml}")
    dt = grid.dt
    dx, dy, dz = grid.dx, grid.dy, grid.dz

    has_src = source_ijk is not None
    has_port = port_ijk is not None
    if has_src != (source_current is not None):
        raise ValueError("source_ijk and source_current must be given together")
    if has_port != (port_voltage is not None):
        raise ValueError("port_ijk and port_voltage must be given together")

    # Resolve the optional channels (Jx/Jy and Mx/My/Mz) to (idx, waveform).
    def _check_pair(idx, cur, name):
        if (idx is None) != (cur is None):
            raise ValueError(f"{name}_ijk and {name}_current must be given together")
        return idx is not None

    has_jx = _check_pair(source_x_ijk, source_x_current, "source_x")
    has_jy = _check_pair(source_y_ijk, source_y_current, "source_y")
    has_mx = _check_pair(mx_ijk, mx_current, "mx")
    has_my = _check_pair(my_ijk, my_current, "my")
    has_mz = _check_pair(mz_ijk, mz_current, "mz")
    if not (has_src or has_port or has_jx or has_jy or has_mx or has_my or has_mz):
        raise ValueError("at least one current/voltage source is required")

    # n_steps from the first available waveform; validate the rest.
    n_steps = None

    def _wave(cur):
        a = np.asarray(cur)
        return a[:, None] if a.ndim == 1 else a

    if has_src:
        sc = _wave(source_current)
        n_steps = sc.shape[0]
    if has_port:
        pv = np.asarray(port_voltage).reshape(-1)
        n_steps = pv.shape[0] if n_steps is None else n_steps
        if pv.shape[0] != n_steps:
            raise ValueError("source_current and port_voltage lengths differ")
        if port_resistance <= 0.0:
            raise ValueError("port_resistance must be positive")
    for cur, present in (
        (source_x_current, has_jx), (source_y_current, has_jy),
        (mx_current, has_mx), (my_current, has_my), (mz_current, has_mz),
    ):
        if present:
            a = _wave(cur)
            if n_steps is None:
                n_steps = a.shape[0]
            elif a.shape[0] != n_steps:
                raise ValueError("channel waveform length does not match n_steps")

    # dtype inference (np.result_type over operands, like simulate_3d).
    if dtype is None:
        ops = [np.asarray(eps_r).dtype, np.asarray(sigma).dtype]
        for cur, present in (
            (source_current, has_src), (port_voltage, has_port),
            (source_x_current, has_jx), (source_y_current, has_jy),
            (mx_current, has_mx), (my_current, has_my), (mz_current, has_mz),
        ):
            if present:
                ops.append(np.asarray(cur).dtype)
        rt = np.result_type(*ops, np.float32)
        dtype = np.float64 if rt == np.float64 else np.float32
    dtype = np.dtype(dtype)
    if dtype not in (np.float32, np.float64):
        raise ValueError(f"native kernel supports float32/float64, got {dtype}")
    fp = ctypes.c_double if dtype == np.float64 else ctypes.c_float
    _configure3d(lib, dtype)
    run = getattr(lib, "gradenna_3d_run_f64" if dtype == np.float64 else "gradenna_3d_run_f32")

    def arr(x, shape=None):
        a = np.ascontiguousarray(np.asarray(x, dtype))
        if shape is not None:
            a = np.ascontiguousarray(np.broadcast_to(a, shape), dtype)
        return a

    probe_idx = _index_array_3d(probe_ijk, "probe_ijk", grid, margin=0)
    n_probes = probe_idx.shape[0]

    # eps, sigma, then Ca/Cb (exactly as simulate_3d).
    eps = EPS0 * arr(eps_r, (nx, ny, nz))
    sig = arr(sigma, (nx, ny, nz))
    half_loss = sig * dt / (2.0 * eps)
    ca = ((1.0 - half_loss) / (1.0 + half_loss)).astype(dtype)
    cb = ((dt / eps) / (1.0 + half_loss)).astype(dtype)
    ca_ex = np.ascontiguousarray(ca[:-1, 1:-1, 1:-1], dtype)
    cb_ex = np.ascontiguousarray(cb[:-1, 1:-1, 1:-1], dtype)
    ca_ey = np.ascontiguousarray(ca[1:-1, :-1, 1:-1], dtype)
    cb_ey = np.ascontiguousarray(cb[1:-1, :-1, 1:-1], dtype)
    ca_ez = np.ascontiguousarray(ca[1:-1, 1:-1, :-1], dtype)
    cb_ez = np.ascontiguousarray(cb[1:-1, 1:-1, :-1], dtype)

    # CPML axis tables (numpy mirror of axis_coefficients).
    cx_e = _axis_np(nx, dx, dt, cpml, half=False, dtype=dtype)
    cy_e = _axis_np(ny, dy, dt, cpml, half=False, dtype=dtype)
    cz_e = _axis_np(nz, dz, dt, cpml, half=False, dtype=dtype)
    cx_h = _axis_np(nx, dx, dt, cpml, half=True, dtype=dtype)
    cy_h = _axis_np(ny, dy, dt, cpml, half=True, dtype=dtype)
    cz_h = _axis_np(nz, dz, dt, cpml, half=True, dtype=dtype)

    # 1/kappa: E tables interior-sliced (length n-2), H tables half (length n-1).
    ikx_e = np.ascontiguousarray(cx_e.inv_kappa[1:-1], dtype)
    iky_e = np.ascontiguousarray(cy_e.inv_kappa[1:-1], dtype)
    ikz_e = np.ascontiguousarray(cz_e.inv_kappa[1:-1], dtype)
    ikx_h = np.ascontiguousarray(cx_h.inv_kappa, dtype)
    iky_h = np.ascontiguousarray(cy_h.inv_kappa, dtype)
    ikz_h = np.ascontiguousarray(cz_h.inv_kappa, dtype)

    # Slab b/c tables along each stretched axis (length npml each). E-type psi
    # live on the PEC interior (b/c sliced [1:-1] first), H-type on the half
    # grid. Each psi uses its stretched-axis table; see appendix A of note 17.
    def slabs(b, c):
        lo, hi = slab_slices(b.shape[0], npml)
        return (
            np.ascontiguousarray(b[lo], dtype),
            np.ascontiguousarray(c[lo], dtype),
            np.ascontiguousarray(b[hi], dtype),
            np.ascontiguousarray(c[hi], dtype),
        )

    # E-type psi stretched axes: exy->y, exz->z, eyz->z, eyx->x, ezx->x, ezy->y.
    bc = {}
    bc["exy"] = slabs(cy_e.b[1:-1], cy_e.c[1:-1])
    bc["exz"] = slabs(cz_e.b[1:-1], cz_e.c[1:-1])
    bc["eyz"] = slabs(cz_e.b[1:-1], cz_e.c[1:-1])
    bc["eyx"] = slabs(cx_e.b[1:-1], cx_e.c[1:-1])
    bc["ezx"] = slabs(cx_e.b[1:-1], cx_e.c[1:-1])
    bc["ezy"] = slabs(cy_e.b[1:-1], cy_e.c[1:-1])
    # H-type psi stretched axes: hxz->z, hxy->y, hyx->x, hyz->z, hzy->y, hzx->x.
    bc["hxz"] = slabs(cz_h.b, cz_h.c)
    bc["hxy"] = slabs(cy_h.b, cy_h.c)
    bc["hyx"] = slabs(cx_h.b, cx_h.c)
    bc["hyz"] = slabs(cz_h.b, cz_h.c)
    bc["hzy"] = slabs(cy_h.b, cy_h.c)
    bc["hzx"] = slabs(cx_h.b, cx_h.c)

    inv_dx, inv_dy, inv_dz = 1.0 / dx, 1.0 / dy, 1.0 / dz

    # Electric-current sources (component-specific cross-section, note 16).
    def setup_e_channel(idx_in, cur_in, present, area_scale):
        if not present:
            return np.zeros((0, 3), np.int32), np.zeros((n_steps, 0), dtype), np.zeros((0,), dtype)
        idx = np.asarray(idx_in, np.int32).reshape(-1, 3)
        w = arr(cur_in)
        if w.ndim == 1:
            w = w[:, None]
        cbv = (cb[idx[:, 0], idx[:, 1], idx[:, 2]] * area_scale).astype(dtype)
        return idx, np.ascontiguousarray(w, dtype), cbv

    if has_src:
        src_idx = _index_array_3d(source_ijk, "source_ijk", grid, margin=1)
        jz_idx, jz_cur, cb_jz = setup_e_channel(
            src_idx, source_current, True, inv_dx * inv_dy
        )
    else:
        jz_idx, jz_cur, cb_jz = setup_e_channel(None, None, False, 0.0)
    jx_idx, jx_cur, cb_jx = setup_e_channel(
        source_x_ijk, source_x_current, has_jx, inv_dy * inv_dz
    )
    jy_idx, jy_cur, cb_jy = setup_e_channel(
        source_y_ijk, source_y_current, has_jy, inv_dx * inv_dz
    )

    # Magnetic-current sources.
    def setup_m_channel(idx_in, cur_in, present):
        if not present:
            return np.zeros((0, 3), np.int32), np.zeros((n_steps, 0), dtype)
        idx = np.asarray(idx_in, np.int32).reshape(-1, 3)
        w = arr(cur_in)
        if w.ndim == 1:
            w = w[:, None]
        return idx, np.ascontiguousarray(w, dtype)

    mx_idx, mx_cur = setup_m_channel(mx_ijk, mx_current, has_mx)
    my_idx, my_cur = setup_m_channel(my_ijk, my_current, has_my)
    mz_idx, mz_cur = setup_m_channel(mz_ijk, mz_current, has_mz)

    # Lumped RVS port (semi-implicit coefficients, mirroring simulate_3d).
    if has_port:
        (pi, pj, pk), = _index_array_3d(port_ijk, "port_ijk", grid, margin=1)
        eps_p = eps[pi, pj, pk]
        h_p = sig[pi, pj, pk] * dt / (2.0 * eps_p)
        beta = dt * dz / (2.0 * port_resistance * eps_p * dx * dy)
        denom = 1.0 + h_p + beta
        a_port = np.array([(1.0 - h_p - beta) / denom], dtype)
        b_port = np.array([(dt / eps_p) / denom], dtype)
        c_port = np.array([-dt / (port_resistance * eps_p * dx * dy * denom)], dtype)
        port_idx = np.array([[pi, pj, pk]], np.int32)
        port_vs = arr(np.asarray(port_voltage).reshape(-1, 1))
        n_ports = 1
    else:
        a_port = b_port = c_port = np.zeros((0,), dtype)
        port_idx = np.zeros((0, 3), np.int32)
        port_vs = np.zeros((n_steps, 0), dtype)
        n_ports = 0

    # DFT exact-phase tables (float64, identical to simulate_3d).
    dft_freqs = () if dft_freqs is None else tuple(float(f) for f in np.atleast_1d(dft_freqs))
    n_freq = len(dft_freqs)

    # Design-region-limited DFT: each component accumulates only on its slab
    # (or not at all when its slab is None). The full-grid shapes below become
    # per-component slab shapes, and a `_DFTRegion3D` struct is handed to the
    # kernel (NULL == the full-grid path). The slab bounds and freq-major /
    # cell-major layout match the XLA region path in `simulate_3d`.
    has_regions = n_freq and dft_regions is not None
    full_shapes = (
        (nx - 1, ny, nz), (nx, ny - 1, nz), (nx, ny, nz - 1),
        (nx, ny - 1, nz - 1), (nx - 1, ny, nz - 1), (nx - 1, ny - 1, nz),
    )
    if has_regions:
        if len(dft_regions) != 6:
            raise ValueError("dft_regions must have six per-component entries")
        slabs = tuple(dft_regions)  # FieldSlab | None per component
        dft_shapes = tuple(
            None if slab is None else tuple(slab.shape)
            for slab in slabs
        )
    else:
        slabs = (None,) * 6
        dft_shapes = full_shapes
    if n_freq:
        f_np = np.asarray(dft_freqs, np.float64)
        n_np = np.arange(n_steps, dtype=np.float64)
        ph_e = np.exp(-2j * np.pi * np.outer(n_np + 1.0, f_np) * dt)  # (n_steps, n_freq)
        ph_h = np.exp(-2j * np.pi * np.outer(n_np + 0.5, f_np) * dt)
        ph_e_re = np.ascontiguousarray(ph_e.real, np.float64)
        ph_e_im = np.ascontiguousarray(ph_e.imag, np.float64)
        ph_h_re = np.ascontiguousarray(ph_h.real, np.float64)
        ph_h_im = np.ascontiguousarray(ph_h.imag, np.float64)
    else:
        ph_e_re = ph_e_im = ph_h_re = ph_h_im = np.zeros((0, 0), np.float64)

    # Field / output buffers (kernel writes; caller-zeroed).
    ex = np.zeros((nx - 1, ny, nz), dtype)
    ey = np.zeros((nx, ny - 1, nz), dtype)
    ez = np.zeros((nx, ny, nz - 1), dtype)
    hx = np.zeros((nx, ny - 1, nz - 1), dtype)
    hy = np.zeros((nx - 1, ny, nz - 1), dtype)
    hz = np.zeros((nx - 1, ny - 1, nz), dtype)
    out_probe = np.zeros((n_steps, n_probes), dtype)
    out_v = np.zeros((n_steps, n_ports), dtype)
    out_i = np.zeros((n_steps, n_ports), dtype)
    out_energy = np.zeros((n_steps,), dtype) if record_energy else np.zeros((0,), dtype)

    def _dft(shape):
        # shape is None for an off component (zero-sized buffer, kernel skips
        # it) or a per-component (full-grid or slab) shape.
        if not n_freq or shape is None:
            a = np.zeros((0,), np.float64)
        else:
            a = np.zeros((n_freq,) + tuple(shape), np.float64)
        return a, np.zeros_like(a)

    dft_ex_re, dft_ex_im = _dft(dft_shapes[0])
    dft_ey_re, dft_ey_im = _dft(dft_shapes[1])
    dft_ez_re, dft_ez_im = _dft(dft_shapes[2])
    dft_hx_re, dft_hx_im = _dft(dft_shapes[3])
    dft_hy_re, dft_hy_im = _dft(dft_shapes[4])
    dft_hz_re, dft_hz_im = _dft(dft_shapes[5])

    # --- assemble the structs ------------------------------------------------
    # Keep references to every numpy array alive for the duration of the call
    # (ctypes pointers do not own the buffers).
    keep = []

    def cvp(a):
        keep.append(a)
        return ctypes.cast(a.ctypes.data, ctypes.c_void_p)

    coeff = _CoeffTables(
        cvp(ca_ex), cvp(cb_ex), cvp(ca_ey), cvp(cb_ey), cvp(ca_ez), cvp(cb_ez),
        cvp(ikx_e), cvp(iky_e), cvp(ikz_e), cvp(ikx_h), cvp(iky_h), cvp(ikz_h),
        cvp(np.ascontiguousarray(eps, dtype)),
    )

    cpml_args = []
    for name in _PSI_NAMES:
        b_lo, c_lo, b_hi, c_hi = bc[name]
        cpml_args += [cvp(b_lo), cvp(c_lo), cvp(b_hi), cvp(c_hi)]
    cpml_struct = _CpmlSlabs(*cpml_args)

    def ci(a):  # int32 column -> c_void_p
        a = np.ascontiguousarray(a, np.int32)
        keep.append(a)
        return ctypes.cast(a.ctypes.data, ctypes.c_void_p)

    srcs = _SourceBuffers(
        ci(jx_idx[:, 0]), ci(jx_idx[:, 1]), ci(jx_idx[:, 2]), cvp(cb_jx), cvp(jx_cur),
        ci(jy_idx[:, 0]), ci(jy_idx[:, 1]), ci(jy_idx[:, 2]), cvp(cb_jy), cvp(jy_cur),
        ci(jz_idx[:, 0]), ci(jz_idx[:, 1]), ci(jz_idx[:, 2]), cvp(cb_jz), cvp(jz_cur),
        ci(mx_idx[:, 0]), ci(mx_idx[:, 1]), ci(mx_idx[:, 2]), cvp(mx_cur),
        ci(my_idx[:, 0]), ci(my_idx[:, 1]), ci(my_idx[:, 2]), cvp(my_cur),
        ci(mz_idx[:, 0]), ci(mz_idx[:, 1]), ci(mz_idx[:, 2]), cvp(mz_cur),
        ci(port_idx[:, 0]), ci(port_idx[:, 1]), ci(port_idx[:, 2]),
        cvp(a_port), cvp(b_port), cvp(c_port), cvp(port_vs),
        ci(probe_idx[:, 0]), ci(probe_idx[:, 1]), ci(probe_idx[:, 2]),
        _dptr_cvp(ph_e_re, keep), _dptr_cvp(ph_e_im, keep),
        _dptr_cvp(ph_h_re, keep), _dptr_cvp(ph_h_im, keep),
    )

    fb = _FieldBuffers(
        cvp(ex), cvp(ey), cvp(ez), cvp(hx), cvp(hy), cvp(hz),
        cvp(out_probe), cvp(out_v), cvp(out_i), cvp(out_energy),
        _dptr_cvp(dft_ex_re, keep), _dptr_cvp(dft_ex_im, keep),
        _dptr_cvp(dft_ey_re, keep), _dptr_cvp(dft_ey_im, keep),
        _dptr_cvp(dft_ez_re, keep), _dptr_cvp(dft_ez_im, keep),
        _dptr_cvp(dft_hx_re, keep), _dptr_cvp(dft_hx_im, keep),
        _dptr_cvp(dft_hy_re, keep), _dptr_cvp(dft_hy_im, keep),
        _dptr_cvp(dft_hz_re, keep), _dptr_cvp(dft_hz_im, keep),
    )

    params = _SimParams3D(
        nx, ny, nz, npml, n_steps,
        jx_idx.shape[0], jy_idx.shape[0], jz_idx.shape[0], n_ports,
        mx_idx.shape[0], my_idx.shape[0], mz_idx.shape[0],
        n_probes, n_freq, 1 if record_energy else 0,
    )

    # Region pointer: NULL for the full-grid DFT, else a `_DFTRegion3D`.
    region_ptr = None
    if has_regions:
        def _slab(slab):
            if slab is None:
                return _DFTSlab3D(0, 0, 0, 0, 0, 0, 0)
            (lo0, lo1, lo2) = (int(v) for v in slab.lo)
            (hi0, hi1, hi2) = (int(v) for v in slab.hi)
            return _DFTSlab3D(1, lo0, hi0, lo1, hi1, lo2, hi2)

        region_struct = _DFTRegion3D(*(_slab(s) for s in slabs))
        region_ptr = ctypes.byref(region_struct)

    run(
        ctypes.byref(params), ctypes.byref(coeff), ctypes.byref(cpml_struct),
        ctypes.byref(srcs), ctypes.byref(fb),
        fp(dt / MU0), fp(inv_dx), fp(inv_dy), fp(inv_dz),
        fp(dx), fp(dy), fp(dz), fp(MU0),
        region_ptr,
    )
    del keep  # release the buffer references after the call returns

    dft = None
    if n_freq and not has_regions:
        dft = DFTMonitor(
            np.asarray(dft_freqs, np.float64),
            (dft_ex_re + 1j * dft_ex_im) * dt,
            (dft_ey_re + 1j * dft_ey_im) * dt,
            (dft_ez_re + 1j * dft_ez_im) * dt,
            (dft_hx_re + 1j * dft_hx_im) * dt,
            (dft_hy_re + 1j * dft_hy_im) * dt,
            (dft_hz_re + 1j * dft_hz_im) * dt,
        )
    elif has_regions:
        # One complex128 slab spectrum per accumulated component; None for an
        # off component (slab is None) -- mirrors the XLA RegionDFTMonitor.
        from gradenna.dft_region import RegionDFTMonitor

        re_im = (
            (dft_ex_re, dft_ex_im), (dft_ey_re, dft_ey_im), (dft_ez_re, dft_ez_im),
            (dft_hx_re, dft_hx_im), (dft_hy_re, dft_hy_im), (dft_hz_re, dft_hz_im),
        )
        comps = tuple(
            None if slab is None else (re + 1j * im) * dt
            for slab, (re, im) in zip(slabs, re_im)
        )
        dft = RegionDFTMonitor(
            np.asarray(dft_freqs, np.float64), dft_regions, *comps
        )

    # simulate_3d supports a single port and returns 1D (n_steps,) V/I series.
    return SimResult3D(
        probe_ez=out_probe,
        port_v=(out_v[:, 0] if n_ports else None),
        port_i=(out_i[:, 0] if n_ports else None),
        energy=(out_energy if record_energy else None),
        dft=dft,
        ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz,
    )


def _dptr_cvp(a, keep):
    """A c_void_p to a float64 array, keeping it alive in `keep`."""
    a = np.ascontiguousarray(a, np.float64)
    keep.append(a)
    return ctypes.cast(a.ctypes.data, ctypes.c_void_p)
