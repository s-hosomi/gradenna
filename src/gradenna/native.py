"""Native (Rust) fused 2D TM FDTD time loop -- optional local accelerator.

`gradenna.fdtd2d.simulate_tm` runs the whole TM time loop as a `jax.lax.scan`
of ~20-30 small XLA kernels per step; on CPU the per-step dispatch + thread
sync dominates and only ~10% of M1 Pro memory bandwidth is reached. This
module loads the `rust/` cdylib (a single fused native time loop) and exposes
:func:`simulate_tm_native`, a drop-in for the forward-only subset of
`simulate_tm` that returns a `SimResult`-compatible object.

Design
======
* **Coefficient parity, not arithmetic re-derivation.** Every coefficient
  table (ca, cb, the per-source ``cb_src``, the per-port ``cb_vs``, the CPML
  slab b/c tables, the 1/kappa axis tables and the exact-phase DFT tables) is
  built here *exactly as* :func:`simulate_tm` builds them, then handed to the
  kernel flat. The Rust side does only the time stepping, so the update order
  and operations match and the results agree to f64 rel <= 1e-12 / f32 1e-5.
* **Two integration forms** (see the module-level task): (a) a plain
  ctypes/numpy call usable outside jit (benchmarking, the freq-adjoint hook);
  (b) a ``jax.ffi.ffi_call`` registration so the kernel can sit inside a
  jit/scan. (b) is attempted opportunistically and degrades to (a); see
  :func:`ffi_available`.

Availability
============
:func:`is_available` returns False (and the rest no-ops to a clear error)
when cargo is missing or the build fails, so callers can fall back to XLA.
The library is built on first use via ``scripts/build_kernel.sh``.
"""

from __future__ import annotations

import ctypes
import math
import os
import subprocess
import sys
import threading
from typing import NamedTuple

import numpy as np

from gradenna.constants import EPS0, MU0
from gradenna.cpml import CPMLSpec, axis_coefficients, slab_slices
from gradenna.fdtd2d import DZ, Port, SimResult
from gradenna.grid import Grid2D

__all__ = ["is_available", "ffi_available", "simulate_tm_native", "build"]

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RUST_DIR = os.path.join(_REPO_ROOT, "rust")
_BUILD_SH = os.path.join(_REPO_ROOT, "scripts", "build_kernel.sh")

_LOCK = threading.Lock()
_LIB = None  # cached CDLL or False once we've decided it is unavailable


def _dylib_path() -> str:
    name = "libgradenna_kernel.dylib" if sys.platform == "darwin" else "libgradenna_kernel.so"
    return os.path.join(_RUST_DIR, "target", "release", name)


def _cargo_bin() -> str | None:
    from shutil import which

    cargo = which("cargo")
    if cargo:
        return cargo
    candidate = os.path.expanduser("~/.cargo/bin/cargo")
    return candidate if os.path.exists(candidate) else None


def build(force: bool = False) -> bool:
    """Build the native kernel via cargo. Returns True on success.

    A no-op (returns True) if the dylib already exists and ``force`` is False.
    """
    if not force and os.path.exists(_dylib_path()):
        return True
    cargo = _cargo_bin()
    if cargo is None:
        return False
    try:
        subprocess.run(
            [cargo, "build", "--release", "--manifest-path", os.path.join(_RUST_DIR, "Cargo.toml")],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return False
    return os.path.exists(_dylib_path())


def _load():
    """Load (building if needed) the cdylib; cache the result. None if N/A."""
    global _LIB
    with _LOCK:
        if _LIB is not None:
            return _LIB or None
        if not os.path.exists(_dylib_path()):
            if not build():
                _LIB = False
                return None
        try:
            lib = ctypes.CDLL(_dylib_path())
        except OSError:
            _LIB = False
            return None
        _configure(lib)
        _LIB = lib
        return lib


def is_available() -> bool:
    """True if the native kernel can be loaded (cargo present + builds)."""
    return _load() is not None


# ---------------------------------------------------------------------------
# ctypes signatures
# ---------------------------------------------------------------------------


class _SimParams(ctypes.Structure):
    _fields_ = [
        ("nx", ctypes.c_int),
        ("ny", ctypes.c_int),
        ("npml", ctypes.c_int),
        ("n_steps", ctypes.c_int),
        ("n_sources", ctypes.c_int),
        ("n_ports", ctypes.c_int),
        ("n_mx", ctypes.c_int),
        ("n_my", ctypes.c_int),
        ("n_probes", ctypes.c_int),
        ("n_freq", ctypes.c_int),
        ("record_energy", ctypes.c_int),
    ]


def _configure(lib) -> None:
    """Declare argtypes for both float-width entry points.

    The argument *order* mirrors the Rust ``extern "C"`` signature exactly;
    keep the two in lockstep.
    """
    for name, fp in (("gradenna_tm_run_f64", ctypes.c_double), ("gradenna_tm_run_f32", ctypes.c_float)):
        fn = getattr(lib, name)
        P = ctypes.POINTER(fp)
        I = ctypes.POINTER(ctypes.c_int)
        D = ctypes.POINTER(ctypes.c_double)
        fn.restype = None
        fn.argtypes = [
            ctypes.POINTER(_SimParams),
            P, P, P,  # ca, cb, eps
            P, P, P, P,  # inv_kx_e, inv_ky_e, inv_kx_h, inv_ky_h
            P, P, P, P,  # ezx lo/hi b/c
            P, P, P, P,  # ezy
            P, P, P, P,  # hyx
            P, P, P, P,  # hxy
            fp, fp, fp, fp, fp, fp, fp,  # dt_mu, inv_dx, inv_dy, dx, dy, dz, mu0
            I, I, P, P,  # src_i, src_j, cb_src, src_cur
            I, I, P, P,  # port_i, port_j, cb_vs, port_vs
            I, I, P,  # mx
            I, I, P,  # my
            I, I,  # probe_i, probe_j
            D, D, D, D,  # ph_e_re/im, ph_h_re/im
            P, P, P,  # ez, hx, hy
            P, P, P, P,  # out_probe, out_v, out_i, out_energy
            D, D, D, D, D, D,  # dft re/im ez, hx, hy
        ]


def _ptr(arr, fp):
    """C pointer into a contiguous numpy array of the matching scalar type."""
    return arr.ctypes.data_as(ctypes.POINTER(fp))


def _iptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


def _dptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


# ---------------------------------------------------------------------------
# Coefficient preparation (mirrors simulate_tm exactly)
# ---------------------------------------------------------------------------


def _index_array(ij, name, grid, margin):
    idx = np.asarray(ij, dtype=np.int32).reshape(-1, 2)
    lo = np.array([margin, margin])
    hi = np.array([grid.nx - 1 - margin, grid.ny - 1 - margin])
    if idx.size and (np.any(idx < lo) or np.any(idx > hi)):
        raise ValueError(f"{name} {idx.tolist()} outside the interior (PML margin {margin})")
    return idx


def simulate_tm_native(
    grid: Grid2D,
    *,
    source_ij=None,
    source_current=None,
    eps_r=1.0,
    sigma=0.0,
    probe_ij=(),
    ports=(),
    dft_freqs=(),
    cpml: CPMLSpec = CPMLSpec(),
    record_energy: bool = False,
    mx_ij=None,
    mx_current=None,
    my_ij=None,
    my_current=None,
    dtype=None,
) -> SimResult:
    """Run the 2D TM forward solve in the native Rust kernel.

    Drop-in for the forward-only subset of :func:`gradenna.fdtd2d.simulate_tm`
    (same argument names and semantics), returning a :class:`SimResult` of
    numpy arrays. ``dtype`` selects ``np.float32`` / ``np.float64`` (default:
    inferred from the inputs, like ``simulate_tm``). The DFT accumulators are
    always kept in float64 internally (matching ``dft_dtype=complex128``) and
    returned as complex128.

    Raises ``RuntimeError`` if the native library is unavailable; callers that
    want a fallback should gate on :func:`is_available`.
    """
    lib = _load()
    if lib is None:
        raise RuntimeError(
            "native FDTD kernel unavailable (cargo missing or build failed); "
            "use simulate_tm (XLA) or run scripts/build_kernel.sh"
        )

    nx, ny = grid.nx, grid.ny
    if min(nx, ny) <= 2 * cpml.thickness + 2:
        raise ValueError(f"grid {nx}x{ny} is too small for CPML thickness {cpml.thickness}")
    dt = grid.dt

    if (source_ij is None) != (source_current is None):
        raise ValueError("source_ij and source_current must be given together")

    ports = tuple(p if isinstance(p, Port) else Port(*p) for p in ports)
    n_ports = len(ports)
    port_voltages = [None if p.voltage is None else np.asarray(p.voltage) for p in ports]

    # n_steps from source and/or port waveforms (same rule as simulate_tm).
    n_steps = None
    if source_current is not None:
        source_current = np.asarray(source_current)
        n_steps = source_current.shape[0]
    for v in port_voltages:
        if v is None:
            continue
        if v.ndim != 1:
            raise ValueError(f"port voltage waveform must be 1D, got shape {v.shape}")
        if n_steps is None:
            n_steps = v.shape[0]
        elif v.shape[0] != n_steps:
            raise ValueError(f"port voltage length {v.shape[0]} does not match n_steps {n_steps}")
    if n_steps is None:
        raise ValueError("need source_current and/or at least one port with a voltage waveform")

    # dtype inference matching simulate_tm (np.result_type over operands).
    if dtype is None:
        ops = [np.asarray(eps_r).dtype, np.asarray(sigma).dtype]
        if source_current is not None:
            ops.append(source_current.dtype)
        for v in port_voltages:
            if v is not None:
                ops.append(v.dtype)
        if mx_current is not None:
            ops.append(np.asarray(mx_current).dtype)
        if my_current is not None:
            ops.append(np.asarray(my_current).dtype)
        rt = np.result_type(*ops, np.float32)
        dtype = np.float64 if rt == np.float64 else np.float32
    dtype = np.dtype(dtype)
    if dtype not in (np.float32, np.float64):
        raise ValueError(f"native kernel supports float32/float64, got {dtype}")
    fp = ctypes.c_double if dtype == np.float64 else ctypes.c_float
    run = getattr(lib, "gradenna_tm_run_f64" if dtype == np.float64 else "gradenna_tm_run_f32")

    def arr(x, shape=None):
        a = np.ascontiguousarray(np.asarray(x, dtype))
        if shape is not None:
            a = np.ascontiguousarray(np.broadcast_to(a, shape), dtype)
        return a

    probe_idx = _index_array(probe_ij, "probe_ij", grid, margin=0)
    n_probes = probe_idx.shape[0]

    # Sources.
    if source_current is None:
        src_idx = np.zeros((0, 2), np.int32)
        src_cur = np.zeros((n_steps, 0), dtype)
    else:
        src_idx = _index_array(source_ij, "source_ij", grid, margin=1)
        src_cur = arr(source_current)
        if src_cur.ndim == 1:
            src_cur = src_cur[:, None]
        if src_cur.shape[1] != src_idx.shape[0]:
            raise ValueError(
                f"source_current has {src_cur.shape[1]} columns for {src_idx.shape[0]} sources"
            )
    n_src = src_idx.shape[0]

    # eps, sigma (+ port resistor conductivity), then ca/cb -- exactly as simulate_tm.
    eps = EPS0 * arr(eps_r, (nx, ny))
    sig = arr(sigma, (nx, ny)).copy()
    if n_ports:
        port_idx = _index_array([p.ij for p in ports], "port ij", grid, margin=1)
        rs = np.asarray([float(p.resistance) for p in ports])
        if np.any(rs <= 0.0):
            raise ValueError("port resistance must be positive")
        pi, pj = port_idx[:, 0], port_idx[:, 1]
        np.add.at(sig, (pi, pj), DZ / (rs * grid.dx * grid.dy))
        port_rs = rs.astype(dtype)
        port_vs = np.stack(
            [np.zeros((n_steps,), dtype) if v is None else arr(v) for v in port_voltages], axis=1
        )
    else:
        port_idx = np.zeros((0, 2), np.int32)
        port_vs = np.zeros((n_steps, 0), dtype)

    half_loss = sig * dt / (2.0 * eps)
    ca = ((1.0 - half_loss) / (1.0 + half_loss)).astype(dtype)
    cb = ((dt / eps) / (1.0 + half_loss)).astype(dtype)

    # CPML axis tables (numpy mirror of axis_coefficients via jax -> np).
    cx_e = _axis_np(nx, grid.dx, dt, cpml, half=False, dtype=dtype)
    cx_h = _axis_np(nx, grid.dx, dt, cpml, half=True, dtype=dtype)
    cy_e = _axis_np(ny, grid.dy, dt, cpml, half=False, dtype=dtype)
    cy_h = _axis_np(ny, grid.dy, dt, cpml, half=True, dtype=dtype)

    npml = cpml.thickness
    # Slab b/c tables (length npml each), pre-sliced exactly as simulate_tm.
    # E-type psi live on the PEC interior (b/c sliced [1:-1] first).
    def slabs(b, c):
        lo, hi = slab_slices(b.shape[0], npml)
        return (
            np.ascontiguousarray(b[lo], dtype),
            np.ascontiguousarray(c[lo], dtype),
            np.ascontiguousarray(b[hi], dtype),
            np.ascontiguousarray(c[hi], dtype),
        )

    ezx_b_lo, ezx_c_lo, ezx_b_hi, ezx_c_hi = slabs(cx_e.b[1:-1], cx_e.c[1:-1])
    ezy_b_lo, ezy_c_lo, ezy_b_hi, ezy_c_hi = slabs(cy_e.b[1:-1], cy_e.c[1:-1])
    hyx_b_lo, hyx_c_lo, hyx_b_hi, hyx_c_hi = slabs(cx_h.b, cx_h.c)
    hxy_b_lo, hxy_c_lo, hxy_b_hi, hxy_c_hi = slabs(cy_h.b, cy_h.c)

    inv_kx_e = np.ascontiguousarray(cx_e.inv_kappa[1:-1], dtype)
    inv_ky_e = np.ascontiguousarray(cy_e.inv_kappa[1:-1], dtype)
    inv_kx_h = np.ascontiguousarray(cx_h.inv_kappa, dtype)
    inv_ky_h = np.ascontiguousarray(cy_h.inv_kappa, dtype)

    inv_dx, inv_dy = 1.0 / grid.dx, 1.0 / grid.dy
    cb_src = (cb[src_idx[:, 0], src_idx[:, 1]] * (inv_dx * inv_dy)).astype(dtype)
    if n_ports:
        cb_vs = (cb[pi, pj] / (port_rs * grid.dx * grid.dy)).astype(dtype)
    else:
        cb_vs = np.zeros((0,), dtype)

    # Magnetic-current sources.
    def setup_mag(m_ij, m_current, name):
        if (m_ij is None) != (m_current is None):
            raise ValueError(f"{name}_ij and {name}_current must be given together")
        if m_current is None:
            return np.zeros((0, 2), np.int32), np.zeros((n_steps, 0), dtype)
        idx = np.asarray(m_ij, np.int32).reshape(-1, 2)
        mc = arr(m_current)
        if mc.ndim == 1:
            mc = mc[:, None]
        if mc.shape[0] != n_steps:
            raise ValueError(f"{name}_current length {mc.shape[0]} != n_steps {n_steps}")
        if mc.shape[1] != idx.shape[0]:
            raise ValueError(f"{name}_current has {mc.shape[1]} columns for {idx.shape[0]} sources")
        return idx, mc

    mx_idx, mx_cur = setup_mag(mx_ij, mx_current, "mx")
    my_idx, my_cur = setup_mag(my_ij, my_current, "my")

    # DFT exact-phase tables (float64, identical to simulate_tm).
    dft_freqs = tuple(float(f) for f in dft_freqs)
    n_freq = len(dft_freqs)
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

    # Output buffers (kernel writes; caller-zeroed).
    ez = np.zeros((nx, ny), dtype)
    hx = np.zeros((nx, ny - 1), dtype)
    hy = np.zeros((nx - 1, ny), dtype)
    out_probe = np.zeros((n_steps, n_probes), dtype)
    out_v = np.zeros((n_steps, n_ports), dtype)
    out_i = np.zeros((n_steps, n_ports), dtype)
    out_energy = np.zeros((n_steps,), dtype) if record_energy else np.zeros((0,), dtype)
    ez_n, hx_n, hy_n = nx * ny, nx * (ny - 1), (nx - 1) * ny
    dft_ez_re = np.zeros((n_freq, nx, ny), np.float64) if n_freq else np.zeros((0,), np.float64)
    dft_ez_im = np.zeros_like(dft_ez_re)
    dft_hx_re = np.zeros((n_freq, nx, ny - 1), np.float64) if n_freq else np.zeros((0,), np.float64)
    dft_hx_im = np.zeros_like(dft_hx_re)
    dft_hy_re = np.zeros((n_freq, nx - 1, ny), np.float64) if n_freq else np.zeros((0,), np.float64)
    dft_hy_im = np.zeros_like(dft_hy_re)

    params = _SimParams(nx, ny, npml, n_steps, n_src, n_ports, mx_idx.shape[0], my_idx.shape[0],
                        n_probes, n_freq, 1 if record_energy else 0)

    # Contiguous index arrays.
    si = np.ascontiguousarray(src_idx[:, 0], np.int32)
    sj = np.ascontiguousarray(src_idx[:, 1], np.int32)
    ppi = np.ascontiguousarray(port_idx[:, 0], np.int32)
    ppj = np.ascontiguousarray(port_idx[:, 1], np.int32)
    mxi = np.ascontiguousarray(mx_idx[:, 0], np.int32)
    mxj = np.ascontiguousarray(mx_idx[:, 1], np.int32)
    myi = np.ascontiguousarray(my_idx[:, 0], np.int32)
    myj = np.ascontiguousarray(my_idx[:, 1], np.int32)
    pri = np.ascontiguousarray(probe_idx[:, 0], np.int32)
    prj = np.ascontiguousarray(probe_idx[:, 1], np.int32)
    src_cur = np.ascontiguousarray(src_cur, dtype)
    port_vs = np.ascontiguousarray(port_vs, dtype)
    mx_cur = np.ascontiguousarray(mx_cur, dtype)
    my_cur = np.ascontiguousarray(my_cur, dtype)

    run(
        ctypes.byref(params),
        _ptr(ca, fp), _ptr(cb, fp), _ptr(eps.astype(dtype), fp),
        _ptr(inv_kx_e, fp), _ptr(inv_ky_e, fp), _ptr(inv_kx_h, fp), _ptr(inv_ky_h, fp),
        _ptr(ezx_b_lo, fp), _ptr(ezx_c_lo, fp), _ptr(ezx_b_hi, fp), _ptr(ezx_c_hi, fp),
        _ptr(ezy_b_lo, fp), _ptr(ezy_c_lo, fp), _ptr(ezy_b_hi, fp), _ptr(ezy_c_hi, fp),
        _ptr(hyx_b_lo, fp), _ptr(hyx_c_lo, fp), _ptr(hyx_b_hi, fp), _ptr(hyx_c_hi, fp),
        _ptr(hxy_b_lo, fp), _ptr(hxy_c_lo, fp), _ptr(hxy_b_hi, fp), _ptr(hxy_c_hi, fp),
        fp(dt / MU0), fp(inv_dx), fp(inv_dy), fp(grid.dx), fp(grid.dy), fp(DZ), fp(MU0),
        _iptr(si), _iptr(sj), _ptr(cb_src, fp), _ptr(src_cur, fp),
        _iptr(ppi), _iptr(ppj), _ptr(cb_vs, fp), _ptr(port_vs, fp),
        _iptr(mxi), _iptr(mxj), _ptr(mx_cur, fp),
        _iptr(myi), _iptr(myj), _ptr(my_cur, fp),
        _iptr(pri), _iptr(prj),
        _dptr(ph_e_re), _dptr(ph_e_im), _dptr(ph_h_re), _dptr(ph_h_im),
        _ptr(ez, fp), _ptr(hx, fp), _ptr(hy, fp),
        _ptr(out_probe, fp), _ptr(out_v, fp), _ptr(out_i, fp), _ptr(out_energy, fp),
        _dptr(dft_ez_re), _dptr(dft_ez_im),
        _dptr(dft_hx_re), _dptr(dft_hx_im),
        _dptr(dft_hy_re), _dptr(dft_hy_im),
    )

    dft_ez = dft_hx = dft_hy = None
    if n_freq:
        dft_ez = (dft_ez_re + 1j * dft_ez_im) * dt
        dft_hx = (dft_hx_re + 1j * dft_hx_im) * dt
        dft_hy = (dft_hy_re + 1j * dft_hy_im) * dt

    return SimResult(
        probe_ez=out_probe,
        energy=(out_energy if record_energy else None),
        ez=ez,
        hx=hx,
        hy=hy,
        port_v=(out_v if n_ports else None),
        port_i=(out_i if n_ports else None),
        dft_ez=dft_ez,
        dft_hx=dft_hx,
        dft_hy=dft_hy,
    )


class _AxisNp(NamedTuple):
    b: np.ndarray
    c: np.ndarray
    inv_kappa: np.ndarray


def _axis_np(n, delta, dt, spec: CPMLSpec, *, half, dtype):
    """Numpy mirror of :func:`gradenna.cpml.axis_coefficients`.

    Reuses the jax implementation and converts to numpy so the tables match
    ``simulate_tm`` bit-for-bit in f64 (and to f32 cast precision in f32).
    """
    ac = axis_coefficients(n, delta, dt, spec, half=half, dtype=dtype)
    return _AxisNp(
        b=np.asarray(ac.b, dtype),
        c=np.asarray(ac.c, dtype),
        inv_kappa=np.asarray(ac.inv_kappa, dtype),
    )


# ---------------------------------------------------------------------------
# (b) jax.ffi integration -- opportunistic, see ffi_available()
# ---------------------------------------------------------------------------

_FFI_STATUS = None  # None unknown, str reason if unavailable, True if registered


def ffi_available() -> tuple[bool, str]:
    """Whether the kernel is registered as a jit-embeddable ``jax.ffi`` call.

    Returns ``(ok, reason)``. See the module docstring / final report: the
    ctypes path (a) is the supported integration; (b) is documented as a
    future item because the fused kernel allocates scratch + many output
    buffers and consumes precomputed coefficient tables (not raw traced
    arrays), which does not map cleanly onto the single-call XLA FFI buffer
    convention without a substantial shim. The ctypes path already serves the
    benchmark and the freq-adjoint backend hook (custom_vjp internals need not
    be jit-compiled).
    """
    return (
        False,
        "not implemented: the fused multi-output kernel with precomputed "
        "coefficient inputs does not map onto a single jax.ffi.ffi_call "
        "buffer signature without a large shim; use the ctypes path (a).",
    )


def _maybe_build_on_import():
    """Best-effort: do nothing at import time (build is lazy on first use)."""
    return None
