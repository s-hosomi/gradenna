//! Fused 3D Yee FDTD time-loop kernel for gradenna (forward-only).
//!
//! This module executes the *entire* `n_steps` time loop of
//! `gradenna.fdtd3d.simulate_3d` in a single native call, the 3D analogue of
//! the 2D kernel in `lib.rs`. The same two ideas make it fast:
//!
//!   1. **True fusion.** Each time step is two field sweeps -- an H sweep
//!      (Hx, Hy, Hz + their CPML psi + magnetic-current injection) and an E
//!      sweep (Ex, Ey, Ez interior + their CPML psi + Ca/Cb + electric-current
//!      / port injection + probe / DFT accumulation). The six curls, the
//!      `1/kappa` scalings and the `term_a - term_b` differences are computed
//!      on the fly in `k`-inner (z, stride-1, auto-vectorizable) loops; no
//!      full-grid scratch buffers are written and re-read.
//!
//!   2. **Resident worker pool + spin barrier.** Workers are spawned once,
//!      outside the time loop, each statically assigned a contiguous block of
//!      x rows (a slab), and synchronize with a `SpinBarrier` only where a
//!      step has a true data dependence: once after the H sweep (the E sweep
//!      reads neighbouring H x-rows) and once after the E sweep + injection
//!      (closing the step; once more after a DFT pass when n_freq > 0).
//!
//! Numerical contract: the arithmetic mirrors `simulate_3d` operation by
//! operation (same update order, same `t_a - t_b` curls, same semi-implicit
//! port update, same f64 DFT phase tables) so results agree to f64 rel
//! <= 1e-12 and f32 rel <= 1e-5. All coefficient tables are built on the
//! Python side (`gradenna.native3d`) and passed in flat; the kernel does no
//! coefficient math.
//!
//! Memory layout: every 3D array is C order (k / z contiguous). Field shapes
//! (note the staggering):
//!     Ex (nx-1, ny,   nz  )   Ey (nx,   ny-1, nz  )   Ez (nx,   ny,   nz-1)
//!     Hx (nx,   ny-1, nz-1)   Hy (nx-1, ny,   nz-1)   Hz (nx-1, ny-1, nz  )

use std::os::raw::c_int;
use std::slice;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Spinning sense-reversal barrier (see `lib.rs` for the rationale).
struct SpinBarrier {
    count: AtomicUsize,
    sense: AtomicUsize,
    n: usize,
}
impl SpinBarrier {
    fn new(n: usize) -> Self {
        SpinBarrier { count: AtomicUsize::new(0), sense: AtomicUsize::new(0), n }
    }
    #[inline]
    fn wait(&self, local: &mut usize) {
        let my = *local ^ 1;
        *local = my;
        if self.count.fetch_add(1, Ordering::AcqRel) + 1 == self.n {
            self.count.store(0, Ordering::Relaxed);
            self.sense.store(my, Ordering::Release);
        } else {
            let mut spins = 0u32;
            while self.sense.load(Ordering::Acquire) != my {
                spins += 1;
                if spins < 1 << 12 {
                    std::hint::spin_loop();
                } else {
                    std::thread::yield_now();
                }
            }
        }
    }
}

/// A raw `*mut T` shared across the worker pool. Safe because workers touch
/// disjoint x-row blocks within a barrier phase and the few cross-block reads
/// happen after a barrier orders them (same pattern as the 2D kernel).
#[derive(Clone, Copy)]
struct Shared<T>(*mut T);
unsafe impl<T> Send for Shared<T> {}
unsafe impl<T> Sync for Shared<T> {}
impl<T> Shared<T> {
    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    unsafe fn m(&self, len: usize) -> &'static mut [T] {
        slice::from_raw_parts_mut(self.0, len)
    }
}

/// Plain old data describing one 3D simulation, shared by both float widths.
#[repr(C)]
pub struct SimParams3D {
    nx: c_int,
    ny: c_int,
    nz: c_int,
    npml: c_int,
    n_steps: c_int,
    n_jx: c_int,
    n_jy: c_int,
    n_jz: c_int,
    n_ports: c_int,
    n_mx: c_int,
    n_my: c_int,
    n_mz: c_int,
    n_probes: c_int,
    n_freq: c_int,
    record_energy: c_int,
}

/// Coefficient tables (read-only), pointers into Python-owned numpy arrays.
#[repr(C)]
pub struct CoeffTables<S> {
    // Ca/Cb at the three E-component interior slices (see field shapes below).
    ca_ex: *const S, // (nx-1, ny-2, nz-2)
    cb_ex: *const S,
    ca_ey: *const S, // (nx-2, ny-1, nz-2)
    cb_ey: *const S,
    ca_ez: *const S, // (nx-2, ny-2, nz-1)
    cb_ez: *const S,
    // 1/kappa axis tables. E tables are interior-sliced (length n-2);
    // H tables are half-grid (length n-1).
    ikx_e: *const S, // (nx-2,)
    iky_e: *const S, // (ny-2,)
    ikz_e: *const S, // (nz-2,)
    ikx_h: *const S, // (nx-1,)
    iky_h: *const S, // (ny-1,)
    ikz_h: *const S, // (nz-1,)
    eps: *const S, // (nx, ny, nz) absolute permittivity (energy diagnostic only)
}

/// CPML slab b/c tables. Each stretched axis has a low/high slab pair; the b/c
/// tables are length-`npml` 1D vectors along the stretched axis (the
/// transverse extent shares them, exactly as `slab_coefficients` broadcasts).
#[repr(C)]
pub struct CpmlSlabs<S> {
    // E-type psi (axis order matches simulate_3d's psi_step calls).
    exy_b_lo: *const S, exy_c_lo: *const S, exy_b_hi: *const S, exy_c_hi: *const S, // y
    exz_b_lo: *const S, exz_c_lo: *const S, exz_b_hi: *const S, exz_c_hi: *const S, // z
    eyz_b_lo: *const S, eyz_c_lo: *const S, eyz_b_hi: *const S, eyz_c_hi: *const S, // z
    eyx_b_lo: *const S, eyx_c_lo: *const S, eyx_b_hi: *const S, eyx_c_hi: *const S, // x
    ezx_b_lo: *const S, ezx_c_lo: *const S, ezx_b_hi: *const S, ezx_c_hi: *const S, // x
    ezy_b_lo: *const S, ezy_c_lo: *const S, ezy_b_hi: *const S, ezy_c_hi: *const S, // y
    // H-type psi.
    hxz_b_lo: *const S, hxz_c_lo: *const S, hxz_b_hi: *const S, hxz_c_hi: *const S, // z
    hxy_b_lo: *const S, hxy_c_lo: *const S, hxy_b_hi: *const S, hxy_c_hi: *const S, // y
    hyx_b_lo: *const S, hyx_c_lo: *const S, hyx_b_hi: *const S, hyx_c_hi: *const S, // x
    hyz_b_lo: *const S, hyz_c_lo: *const S, hyz_b_hi: *const S, hyz_c_hi: *const S, // z
    hzy_b_lo: *const S, hzy_c_lo: *const S, hzy_b_hi: *const S, hzy_c_hi: *const S, // y
    hzx_b_lo: *const S, hzx_c_lo: *const S, hzx_b_hi: *const S, hzx_c_hi: *const S, // x
}

/// Source / port / probe descriptors (read-only).
#[repr(C)]
pub struct SourceBuffers<S> {
    jx_i: *const c_int, jx_j: *const c_int, jx_k: *const c_int, cb_jx: *const S, jx_cur: *const S,
    jy_i: *const c_int, jy_j: *const c_int, jy_k: *const c_int, cb_jy: *const S, jy_cur: *const S,
    jz_i: *const c_int, jz_j: *const c_int, jz_k: *const c_int, cb_jz: *const S, jz_cur: *const S,
    mx_i: *const c_int, mx_j: *const c_int, mx_k: *const c_int, mx_cur: *const S,
    my_i: *const c_int, my_j: *const c_int, my_k: *const c_int, my_cur: *const S,
    mz_i: *const c_int, mz_j: *const c_int, mz_k: *const c_int, mz_cur: *const S,
    // Single z-directed RVS port (n_ports is 0 or 1).
    port_i: *const c_int, port_j: *const c_int, port_k: *const c_int,
    a_port: *const S, b_port: *const S, c_port: *const S,
    port_vs: *const S,
    probe_i: *const c_int, probe_j: *const c_int, probe_k: *const c_int,
    ph_e_re: *const f64, ph_e_im: *const f64,
    ph_h_re: *const f64, ph_h_im: *const f64,
}

/// Mutable fields and outputs (kernel writes; caller zero-initialized).
#[repr(C)]
pub struct FieldBuffers<S> {
    ex: *mut S, ey: *mut S, ez: *mut S,
    hx: *mut S, hy: *mut S, hz: *mut S,
    out_probe: *mut S,
    out_v: *mut S, out_i: *mut S,
    out_energy: *mut S,
    dft_ex_re: *mut f64, dft_ex_im: *mut f64,
    dft_ey_re: *mut f64, dft_ey_im: *mut f64,
    dft_ez_re: *mut f64, dft_ez_im: *mut f64,
    dft_hx_re: *mut f64, dft_hx_im: *mut f64,
    dft_hy_re: *mut f64, dft_hy_im: *mut f64,
    dft_hz_re: *mut f64, dft_hz_im: *mut f64,
}

/// Pick a worker count for an `nx*ny*nz` grid (tuned on an M1 Pro; override
/// with `GRADENNA_NTHREADS`). 3D cells are heavier than 2D so the thresholds
/// for stepping up to 4 / 6 threads sit at smaller cell counts.
fn choose_threads(cells: usize, nx: usize) -> usize {
    let hw = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    if let Ok(v) = std::env::var("GRADENNA_NTHREADS") {
        if let Ok(n) = v.parse::<usize>() {
            if n >= 1 {
                return n.min(hw).min(nx.max(1)).max(1);
            }
        }
    }
    // Cap at 6 P-cores (the 2 E-cores stall every barrier). 64^3 = 262144
    // cells uses all 6; tiny grids fall back to fewer x-slabs.
    let p_cores = 6;
    let want = if cells < 32 * 1024 {
        2
    } else if cells < 96 * 1024 {
        4
    } else {
        p_cores
    };
    want.min(hw).min(nx.max(1)).max(1)
}

/// Contiguous x-row block `[r0, r1)` for worker `t` of `nthreads`.
#[inline]
fn row_block(nx: usize, nthreads: usize, t: usize) -> (usize, usize) {
    let base = nx / nthreads;
    let rem = nx % nthreads;
    let r0 = t * base + t.min(rem);
    let r1 = r0 + base + if t < rem { 1 } else { 0 };
    (r0, r1)
}

/// Static low/high PML slab membership along an axis of `n` positions.
#[derive(Clone, Copy)]
struct Slab {
    lo1: usize, // [0, lo1) is the low PML slab
    hi0: usize, // [hi0, n) is the high PML slab
}
impl Slab {
    fn new(n: usize, npml: usize) -> Self {
        Slab { lo1: npml, hi0: n - npml }
    }
}

macro_rules! make_kernel3d {
    ($scalar:ty, $run_name:ident) => {
        /// Run the whole 3D time loop. See module docs for the contract.
        ///
        /// # Safety
        /// All pointers must be valid for the lengths implied by `SimParams3D`
        /// and the documented array shapes; buffers must outlive the call.
        #[no_mangle]
        pub unsafe extern "C" fn $run_name(
            p: *const SimParams3D,
            coeff: *const CoeffTables<$scalar>,
            cpml: *const CpmlSlabs<$scalar>,
            srcs: *const SourceBuffers<$scalar>,
            fields: *const FieldBuffers<$scalar>,
            dt_mu: $scalar,
            inv_dx: $scalar,
            inv_dy: $scalar,
            inv_dz: $scalar,
            dx: $scalar,
            dy: $scalar,
            dz: $scalar,
            mu0: $scalar,
        ) {
            let p = &*p;
            let c = &*coeff;
            let pml_t = &*cpml;
            let s = &*srcs;
            let f = &*fields;

            let nx = p.nx as usize;
            let ny = p.ny as usize;
            let nz = p.nz as usize;
            let npml = p.npml as usize;
            let n_steps = p.n_steps as usize;
            let n_jx = p.n_jx as usize;
            let n_jy = p.n_jy as usize;
            let n_jz = p.n_jz as usize;
            let n_ports = p.n_ports as usize;
            let n_mx = p.n_mx as usize;
            let n_my = p.n_my as usize;
            let n_mz = p.n_mz as usize;
            let n_probes = p.n_probes as usize;
            let n_freq = p.n_freq as usize;
            let rec_energy = p.record_energy != 0;
            let pml = npml > 0;

            // Field element counts.
            let ex_n = (nx - 1) * ny * nz;
            let ey_n = nx * (ny - 1) * nz;
            let ez_n = nx * ny * (nz - 1);
            let hx_n = nx * (ny - 1) * (nz - 1);
            let hy_n = (nx - 1) * ny * (nz - 1);
            let hz_n = (nx - 1) * (ny - 1) * nz;

            let sl = |ptr: *const $scalar, n: usize| -> &[$scalar] {
                slice::from_raw_parts(ptr, n)
            };
            let sli = |ptr: *const c_int, n: usize| -> &[c_int] {
                slice::from_raw_parts(ptr, n)
            };

            // Interior dims.
            let nim = nx.saturating_sub(2); // interior x (E)
            let njm = ny.saturating_sub(2); // interior y (E)
            let nkm = nz.saturating_sub(2); // interior z (E)

            // Coefficient tables.
            let ca_ex = sl(c.ca_ex, (nx - 1) * njm * nkm);
            let cb_ex = sl(c.cb_ex, (nx - 1) * njm * nkm);
            let ca_ey = sl(c.ca_ey, nim * (ny - 1) * nkm);
            let cb_ey = sl(c.cb_ey, nim * (ny - 1) * nkm);
            let ca_ez = sl(c.ca_ez, nim * njm * (nz - 1));
            let cb_ez = sl(c.cb_ez, nim * njm * (nz - 1));
            let ikx_e = sl(c.ikx_e, nim);
            let iky_e = sl(c.iky_e, njm);
            let ikz_e = sl(c.ikz_e, nkm);
            let ikx_h = sl(c.ikx_h, nx - 1);
            let iky_h = sl(c.iky_h, ny - 1);
            let ikz_h = sl(c.ikz_h, nz - 1);
            let eps = sl(c.eps, nx * ny * nz);

            // CPML slab b/c tables (length npml each).
            let np = if pml { npml } else { 0 };
            let exy_b_lo = sl(pml_t.exy_b_lo, np); let exy_c_lo = sl(pml_t.exy_c_lo, np);
            let exy_b_hi = sl(pml_t.exy_b_hi, np); let exy_c_hi = sl(pml_t.exy_c_hi, np);
            let exz_b_lo = sl(pml_t.exz_b_lo, np); let exz_c_lo = sl(pml_t.exz_c_lo, np);
            let exz_b_hi = sl(pml_t.exz_b_hi, np); let exz_c_hi = sl(pml_t.exz_c_hi, np);
            let eyz_b_lo = sl(pml_t.eyz_b_lo, np); let eyz_c_lo = sl(pml_t.eyz_c_lo, np);
            let eyz_b_hi = sl(pml_t.eyz_b_hi, np); let eyz_c_hi = sl(pml_t.eyz_c_hi, np);
            let eyx_b_lo = sl(pml_t.eyx_b_lo, np); let eyx_c_lo = sl(pml_t.eyx_c_lo, np);
            let eyx_b_hi = sl(pml_t.eyx_b_hi, np); let eyx_c_hi = sl(pml_t.eyx_c_hi, np);
            let ezx_b_lo = sl(pml_t.ezx_b_lo, np); let ezx_c_lo = sl(pml_t.ezx_c_lo, np);
            let ezx_b_hi = sl(pml_t.ezx_b_hi, np); let ezx_c_hi = sl(pml_t.ezx_c_hi, np);
            let ezy_b_lo = sl(pml_t.ezy_b_lo, np); let ezy_c_lo = sl(pml_t.ezy_c_lo, np);
            let ezy_b_hi = sl(pml_t.ezy_b_hi, np); let ezy_c_hi = sl(pml_t.ezy_c_hi, np);
            let hxz_b_lo = sl(pml_t.hxz_b_lo, np); let hxz_c_lo = sl(pml_t.hxz_c_lo, np);
            let hxz_b_hi = sl(pml_t.hxz_b_hi, np); let hxz_c_hi = sl(pml_t.hxz_c_hi, np);
            let hxy_b_lo = sl(pml_t.hxy_b_lo, np); let hxy_c_lo = sl(pml_t.hxy_c_lo, np);
            let hxy_b_hi = sl(pml_t.hxy_b_hi, np); let hxy_c_hi = sl(pml_t.hxy_c_hi, np);
            let hyx_b_lo = sl(pml_t.hyx_b_lo, np); let hyx_c_lo = sl(pml_t.hyx_c_lo, np);
            let hyx_b_hi = sl(pml_t.hyx_b_hi, np); let hyx_c_hi = sl(pml_t.hyx_c_hi, np);
            let hyz_b_lo = sl(pml_t.hyz_b_lo, np); let hyz_c_lo = sl(pml_t.hyz_c_lo, np);
            let hyz_b_hi = sl(pml_t.hyz_b_hi, np); let hyz_c_hi = sl(pml_t.hyz_c_hi, np);
            let hzy_b_lo = sl(pml_t.hzy_b_lo, np); let hzy_c_lo = sl(pml_t.hzy_c_lo, np);
            let hzy_b_hi = sl(pml_t.hzy_b_hi, np); let hzy_c_hi = sl(pml_t.hzy_c_hi, np);
            let hzx_b_lo = sl(pml_t.hzx_b_lo, np); let hzx_c_lo = sl(pml_t.hzx_c_lo, np);
            let hzx_b_hi = sl(pml_t.hzx_b_hi, np); let hzx_c_hi = sl(pml_t.hzx_c_hi, np);

            // Sources.
            let jx_i = sli(s.jx_i, n_jx); let jx_j = sli(s.jx_j, n_jx); let jx_k = sli(s.jx_k, n_jx);
            let cb_jx = sl(s.cb_jx, n_jx); let jx_cur = sl(s.jx_cur, n_steps * n_jx);
            let jy_i = sli(s.jy_i, n_jy); let jy_j = sli(s.jy_j, n_jy); let jy_k = sli(s.jy_k, n_jy);
            let cb_jy = sl(s.cb_jy, n_jy); let jy_cur = sl(s.jy_cur, n_steps * n_jy);
            let jz_i = sli(s.jz_i, n_jz); let jz_j = sli(s.jz_j, n_jz); let jz_k = sli(s.jz_k, n_jz);
            let cb_jz = sl(s.cb_jz, n_jz); let jz_cur = sl(s.jz_cur, n_steps * n_jz);
            let mx_i = sli(s.mx_i, n_mx); let mx_j = sli(s.mx_j, n_mx); let mx_k = sli(s.mx_k, n_mx);
            let mx_cur = sl(s.mx_cur, n_steps * n_mx);
            let my_i = sli(s.my_i, n_my); let my_j = sli(s.my_j, n_my); let my_k = sli(s.my_k, n_my);
            let my_cur = sl(s.my_cur, n_steps * n_my);
            let mz_i = sli(s.mz_i, n_mz); let mz_j = sli(s.mz_j, n_mz); let mz_k = sli(s.mz_k, n_mz);
            let mz_cur = sl(s.mz_cur, n_steps * n_mz);
            let port_i = sli(s.port_i, n_ports); let port_j = sli(s.port_j, n_ports);
            let port_k = sli(s.port_k, n_ports);
            let a_port = sl(s.a_port, n_ports); let b_port = sl(s.b_port, n_ports);
            let c_port = sl(s.c_port, n_ports); let port_vs = sl(s.port_vs, n_steps * n_ports);
            let probe_i = sli(s.probe_i, n_probes); let probe_j = sli(s.probe_j, n_probes);
            let probe_k = sli(s.probe_k, n_probes);
            let ph_e_re = slice::from_raw_parts(s.ph_e_re, n_steps * n_freq);
            let ph_e_im = slice::from_raw_parts(s.ph_e_im, n_steps * n_freq);
            let ph_h_re = slice::from_raw_parts(s.ph_h_re, n_steps * n_freq);
            let ph_h_im = slice::from_raw_parts(s.ph_h_im, n_steps * n_freq);

            // Mutable field / output pointers.
            let ex_p = Shared(f.ex); let ey_p = Shared(f.ey); let ez_p = Shared(f.ez);
            let hx_p = Shared(f.hx); let hy_p = Shared(f.hy); let hz_p = Shared(f.hz);
            let out_probe_p = Shared(f.out_probe);
            let out_v_p = Shared(f.out_v); let out_i_p = Shared(f.out_i);
            let out_energy_p = Shared(f.out_energy);
            let dft_ex_re_p = Shared(f.dft_ex_re); let dft_ex_im_p = Shared(f.dft_ex_im);
            let dft_ey_re_p = Shared(f.dft_ey_re); let dft_ey_im_p = Shared(f.dft_ey_im);
            let dft_ez_re_p = Shared(f.dft_ez_re); let dft_ez_im_p = Shared(f.dft_ez_im);
            let dft_hx_re_p = Shared(f.dft_hx_re); let dft_hx_im_p = Shared(f.dft_hx_im);
            let dft_hy_re_p = Shared(f.dft_hy_re); let dft_hy_im_p = Shared(f.dft_hy_im);
            let dft_hz_re_p = Shared(f.dft_hz_re); let dft_hz_im_p = Shared(f.dft_hz_im);

            // CPML psi slab storage. Shapes per appendix A of note 17; each is
            // (transverse..., npml-along-stretched-axis) flattened C-order, so
            // the stretched-axis slab index is the *innermost* unless the axis
            // is z (then the npml block is along k). We store each as a Vec and
            // hand workers Shared pointers.
            let mk = |n: usize| -> Vec<$scalar> {
                vec![<$scalar>::default(); if pml { n } else { 0 }]
            };
            // E-type psi.
            let mut exy_lo = mk((nx - 1) * npml * nkm); // exy: stretched y -> (nx-1, npml, nz-2)
            let mut exy_hi = mk((nx - 1) * npml * nkm);
            let mut exz_lo = mk((nx - 1) * njm * npml); // exz: stretched z -> (nx-1, ny-2, npml)
            let mut exz_hi = mk((nx - 1) * njm * npml);
            let mut eyz_lo = mk(nim * (ny - 1) * npml); // eyz: (nx-2, ny-1, npml)
            let mut eyz_hi = mk(nim * (ny - 1) * npml);
            let mut eyx_lo = mk(npml * (ny - 1) * nkm); // eyx: (npml, ny-1, nz-2)
            let mut eyx_hi = mk(npml * (ny - 1) * nkm);
            let mut ezx_lo = mk(npml * njm * (nz - 1)); // ezx: (npml, ny-2, nz-1)
            let mut ezx_hi = mk(npml * njm * (nz - 1));
            let mut ezy_lo = mk(nim * npml * (nz - 1)); // ezy: (nx-2, npml, nz-1)
            let mut ezy_hi = mk(nim * npml * (nz - 1));
            // H-type psi.
            let mut hxz_lo = mk(nx * (ny - 1) * npml); // hxz: (nx, ny-1, npml)
            let mut hxz_hi = mk(nx * (ny - 1) * npml);
            let mut hxy_lo = mk(nx * npml * (nz - 1)); // hxy: (nx, npml, nz-1)
            let mut hxy_hi = mk(nx * npml * (nz - 1));
            let mut hyx_lo = mk(npml * ny * (nz - 1)); // hyx: (npml, ny, nz-1)
            let mut hyx_hi = mk(npml * ny * (nz - 1));
            let mut hyz_lo = mk((nx - 1) * ny * npml); // hyz: (nx-1, ny, npml)
            let mut hyz_hi = mk((nx - 1) * ny * npml);
            let mut hzy_lo = mk((nx - 1) * npml * nz); // hzy: (nx-1, npml, nz)
            let mut hzy_hi = mk((nx - 1) * npml * nz);
            let mut hzx_lo = mk(npml * (ny - 1) * nz); // hzx: (npml, ny-1, nz)
            let mut hzx_hi = mk(npml * (ny - 1) * nz);

            let exy_lo_p = Shared(exy_lo.as_mut_ptr()); let exy_hi_p = Shared(exy_hi.as_mut_ptr());
            let exz_lo_p = Shared(exz_lo.as_mut_ptr()); let exz_hi_p = Shared(exz_hi.as_mut_ptr());
            let eyz_lo_p = Shared(eyz_lo.as_mut_ptr()); let eyz_hi_p = Shared(eyz_hi.as_mut_ptr());
            let eyx_lo_p = Shared(eyx_lo.as_mut_ptr()); let eyx_hi_p = Shared(eyx_hi.as_mut_ptr());
            let ezx_lo_p = Shared(ezx_lo.as_mut_ptr()); let ezx_hi_p = Shared(ezx_hi.as_mut_ptr());
            let ezy_lo_p = Shared(ezy_lo.as_mut_ptr()); let ezy_hi_p = Shared(ezy_hi.as_mut_ptr());
            let hxz_lo_p = Shared(hxz_lo.as_mut_ptr()); let hxz_hi_p = Shared(hxz_hi.as_mut_ptr());
            let hxy_lo_p = Shared(hxy_lo.as_mut_ptr()); let hxy_hi_p = Shared(hxy_hi.as_mut_ptr());
            let hyx_lo_p = Shared(hyx_lo.as_mut_ptr()); let hyx_hi_p = Shared(hyx_hi.as_mut_ptr());
            let hyz_lo_p = Shared(hyz_lo.as_mut_ptr()); let hyz_hi_p = Shared(hyz_hi.as_mut_ptr());
            let hzy_lo_p = Shared(hzy_lo.as_mut_ptr()); let hzy_hi_p = Shared(hzy_hi.as_mut_ptr());
            let hzx_lo_p = Shared(hzx_lo.as_mut_ptr()); let hzx_hi_p = Shared(hzx_hi.as_mut_ptr());

            // ez snapshot at the port cell (Ez^n before the E sweep overwrites).
            let mut ez_before = vec![<$scalar>::default(); n_ports];
            let ez_before_p = Shared(ez_before.as_mut_ptr());

            let half: $scalar = 0.5 as $scalar;
            let zero = <$scalar>::default();

            // Slab membership along the relevant axes (E interior / H half).
            let sx_eh = Slab::new(nx - 1, npml); // half-grid x (Hy/Hz, Ex rows)
            let sy_eh = Slab::new(ny - 1, npml); // half-grid y
            let sz_eh = Slab::new(nz - 1, npml); // half-grid z
            let sx_ei = Slab::new(nim, npml); // interior x (E)
            let sy_ei = Slab::new(njm, npml); // interior y (E)
            let sz_ei = Slab::new(nkm, npml); // interior z (E)

            let cells = nx * ny * nz;
            let nthreads = choose_threads(cells, nx);

            // Strides (k contiguous). Named <field>_si / _sj for i / j stride.
            // Ex (nx-1, ny, nz):   si = ny*nz,       sj = nz
            // Ey (nx, ny-1, nz):   si = (ny-1)*nz,   sj = nz
            // Ez (nx, ny, nz-1):   si = ny*(nz-1),   sj = nz-1
            // Hx (nx, ny-1, nz-1): si = (ny-1)*(nz-1), sj = nz-1
            // Hy (nx-1, ny, nz-1): si = ny*(nz-1),   sj = nz-1
            // Hz (nx-1, ny-1, nz): si = (ny-1)*nz,   sj = nz
            let ex_si = ny * nz; let ex_sj = nz;
            let ey_si = (ny - 1) * nz; let ey_sj = nz;
            let ez_si = ny * (nz - 1); let ez_sj = nz - 1;
            let hx_si = (ny - 1) * (nz - 1); let hx_sj = nz - 1;
            let hy_si = ny * (nz - 1); let hy_sj = nz - 1;
            let hz_si = (ny - 1) * nz; let hz_sj = nz;

            let worker = |t: usize, barrier: &SpinBarrier| {
                let mut sense = 0usize;
                let ex = ex_p.m(ex_n); let ey = ey_p.m(ey_n); let ez = ez_p.m(ez_n);
                let hx = hx_p.m(hx_n); let hy = hy_p.m(hy_n); let hz = hz_p.m(hz_n);
                let exy_lo = exy_lo_p.m(exy_lo.len()); let exy_hi = exy_hi_p.m(exy_hi.len());
                let exz_lo = exz_lo_p.m(exz_lo.len()); let exz_hi = exz_hi_p.m(exz_hi.len());
                let eyz_lo = eyz_lo_p.m(eyz_lo.len()); let eyz_hi = eyz_hi_p.m(eyz_hi.len());
                let eyx_lo = eyx_lo_p.m(eyx_lo.len()); let eyx_hi = eyx_hi_p.m(eyx_hi.len());
                let ezx_lo = ezx_lo_p.m(ezx_lo.len()); let ezx_hi = ezx_hi_p.m(ezx_hi.len());
                let ezy_lo = ezy_lo_p.m(ezy_lo.len()); let ezy_hi = ezy_hi_p.m(ezy_hi.len());
                let hxz_lo = hxz_lo_p.m(hxz_lo.len()); let hxz_hi = hxz_hi_p.m(hxz_hi.len());
                let hxy_lo = hxy_lo_p.m(hxy_lo.len()); let hxy_hi = hxy_hi_p.m(hxy_hi.len());
                let hyx_lo = hyx_lo_p.m(hyx_lo.len()); let hyx_hi = hyx_hi_p.m(hyx_hi.len());
                let hyz_lo = hyz_lo_p.m(hyz_lo.len()); let hyz_hi = hyz_hi_p.m(hyz_hi.len());
                let hzy_lo = hzy_lo_p.m(hzy_lo.len()); let hzy_hi = hzy_hi_p.m(hzy_hi.len());
                let hzx_lo = hzx_lo_p.m(hzx_lo.len()); let hzx_hi = hzx_hi_p.m(hzx_hi.len());
                let out_probe = out_probe_p.m(n_steps * n_probes);
                let out_v = out_v_p.m(n_steps * n_ports);
                let out_i = out_i_p.m(n_steps * n_ports);
                let ez_before = ez_before_p.m(n_ports);

                let (r0, r1) = row_block(nx, nthreads, t);

                for n in 0..n_steps {
                    // =====================================================
                    // (1) H sweep over owned x rows [r0, r1).
                    // =====================================================
                    h_sweep!(
                        $scalar,
                        r0, r1, nx, ny, nz, npml, pml, dt_mu, inv_dx, inv_dy, inv_dz,
                        ex, ey, ez, hx, hy, hz,
                        ex_si, ex_sj, ey_si, ey_sj, ez_si, ez_sj,
                        hx_si, hx_sj, hy_si, hy_sj, hz_si, hz_sj,
                        ikx_h, iky_h, ikz_h,
                        sx_eh, sy_eh, sz_eh,
                        hxz_lo, hxz_hi, hxz_b_lo, hxz_c_lo, hxz_b_hi, hxz_c_hi,
                        hxy_lo, hxy_hi, hxy_b_lo, hxy_c_lo, hxy_b_hi, hxy_c_hi,
                        hyx_lo, hyx_hi, hyx_b_lo, hyx_c_lo, hyx_b_hi, hyx_c_hi,
                        hyz_lo, hyz_hi, hyz_b_lo, hyz_c_lo, hyz_b_hi, hyz_c_hi,
                        hzy_lo, hzy_hi, hzy_b_lo, hzy_c_lo, hzy_b_hi, hzy_c_hi,
                        hzx_lo, hzx_hi, hzx_b_lo, hzx_c_lo, hzx_b_hi, hzx_c_hi
                    );

                    // Magnetic-current injection (owner of the row applies it).
                    for m in 0..n_mx {
                        let mi = mx_i[m] as usize;
                        if mi >= r0 && mi < r1 {
                            let idx = mi * hx_si + mx_j[m] as usize * hx_sj + mx_k[m] as usize;
                            hx[idx] -= dt_mu * mx_cur[n * n_mx + m];
                        }
                    }
                    for m in 0..n_my {
                        let mi = my_i[m] as usize;
                        if mi >= r0 && mi < r1.min(nx - 1) {
                            let idx = mi * hy_si + my_j[m] as usize * hy_sj + my_k[m] as usize;
                            hy[idx] -= dt_mu * my_cur[n * n_my + m];
                        }
                    }
                    for m in 0..n_mz {
                        let mi = mz_i[m] as usize;
                        if mi >= r0 && mi < r1.min(nx - 1) {
                            let idx = mi * hz_si + mz_j[m] as usize * hz_sj + mz_k[m] as usize;
                            hz[idx] -= dt_mu * mz_cur[n * n_mz + m];
                        }
                    }

                    barrier.wait(&mut sense);

                    // =====================================================
                    // (2) Port current (Ampere loop) + Ez^n snapshot.
                    //     Owner of the port row records I^{n+1/2} and the
                    //     pre-update Ez (read before the E sweep below).
                    // =====================================================
                    for kp in 0..n_ports {
                        let i = port_i[kp] as usize;
                        if i >= r0 && i < r1 {
                            let j = port_j[kp] as usize;
                            let k = port_k[kp] as usize;
                            // i_loop = (Hy[i,j,k]-Hy[i-1,j,k]) dy
                            //        + (Hx[i,j-1,k]-Hx[i,j,k]) dx
                            let hy_a = hy[i * hy_si + j * hy_sj + k];
                            let hy_b = hy[(i - 1) * hy_si + j * hy_sj + k];
                            let hx_a = hx[i * hx_si + (j - 1) * hx_sj + k];
                            let hx_b = hx[i * hx_si + j * hx_sj + k];
                            out_i[n * n_ports + kp] = (hy_a - hy_b) * dy + (hx_a - hx_b) * dx;
                            ez_before[kp] = ez[i * ez_si + j * ez_sj + k];
                        }
                    }

                    // =====================================================
                    // (3) E sweep over owned interior x rows.
                    // =====================================================
                    e_sweep!(
                        $scalar,
                        r0, r1, nx, ny, nz, npml, pml, inv_dx, inv_dy, inv_dz,
                        ex, ey, ez, hx, hy, hz,
                        ex_si, ex_sj, ey_si, ey_sj, ez_si, ez_sj,
                        hx_si, hx_sj, hy_si, hy_sj, hz_si, hz_sj,
                        ca_ex, cb_ex, ca_ey, cb_ey, ca_ez, cb_ez,
                        ikx_e, iky_e, ikz_e,
                        nim, njm, nkm, sx_ei, sy_ei, sz_ei,
                        exy_lo, exy_hi, exy_b_lo, exy_c_lo, exy_b_hi, exy_c_hi,
                        exz_lo, exz_hi, exz_b_lo, exz_c_lo, exz_b_hi, exz_c_hi,
                        eyz_lo, eyz_hi, eyz_b_lo, eyz_c_lo, eyz_b_hi, eyz_c_hi,
                        eyx_lo, eyx_hi, eyx_b_lo, eyx_c_lo, eyx_b_hi, eyx_c_hi,
                        ezx_lo, ezx_hi, ezx_b_lo, ezx_c_lo, ezx_b_hi, ezx_c_hi,
                        ezy_lo, ezy_hi, ezy_b_lo, ezy_c_lo, ezy_b_hi, ezy_c_hi
                    );

                    // =====================================================
                    // (4) Electric-current sources + port RVS + probes.
                    // =====================================================
                    for m in 0..n_jx {
                        let mi = jx_i[m] as usize;
                        if mi >= r0 && mi < r1.min(nx - 1) {
                            let idx = mi * ex_si + jx_j[m] as usize * ex_sj + jx_k[m] as usize;
                            ex[idx] -= cb_jx[m] * jx_cur[n * n_jx + m];
                        }
                    }
                    for m in 0..n_jy {
                        let mi = jy_i[m] as usize;
                        if mi >= r0 && mi < r1 {
                            let idx = mi * ey_si + jy_j[m] as usize * ey_sj + jy_k[m] as usize;
                            ey[idx] -= cb_jy[m] * jy_cur[n * n_jy + m];
                        }
                    }
                    for m in 0..n_jz {
                        let mi = jz_i[m] as usize;
                        if mi >= r0 && mi < r1 {
                            let idx = mi * ez_si + jz_j[m] as usize * ez_sj + jz_k[m] as usize;
                            ez[idx] -= cb_jz[m] * jz_cur[n * n_jz + m];
                        }
                    }
                    // Port RVS update overwrites the port Ez edge with the
                    // semi-implicit form, then records V^{n+1/2}.
                    for kp in 0..n_ports {
                        let i = port_i[kp] as usize;
                        if i >= r0 && i < r1 {
                            let j = port_j[kp] as usize;
                            let k = port_k[kp] as usize;
                            let g = i * ez_si + j * ez_sj + k;
                            // curl_z at the port interior cell (a=i-1, b=j-1).
                            // The E sweep already wrote ez[g] = ca*Ez^n +
                            // cb*curl_z; recover curl_z exactly as
                            // (ez_new - ca*Ez^n)/cb is fragile, so instead
                            // recompute curl_z directly from H (matches the
                            // simulate_3d port overwrite which uses curl_z).
                            let dhy_dx = (hy[i * hy_si + j * hy_sj + k]
                                - hy[(i - 1) * hy_si + j * hy_sj + k]) * inv_dx;
                            let dhx_dy = (hx[i * hx_si + j * hx_sj + k]
                                - hx[i * hx_si + (j - 1) * hx_sj + k]) * inv_dy;
                            // term_x - term_y with the ezx/ezy psi at this cell.
                            // The port lies strictly inside the PEC interior and
                            // (for the tested geometries) outside the PML, so
                            // the curl is the plain difference. Match the PML
                            // path by reading the psi already updated in (3).
                            let a = i - 1; let b = j - 1;
                            let mut term_x = dhy_dx * ikx_e[a];
                            if pml && a < sx_ei.lo1 {
                                let xs = a;
                                let pidx = xs * njm * (nz - 1) + b * (nz - 1) + k;
                                term_x = dhy_dx * ikx_e[a] + ezx_lo[pidx];
                            } else if pml && a >= sx_ei.hi0 {
                                let xs = a - sx_ei.hi0;
                                let pidx = xs * njm * (nz - 1) + b * (nz - 1) + k;
                                term_x = dhy_dx * ikx_e[a] + ezx_hi[pidx];
                            }
                            let mut term_y = dhx_dy * iky_e[b];
                            if pml && b < sy_ei.lo1 {
                                let ys = b;
                                let pidx = a * npml * (nz - 1) + ys * (nz - 1) + k;
                                term_y = dhx_dy * iky_e[b] + ezy_lo[pidx];
                            } else if pml && b >= sy_ei.hi0 {
                                let ys = b - sy_ei.hi0;
                                let pidx = a * npml * (nz - 1) + ys * (nz - 1) + k;
                                term_y = dhx_dy * iky_e[b] + ezy_hi[pidx];
                            }
                            let curl_z = term_x - term_y;
                            let ez_prev = ez_before[kp];
                            let ez_new = a_port[kp] * ez_prev + b_port[kp] * curl_z
                                + c_port[kp] * port_vs[n * n_ports + kp];
                            ez[g] = ez_new;
                            out_v[n * n_ports + kp] = -half * dz * (ez_prev + ez_new);
                        }
                    }
                    for kp in 0..n_probes {
                        let i = probe_i[kp] as usize;
                        if i >= r0 && i < r1 {
                            let idx = i * ez_si + probe_j[kp] as usize * ez_sj
                                + probe_k[kp] as usize;
                            out_probe[n * n_probes + kp] = ez[idx];
                        }
                    }

                    barrier.wait(&mut sense);

                    // =====================================================
                    // (5) Running-DFT accumulation over owned rows.
                    // =====================================================
                    if n_freq > 0 {
                        let dft_ex_re = dft_ex_re_p.m(n_freq * ex_n);
                        let dft_ex_im = dft_ex_im_p.m(n_freq * ex_n);
                        let dft_ey_re = dft_ey_re_p.m(n_freq * ey_n);
                        let dft_ey_im = dft_ey_im_p.m(n_freq * ey_n);
                        let dft_ez_re = dft_ez_re_p.m(n_freq * ez_n);
                        let dft_ez_im = dft_ez_im_p.m(n_freq * ez_n);
                        let dft_hx_re = dft_hx_re_p.m(n_freq * hx_n);
                        let dft_hx_im = dft_hx_im_p.m(n_freq * hx_n);
                        let dft_hy_re = dft_hy_re_p.m(n_freq * hy_n);
                        let dft_hy_im = dft_hy_im_p.m(n_freq * hy_n);
                        let dft_hz_re = dft_hz_re_p.m(n_freq * hz_n);
                        let dft_hz_im = dft_hz_im_p.m(n_freq * hz_n);
                        // Owned flat ranges per field (full j,k for x rows).
                        let ex_lo = r0 * ex_si; let ex_hi = r1.min(nx - 1) * ex_si;
                        let ey_lo = r0 * ey_si; let ey_hi = r1 * ey_si;
                        let ez_lo = r0 * ez_si; let ez_hi = r1 * ez_si;
                        let hx_lo = r0 * hx_si; let hx_hi = r1 * hx_si;
                        let hy_lo = r0 * hy_si; let hy_hi = r1.min(nx - 1) * hy_si;
                        let hz_lo = r0 * hz_si; let hz_hi = r1.min(nx - 1) * hz_si;
                        for kf in 0..n_freq {
                            let pe_re = ph_e_re[n * n_freq + kf];
                            let pe_im = ph_e_im[n * n_freq + kf];
                            let ph_re = ph_h_re[n * n_freq + kf];
                            let ph_im = ph_h_im[n * n_freq + kf];
                            dft_acc!(ex, dft_ex_re, dft_ex_im, kf * ex_n, ex_lo, ex_hi, pe_re, pe_im);
                            dft_acc!(ey, dft_ey_re, dft_ey_im, kf * ey_n, ey_lo, ey_hi, pe_re, pe_im);
                            dft_acc!(ez, dft_ez_re, dft_ez_im, kf * ez_n, ez_lo, ez_hi, pe_re, pe_im);
                            dft_acc!(hx, dft_hx_re, dft_hx_im, kf * hx_n, hx_lo, hx_hi, ph_re, ph_im);
                            dft_acc!(hy, dft_hy_re, dft_hy_im, kf * hy_n, hy_lo, hy_hi, ph_re, ph_im);
                            dft_acc!(hz, dft_hz_re, dft_hz_im, kf * hz_n, hz_lo, hz_hi, ph_re, ph_im);
                        }
                    }

                    // =====================================================
                    // (6) Energy diagnostic (worker 0, serial).
                    // =====================================================
                    if rec_energy && t == 0 {
                        let out_energy = out_energy_p.m(n_steps);
                        let cell = dx * dy * dz;
                        // ue = sum eps[:-1,:,:]*ex^2 + eps[:,:-1,:]*ey^2
                        //    + eps[:,:,:-1]*ez^2 (eps sliced to each E shape).
                        let mut ue = zero;
                        // Ex: eps[:-1, :, :] over (nx-1, ny, nz).
                        for i in 0..(nx - 1) {
                            for j in 0..ny {
                                for k in 0..nz {
                                    let e = ex[i * ex_si + j * ex_sj + k];
                                    ue += eps[i * ny * nz + j * nz + k] * e * e;
                                }
                            }
                        }
                        for i in 0..nx {
                            for j in 0..(ny - 1) {
                                for k in 0..nz {
                                    let e = ey[i * ey_si + j * ey_sj + k];
                                    ue += eps[i * ny * nz + j * nz + k] * e * e;
                                }
                            }
                        }
                        for i in 0..nx {
                            for j in 0..ny {
                                for k in 0..(nz - 1) {
                                    let e = ez[i * ez_si + j * ez_sj + k];
                                    ue += eps[i * ny * nz + j * nz + k] * e * e;
                                }
                            }
                        }
                        let mut uh = zero;
                        for &v in hx.iter() { uh += v * v; }
                        for &v in hy.iter() { uh += v * v; }
                        for &v in hz.iter() { uh += v * v; }
                        out_energy[n] = half * ue * cell + half * mu0 * uh * cell;
                    }

                    if n_freq > 0 || rec_energy {
                        barrier.wait(&mut sense);
                    }
                }
            };

            if nthreads <= 1 {
                let barrier = SpinBarrier::new(1);
                worker(0, &barrier);
            } else {
                let barrier = SpinBarrier::new(nthreads);
                std::thread::scope(|scope| {
                    for t in 1..nthreads {
                        let worker = &worker;
                        let barrier = &barrier;
                        scope.spawn(move || worker(t, barrier));
                    }
                    worker(0, &barrier);
                });
            }
        }
    };
}

/// One frequency's DFT accumulation over a flat owned range.
macro_rules! dft_acc {
    ($field:ident, $re:ident, $im:ident, $off:expr, $lo:expr, $hi:expr, $pr:expr, $pi:expr) => {{
        let off = $off;
        // Contiguous spans so LLVM widens the f32->f64 accumulate cleanly.
        let fld = &$field[$lo..$hi];
        let re = &mut $re[off + $lo..off + $hi];
        let im = &mut $im[off + $lo..off + $hi];
        let (pr, pi) = ($pr, $pi);
        for n in 0..fld.len() {
            let fv = fld[n] as f64;
            re[n] += pr * fv;
            im[n] += pi * fv;
        }
    }};
}

include!("lib3d_sweeps.rs");

make_kernel3d!(f64, gradenna_3d_run_f64);
make_kernel3d!(f32, gradenna_3d_run_f32);
