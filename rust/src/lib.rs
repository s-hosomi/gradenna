//! Fused 2D TM FDTD time-loop kernel for gradenna (forward-only).
//!
//! This crate executes the *entire* `n_steps` time loop of
//! `gradenna.fdtd2d.simulate_tm` in a single native call. Two things make it
//! fast relative to the XLA-on-CPU path it replaces:
//!
//!   1. **True fusion.** Each time step is two field sweeps -- an H sweep
//!      (Hx, Hy + their CPML psi) and an E sweep (Ez interior + its CPML psi)
//!      -- and the spatial differences (`dEz/dy`, `dEz/dx`, `dHy/dx`,
//!      `dHx/dy`), the `1/kappa` scaling and the `term_x - term_y` curl are
//!      computed on the fly in row loops, kept in registers / per-row spans.
//!      The old design wrote full-grid `term_hx`, `term_hy`, `curl` scratch
//!      buffers and re-read them in a later pass; that traffic is gone.
//!
//!   2. **Resident worker pool + spin barrier.** Instead of forking and
//!      joining a rayon parallel region several times *per step* (each join
//!      costs microseconds -- ruinous on a 65 k-element sweep), we spawn the
//!      workers once, outside the time loop, statically assign each a
//!      contiguous block of grid rows, and synchronize with a lightweight
//!      sense-reversal `SpinBarrier` only where a step truly has a data
//!      dependence: once after the H sweep (the E sweep reads neighbouring H
//!      rows) and once after the E sweep + injection (closing the step, and --
//!      when a running DFT / energy diagnostic is active -- once more after
//!      that pass). The per-step scalar bookkeeping (ports, sources, probes)
//!      is distributed: each worker handles the cells that fall in its own row
//!      block, so it needs no extra synchronization.
//!
//! The worker count is chosen by grid size (see `choose_threads`): with the
//! cheap spin barrier even a cache-resident 256^2 grid scales to all 6 M1 Pro
//! performance cores; tiny grids use fewer.
//!
//! Numerical contract: the arithmetic mirrors `simulate_tm` operation by
//! operation (same update order, same `term_x - term_y` curl, same
//! semi-implicit port update) so results agree to f64 rel <= 1e-12 and
//! f32 rel <= 1e-5. The on-the-fly curl computes exactly the same value the
//! old scratch-buffer path did (`dhy_dx*1/kx + psi_x` minus
//! `dhx_dy*1/ky + psi_y`, plain differences outside the PML slabs); only the
//! intermediate storage changed. All coefficient tables are built on the
//! Python side and passed in flat; the kernel does no coefficient math.
//!
//! Memory layout: every 2D array is row-major (C order), last axis
//! contiguous. Field shapes: ez (nx, ny), hx (nx, ny-1), hy (nx-1, ny).

// Fused 3D Yee FDTD kernel (forward-only), the 3D analogue of this 2D kernel.
// Kept in its own module so this file (the 2D kernel) is untouched.
mod lib3d;

use std::os::raw::c_int;
use std::slice;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A spinning sense-reversal barrier. `std::sync::Barrier` parks threads on a
/// mutex + condvar, whose wake latency (microseconds) dominates when the work
/// between barriers is a single short field sweep on a cache-resident grid.
/// Here every worker is busy the whole run and the imbalance per phase is
/// tiny, so a brief spin (with `spin_loop` hints, then a yield) reaches the
/// rendezvous far faster. Correctness: the last arriver flips the shared
/// `sense`, which the spinners observe with Acquire/Release ordering -- the
/// flip also publishes all of that worker's prior writes to the others.
struct SpinBarrier {
    count: AtomicUsize,
    sense: AtomicUsize,
    n: usize,
}
impl SpinBarrier {
    fn new(n: usize) -> Self {
        SpinBarrier { count: AtomicUsize::new(0), sense: AtomicUsize::new(0), n }
    }
    /// `local` is this thread's private sense, toggled each call.
    #[inline]
    fn wait(&self, local: &mut usize) {
        let my = *local ^ 1;
        *local = my;
        if self.count.fetch_add(1, Ordering::AcqRel) + 1 == self.n {
            // Last in: reset the counter and release everyone.
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

/// Plain old data describing one simulation, shared by both float widths.
#[repr(C)]
pub struct SimParams {
    nx: c_int,
    ny: c_int,
    npml: c_int,
    n_steps: c_int,
    n_sources: c_int,
    n_ports: c_int,
    n_mx: c_int,
    n_my: c_int,
    n_probes: c_int,
    n_freq: c_int,
    record_energy: c_int,
}

/// A raw `*mut T` that we promise is safe to share across the worker pool
/// because the workers only ever touch disjoint regions of the pointee at any
/// given barrier phase (rows are statically partitioned), and the few shared
/// reads happen after a barrier orders them. Standard pattern for a
/// hand-partitioned SIMD/threaded kernel.
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

/// Static slab slice [start, stop) pair (low / high PML strip) on one axis.
#[derive(Clone, Copy)]
struct Slab {
    lo0: usize,
    lo1: usize,
    hi0: usize,
    #[allow(dead_code)]
    hi1: usize, // high-slab end; kept for symmetry / documentation
}
impl Slab {
    fn new(n: usize, npml: usize) -> Self {
        Slab { lo0: 0, lo1: npml, hi0: n - npml, hi1: n }
    }
}

/// Pick a worker count for an `nx*ny` grid. Tuned by measurement on an M1 Pro;
/// override with `GRADENNA_NTHREADS` for other machines.
fn choose_threads(cells: usize, nx: usize) -> usize {
    let hw = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    // Manual override for tuning / unusual machines.
    if let Ok(v) = std::env::var("GRADENNA_NTHREADS") {
        if let Ok(n) = v.parse::<usize>() {
            if n >= 1 {
                return n.min(hw).min(nx.max(1)).max(1);
            }
        }
    }
    // Tuned on an M1 Pro (6 P-cores + 2 E-cores). With the spin barrier the
    // per-step sync is cheap, so even a cache-resident 256^2 grid scales to all
    // 6 performance cores. Spilling onto the 2 E-cores (8 threads) is a large
    // net loss -- they run the kernel several times slower and every barrier
    // then waits on them -- so we cap at 6 regardless of `hw`. Tiny grids fall
    // back to fewer threads where the row blocks would otherwise be a handful
    // of rows each.
    let p_cores = 6;
    let want = if cells < 16 * 1024 {
        2 // ~128^2 and below: not enough work to amortize 6-way setup
    } else if cells < 64 * 1024 {
        4 // ~192^2
    } else {
        p_cores // 256^2 and up
    };
    want.min(hw).min(nx.max(1)).max(1)
}

/// Contiguous row block `[r0, r1)` assigned to worker `t` of `nthreads`.
#[inline]
fn row_block(nx: usize, nthreads: usize, t: usize) -> (usize, usize) {
    let base = nx / nthreads;
    let rem = nx % nthreads;
    let r0 = t * base + t.min(rem);
    let r1 = r0 + base + if t < rem { 1 } else { 0 };
    (r0, r1)
}

/// Generate the f32 and f64 kernels from one generic body.
macro_rules! make_kernel {
    ($scalar:ty, $run_name:ident) => {
        /// Run the whole time loop. See module docs for the contract.
        ///
        /// # Safety
        /// All pointers must be valid for the lengths implied by `SimParams`
        /// and the documented array shapes; the buffers must outlive the call.
        #[no_mangle]
        pub unsafe extern "C" fn $run_name(
            p: *const SimParams,
            ca: *const $scalar,
            cb: *const $scalar,
            eps: *const $scalar,
            inv_kx_e: *const $scalar,
            inv_ky_e: *const $scalar,
            inv_kx_h: *const $scalar,
            inv_ky_h: *const $scalar,
            b_ezx_lo: *const $scalar, c_ezx_lo: *const $scalar,
            b_ezx_hi: *const $scalar, c_ezx_hi: *const $scalar,
            b_ezy_lo: *const $scalar, c_ezy_lo: *const $scalar,
            b_ezy_hi: *const $scalar, c_ezy_hi: *const $scalar,
            b_hyx_lo: *const $scalar, c_hyx_lo: *const $scalar,
            b_hyx_hi: *const $scalar, c_hyx_hi: *const $scalar,
            b_hxy_lo: *const $scalar, c_hxy_lo: *const $scalar,
            b_hxy_hi: *const $scalar, c_hxy_hi: *const $scalar,
            dt_mu: $scalar,
            inv_dx: $scalar,
            inv_dy: $scalar,
            dx: $scalar,
            dy: $scalar,
            dz: $scalar,
            mu0: $scalar,
            src_i: *const c_int, src_j: *const c_int,
            cb_src: *const $scalar,
            src_cur: *const $scalar,
            port_i: *const c_int, port_j: *const c_int,
            cb_vs: *const $scalar,
            port_vs: *const $scalar,
            mx_i: *const c_int, mx_j: *const c_int, mx_cur: *const $scalar,
            my_i: *const c_int, my_j: *const c_int, my_cur: *const $scalar,
            probe_i: *const c_int, probe_j: *const c_int,
            ph_e_re: *const f64, ph_e_im: *const f64,
            ph_h_re: *const f64, ph_h_im: *const f64,
            ez: *mut $scalar,
            hx: *mut $scalar,
            hy: *mut $scalar,
            out_probe: *mut $scalar,
            out_v: *mut $scalar,
            out_i: *mut $scalar,
            out_energy: *mut $scalar,
            dft_ez_re: *mut f64, dft_ez_im: *mut f64,
            dft_hx_re: *mut f64, dft_hx_im: *mut f64,
            dft_hy_re: *mut f64, dft_hy_im: *mut f64,
        ) {
            let p = &*p;
            let nx = p.nx as usize;
            let ny = p.ny as usize;
            let npml = p.npml as usize;
            let n_steps = p.n_steps as usize;
            let n_src = p.n_sources as usize;
            let n_ports = p.n_ports as usize;
            let n_mx = p.n_mx as usize;
            let n_my = p.n_my as usize;
            let n_probes = p.n_probes as usize;
            let n_freq = p.n_freq as usize;
            let rec_energy = p.record_energy != 0;

            let ez_n = nx * ny;
            let hx_n = nx * (ny - 1);
            let hy_n = (nx - 1) * ny;

            // Coefficients / immutable tables (shared, read-only).
            let ca = slice::from_raw_parts(ca, ez_n);
            let cb = slice::from_raw_parts(cb, ez_n);
            let eps = slice::from_raw_parts(eps, ez_n);
            let inv_kx_e = slice::from_raw_parts(inv_kx_e, nx.saturating_sub(2));
            let inv_ky_e = slice::from_raw_parts(inv_ky_e, ny.saturating_sub(2));
            let inv_kx_h = slice::from_raw_parts(inv_kx_h, nx - 1);
            let inv_ky_h = slice::from_raw_parts(inv_ky_h, ny - 1);
            let sl = |ptr: *const $scalar, n: usize| -> &[$scalar] {
                slice::from_raw_parts(ptr, n)
            };
            let (b_ezx_lo, c_ezx_lo) = (sl(b_ezx_lo, npml), sl(c_ezx_lo, npml));
            let (b_ezx_hi, c_ezx_hi) = (sl(b_ezx_hi, npml), sl(c_ezx_hi, npml));
            let (b_ezy_lo, c_ezy_lo) = (sl(b_ezy_lo, npml), sl(c_ezy_lo, npml));
            let (b_ezy_hi, c_ezy_hi) = (sl(b_ezy_hi, npml), sl(c_ezy_hi, npml));
            let (b_hyx_lo, c_hyx_lo) = (sl(b_hyx_lo, npml), sl(c_hyx_lo, npml));
            let (b_hyx_hi, c_hyx_hi) = (sl(b_hyx_hi, npml), sl(c_hyx_hi, npml));
            let (b_hxy_lo, c_hxy_lo) = (sl(b_hxy_lo, npml), sl(c_hxy_lo, npml));
            let (b_hxy_hi, c_hxy_hi) = (sl(b_hxy_hi, npml), sl(c_hxy_hi, npml));

            let src_i = slice::from_raw_parts(src_i, n_src);
            let src_j = slice::from_raw_parts(src_j, n_src);
            let cb_src = sl(cb_src, n_src);
            let src_cur = sl(src_cur, n_steps * n_src);
            let port_i = slice::from_raw_parts(port_i, n_ports);
            let port_j = slice::from_raw_parts(port_j, n_ports);
            let cb_vs = sl(cb_vs, n_ports);
            let port_vs = sl(port_vs, n_steps * n_ports);
            let mx_i = slice::from_raw_parts(mx_i, n_mx);
            let mx_j = slice::from_raw_parts(mx_j, n_mx);
            let mx_cur = sl(mx_cur, n_steps * n_mx);
            let my_i = slice::from_raw_parts(my_i, n_my);
            let my_j = slice::from_raw_parts(my_j, n_my);
            let my_cur = sl(my_cur, n_steps * n_my);
            let probe_i = slice::from_raw_parts(probe_i, n_probes);
            let probe_j = slice::from_raw_parts(probe_j, n_probes);

            let ph_e_re = slice::from_raw_parts(ph_e_re, n_steps * n_freq);
            let ph_e_im = slice::from_raw_parts(ph_e_im, n_steps * n_freq);
            let ph_h_re = slice::from_raw_parts(ph_h_re, n_steps * n_freq);
            let ph_h_im = slice::from_raw_parts(ph_h_im, n_steps * n_freq);

            // Mutable fields / outputs as Shared raw pointers (workers touch
            // disjoint row blocks; barriers order the shared reads).
            let ez_p = Shared(ez);
            let hx_p = Shared(hx);
            let hy_p = Shared(hy);
            let out_probe_p = Shared(out_probe);
            let out_v_p = Shared(out_v);
            let out_i_p = Shared(out_i);
            let out_energy_p = Shared(out_energy);
            let dft_ez_re_p = Shared(dft_ez_re);
            let dft_ez_im_p = Shared(dft_ez_im);
            let dft_hx_re_p = Shared(dft_hx_re);
            let dft_hx_im_p = Shared(dft_hx_im);
            let dft_hy_re_p = Shared(dft_hy_re);
            let dft_hy_im_p = Shared(dft_hy_im);

            // CPML psi slab storage. Each is row-partitionable: hxy/hyx by Hx/Hy
            // row, ezx/ezy by interior-Ez row. We store them in plain Vecs and
            // hand workers Shared pointers; the partitioning keeps writes
            // disjoint.
            let pml = npml > 0;
            let mk = |n: usize| -> Vec<$scalar> {
                vec![<$scalar>::default(); if pml { n } else { 0 }]
            };
            let mut ezx_lo = mk(npml * (ny.saturating_sub(2)));
            let mut ezx_hi = mk(npml * (ny.saturating_sub(2)));
            let mut ezy_lo = mk((nx.saturating_sub(2)) * npml);
            let mut ezy_hi = mk((nx.saturating_sub(2)) * npml);
            let mut hyx_lo = mk(npml * ny);
            let mut hyx_hi = mk(npml * ny);
            let mut hxy_lo = mk(nx * npml);
            let mut hxy_hi = mk(nx * npml);
            let ezx_lo_p = Shared(ezx_lo.as_mut_ptr());
            let ezx_hi_p = Shared(ezx_hi.as_mut_ptr());
            let ezy_lo_p = Shared(ezy_lo.as_mut_ptr());
            let ezy_hi_p = Shared(ezy_hi.as_mut_ptr());
            let hyx_lo_p = Shared(hyx_lo.as_mut_ptr());
            let hyx_hi_p = Shared(hyx_hi.as_mut_ptr());
            let hxy_lo_p = Shared(hxy_lo.as_mut_ptr());
            let hxy_hi_p = Shared(hxy_hi.as_mut_ptr());

            // ez snapshot at port cells (written by worker 0 between barriers).
            let mut ez_before = vec![<$scalar>::default(); n_ports];
            let ez_before_p = Shared(ez_before.as_mut_ptr());

            let half: $scalar = 0.5 as $scalar;
            let zero = <$scalar>::default();

            // Slabs along the relevant axes.
            let sy_h = Slab::new(ny - 1, npml); // hxy stretched axis (y)
            let sx_h = Slab::new(nx - 1, npml); // hyx stretched axis (x)
            let sx_e = Slab::new(nx.saturating_sub(2), npml); // ezx (interior x)
            let sy_e = Slab::new(ny.saturating_sub(2), npml); // ezy (interior y)

            let cells = nx * ny;
            let nthreads = choose_threads(cells, nx);

            // ----- the per-worker time loop body -----
            // Captured by value/copy where possible; field & psi storage via
            // Shared. `t` is the worker index, `nthreads` the total.
            let ni = nx.saturating_sub(2);
            let nj = ny.saturating_sub(2);

            let worker = |t: usize, barrier: &SpinBarrier| {
                let _ = t;
                let mut sense = 0usize;
                let ez = ez_p.m(ez_n);
                let hx = hx_p.m(hx_n);
                let hy = hy_p.m(hy_n);
                let ezx_lo = ezx_lo_p.m(ezx_lo.len());
                let ezx_hi = ezx_hi_p.m(ezx_hi.len());
                let ezy_lo = ezy_lo_p.m(ezy_lo.len());
                let ezy_hi = ezy_hi_p.m(ezy_hi.len());
                let hyx_lo = hyx_lo_p.m(hyx_lo.len());
                let hyx_hi = hyx_hi_p.m(hyx_hi.len());
                let hxy_lo = hxy_lo_p.m(hxy_lo.len());
                let hxy_hi = hxy_hi_p.m(hxy_hi.len());
                let out_probe = out_probe_p.m(n_steps * n_probes);
                let out_v = out_v_p.m(n_steps * n_ports);
                let out_i = out_i_p.m(n_steps * n_ports);
                let ez_before = ez_before_p.m(n_ports);

                // Hx row block: this worker owns ez rows [r0, r1); it writes
                // Hx rows [r0, r1) and Hy rows [r0, min(r1, nx-1)).
                let (r0, r1) = row_block(nx, nthreads, t);
                let hy_r1 = r1.min(nx - 1);
                // Interior-Ez row block (a in [0, ni)) corresponds to ez rows
                // [1, nx-1); map this worker's ez rows to interior rows.
                let ia0 = if r0 == 0 { 0 } else { r0 - 1 };
                let ia1 = if r1 >= nx - 1 { ni } else { r1 - 1 };

                for n in 0..n_steps {
                    // =====================================================
                    // (1) H sweep over owned rows. term = dEz/d* (/kappa +
                    //     psi in slabs); Hx -= dt_mu*term, Hy += dt_mu*term.
                    // =====================================================
                    // ---- Hx rows r0..r1 (dez_dy on (nx, ny-1)) ----
                    for i in r0..r1 {
                        let ez_row = &ez[i * ny..i * ny + ny];
                        let hx_row = &mut hx[i * (ny - 1)..i * (ny - 1) + (ny - 1)];
                        if pml {
                            let lp = i * npml;
                            // y-slab-lo, plain middle, y-slab-hi (slabs carry
                            // the hxy psi; the plain middle vectorizes).
                            for j in 0..sy_h.lo1 {
                                let diff = (ez_row[j + 1] - ez_row[j]) * inv_dy;
                                let s = j - sy_h.lo0;
                                let pidx = lp + s;
                                hxy_lo[pidx] = b_hxy_lo[s] * hxy_lo[pidx] + c_hxy_lo[s] * diff;
                                hx_row[j] -= dt_mu * (diff * inv_ky_h[j] + hxy_lo[pidx]);
                            }
                            for j in sy_h.lo1..sy_h.hi0 {
                                let diff = (ez_row[j + 1] - ez_row[j]) * inv_dy;
                                hx_row[j] -= dt_mu * diff;
                            }
                            for j in sy_h.hi0..(ny - 1) {
                                let diff = (ez_row[j + 1] - ez_row[j]) * inv_dy;
                                let s = j - sy_h.hi0;
                                let pidx = lp + s;
                                hxy_hi[pidx] = b_hxy_hi[s] * hxy_hi[pidx] + c_hxy_hi[s] * diff;
                                hx_row[j] -= dt_mu * (diff * inv_ky_h[j] + hxy_hi[pidx]);
                            }
                        } else {
                            for j in 0..(ny - 1) {
                                let diff = (ez_row[j + 1] - ez_row[j]) * inv_dy;
                                hx_row[j] -= dt_mu * diff;
                            }
                        }
                    }
                    // ---- Hy rows r0..hy_r1 (dez_dx on (nx-1, ny)) ----
                    for i in r0..hy_r1 {
                        let ez_lo = &ez[i * ny..i * ny + ny];
                        let ez_hi = &ez[(i + 1) * ny..(i + 1) * ny + ny];
                        let hy_row = &mut hy[i * ny..i * ny + ny];
                        if pml && i < sx_h.lo1 {
                            let s = i - sx_h.lo0;
                            let ikx = inv_kx_h[i];
                            for j in 0..ny {
                                let diff = (ez_hi[j] - ez_lo[j]) * inv_dx;
                                let pidx = s * ny + j;
                                hyx_lo[pidx] = b_hyx_lo[s] * hyx_lo[pidx] + c_hyx_lo[s] * diff;
                                hy_row[j] += dt_mu * (diff * ikx + hyx_lo[pidx]);
                            }
                        } else if pml && i >= sx_h.hi0 {
                            let s = i - sx_h.hi0;
                            let ikx = inv_kx_h[i];
                            for j in 0..ny {
                                let diff = (ez_hi[j] - ez_lo[j]) * inv_dx;
                                let pidx = s * ny + j;
                                hyx_hi[pidx] = b_hyx_hi[s] * hyx_hi[pidx] + c_hyx_hi[s] * diff;
                                hy_row[j] += dt_mu * (diff * ikx + hyx_hi[pidx]);
                            }
                        } else {
                            for j in 0..ny {
                                let diff = (ez_hi[j] - ez_lo[j]) * inv_dx;
                                hy_row[j] += dt_mu * diff;
                            }
                        }
                    }
                    // Mx/My injection: owner of the row applies it.
                    for m in 0..n_mx {
                        let mi = mx_i[m] as usize;
                        if mi >= r0 && mi < r1 {
                            let idx = mi * (ny - 1) + mx_j[m] as usize;
                            hx[idx] -= dt_mu * mx_cur[n * n_mx + m];
                        }
                    }
                    for m in 0..n_my {
                        let mi = my_i[m] as usize;
                        if mi >= r0 && mi < hy_r1 {
                            let idx = mi * ny + my_j[m] as usize;
                            hy[idx] -= dt_mu * my_cur[n * n_my + m];
                        }
                    }

                    barrier.wait(&mut sense);

                    // =====================================================
                    // (2) Port current (Ampere loop) + Ez^n snapshot.
                    //     Done by the worker owning the port's row (i in its ez
                    //     block). The Ampere loop reads H at rows i-1, i (both
                    //     finalized by barrier A; the read may cross a block
                    //     boundary but it is read-only). ez_before is the
                    //     pre-E-update Ez at the port cell, so it must be read
                    //     here, before this worker's E sweep touches that row;
                    //     it is consumed in section (4) by the same worker.
                    // =====================================================
                    for k in 0..n_ports {
                        let i = port_i[k] as usize;
                        if i >= r0 && i < r1 {
                            let j = port_j[k] as usize;
                            let i_loop = (hy[i * ny + j] - hy[(i - 1) * ny + j]) * dy
                                - (hx[i * (ny - 1) + j] - hx[i * (ny - 1) + (j - 1)]) * dx;
                            out_i[n * n_ports + k] = i_loop;
                            ez_before[k] = ez[i * ny + j];
                        }
                    }

                    // =====================================================
                    // (3) E sweep over owned interior rows. curl on the fly.
                    //     Ez[1:-1,1:-1] = ca*Ez + cb*(term_x - term_y).
                    // =====================================================
                    if ni > 0 && nj > 0 {
                        // Column segments: y-slab-lo [0,ylo1), plain [ylo1,yhi0),
                        // y-slab-hi [yhi0,nj). The plain middle is a tight,
                        // branch-free, auto-vectorizable update; the slabs are
                        // O(npml) columns wide and carry the psi recursion.
                        let (ylo1, yhi0) = if pml { (sy_e.lo1, sy_e.hi0) } else { (0, nj) };
                        for a in ia0..ia1 {
                            let i = a + 1;
                            let hy_lo = &hy[a * ny..a * ny + ny];
                            let hy_hi = &hy[(a + 1) * ny..(a + 1) * ny + ny];
                            let hx_row =
                                &hx[(a + 1) * (ny - 1)..(a + 1) * (ny - 1) + (ny - 1)];
                            let ez_row = &mut ez[i * ny..i * ny + ny];
                            let ca_row = &ca[i * ny..i * ny + ny];
                            let cb_row = &cb[i * ny..i * ny + ny];

                            // x-slab membership is constant across the row.
                            let in_xlo = pml && a < sx_e.lo1;
                            let in_xhi = pml && a >= sx_e.hi0;
                            let in_xslab = in_xlo || in_xhi;

                            if !in_xslab {
                                // term_x == dhy_dx for the whole row. The middle
                                // column band is then fully plain -> vectorizes.
                                // y-slab-lo
                                for b in 0..ylo1 {
                                    let dhy_dx = (hy_hi[b + 1] - hy_lo[b + 1]) * inv_dx;
                                    let dhx_dy = (hx_row[b + 1] - hx_row[b]) * inv_dy;
                                    let s = b - sy_e.lo0;
                                    let pidx = a * npml + s;
                                    ezy_lo[pidx] =
                                        b_ezy_lo[s] * ezy_lo[pidx] + c_ezy_lo[s] * dhx_dy;
                                    let term_y = dhx_dy * inv_ky_e[b] + ezy_lo[pidx];
                                    let j = b + 1;
                                    ez_row[j] =
                                        ca_row[j] * ez_row[j] + cb_row[j] * (dhy_dx - term_y);
                                }
                                // plain middle (hot path)
                                for b in ylo1..yhi0 {
                                    let dhy_dx = (hy_hi[b + 1] - hy_lo[b + 1]) * inv_dx;
                                    let dhx_dy = (hx_row[b + 1] - hx_row[b]) * inv_dy;
                                    let j = b + 1;
                                    ez_row[j] =
                                        ca_row[j] * ez_row[j] + cb_row[j] * (dhy_dx - dhx_dy);
                                }
                                // y-slab-hi
                                for b in yhi0..nj {
                                    let dhy_dx = (hy_hi[b + 1] - hy_lo[b + 1]) * inv_dx;
                                    let dhx_dy = (hx_row[b + 1] - hx_row[b]) * inv_dy;
                                    let s = b - sy_e.hi0;
                                    let pidx = a * npml + s;
                                    ezy_hi[pidx] =
                                        b_ezy_hi[s] * ezy_hi[pidx] + c_ezy_hi[s] * dhx_dy;
                                    let term_y = dhx_dy * inv_ky_e[b] + ezy_hi[pidx];
                                    let j = b + 1;
                                    ez_row[j] =
                                        ca_row[j] * ez_row[j] + cb_row[j] * (dhy_dx - term_y);
                                }
                            } else {
                                // Row inside an x-slab (there are only 2*npml of
                                // them): term_x carries the ezx psi recursion.
                                let (xs, ezx_slab, b_ezx, c_ezx): (
                                    usize,
                                    &mut [$scalar],
                                    &[$scalar],
                                    &[$scalar],
                                ) = if in_xlo {
                                    (a - sx_e.lo0, ezx_lo, b_ezx_lo, c_ezx_lo)
                                } else {
                                    (a - sx_e.hi0, ezx_hi, b_ezx_hi, c_ezx_hi)
                                };
                                let ikx = inv_kx_e[a];
                                let bx = b_ezx[xs];
                                let cx = c_ezx[xs];
                                let xbase = xs * nj;
                                for b in 0..nj {
                                    let j = b + 1;
                                    let dhy_dx = (hy_hi[b + 1] - hy_lo[b + 1]) * inv_dx;
                                    let dhx_dy = (hx_row[b + 1] - hx_row[b]) * inv_dy;
                                    let pidx = xbase + b;
                                    ezx_slab[pidx] = bx * ezx_slab[pidx] + cx * dhy_dx;
                                    let term_x = dhy_dx * ikx + ezx_slab[pidx];
                                    let term_y = if b < ylo1 {
                                        let s = b - sy_e.lo0;
                                        let p = a * npml + s;
                                        ezy_lo[p] = b_ezy_lo[s] * ezy_lo[p] + c_ezy_lo[s] * dhx_dy;
                                        dhx_dy * inv_ky_e[b] + ezy_lo[p]
                                    } else if b >= yhi0 {
                                        let s = b - sy_e.hi0;
                                        let p = a * npml + s;
                                        ezy_hi[p] = b_ezy_hi[s] * ezy_hi[p] + c_ezy_hi[s] * dhx_dy;
                                        dhx_dy * inv_ky_e[b] + ezy_hi[p]
                                    } else {
                                        dhx_dy
                                    };
                                    ez_row[j] = ca_row[j] * ez_row[j] + cb_row[j] * (term_x - term_y);
                                }
                            }
                        }
                    }

                    // =====================================================
                    // (4) Jz source + port RVS injection + V record + probes.
                    //     Each worker handles the cells in *its* row block, so
                    //     no extra barrier is needed: writes hit only owned ez
                    //     rows, and the close barrier below orders them against
                    //     the next H sweep / the DFT pass. Order within a worker
                    //     (source, then port, then probe) matches simulate_tm.
                    // =====================================================
                    for m in 0..n_src {
                        let si = src_i[m] as usize;
                        if si >= r0 && si < r1 {
                            let idx = si * ny + src_j[m] as usize;
                            ez[idx] -= cb_src[m] * src_cur[n * n_src + m];
                        }
                    }
                    for k in 0..n_ports {
                        let i = port_i[k] as usize;
                        if i >= r0 && i < r1 {
                            let j = port_j[k] as usize;
                            let g = i * ny + j;
                            ez[g] -= cb_vs[k] * port_vs[n * n_ports + k];
                            out_v[n * n_ports + k] = -half * dz * (ez_before[k] + ez[g]);
                        }
                    }
                    for k in 0..n_probes {
                        let pi = probe_i[k] as usize;
                        if pi >= r0 && pi < r1 {
                            let idx = pi * ny + probe_j[k] as usize;
                            out_probe[n * n_probes + k] = ez[idx];
                        }
                    }

                    // Barrier B: all E updates + injections complete. This both
                    // closes the field state for the next step's H sweep and
                    // makes the fully-updated, source-injected fields visible to
                    // the DFT / energy passes below.
                    barrier.wait(&mut sense);

                    // =====================================================
                    // (5) Running-DFT accumulation over owned row blocks.
                    // =====================================================
                    if n_freq > 0 {
                        let dft_ez_re = dft_ez_re_p.m(n_freq * ez_n);
                        let dft_ez_im = dft_ez_im_p.m(n_freq * ez_n);
                        let dft_hx_re = dft_hx_re_p.m(n_freq * hx_n);
                        let dft_hx_im = dft_hx_im_p.m(n_freq * hx_n);
                        let dft_hy_re = dft_hy_re_p.m(n_freq * hy_n);
                        let dft_hy_im = dft_hy_im_p.m(n_freq * hy_n);
                        // Row ranges for each field for this worker.
                        let ez_lo = r0 * ny;
                        let ez_hi = r1 * ny;
                        let hx_lo = r0 * (ny - 1);
                        let hx_hi = r1 * (ny - 1);
                        let hy_lo = r0 * ny;
                        let hy_hi = hy_r1 * ny;
                        for k in 0..n_freq {
                            let pe_re = ph_e_re[n * n_freq + k];
                            let pe_im = ph_e_im[n * n_freq + k];
                            let ph_re = ph_h_re[n * n_freq + k];
                            let ph_im = ph_h_im[n * n_freq + k];
                            let eoff = k * ez_n;
                            for g in ez_lo..ez_hi {
                                let fv = ez[g] as f64;
                                dft_ez_re[eoff + g] += pe_re * fv;
                                dft_ez_im[eoff + g] += pe_im * fv;
                            }
                            let xoff = k * hx_n;
                            for g in hx_lo..hx_hi {
                                let fv = hx[g] as f64;
                                dft_hx_re[xoff + g] += ph_re * fv;
                                dft_hx_im[xoff + g] += ph_im * fv;
                            }
                            let yoff = k * hy_n;
                            for g in hy_lo..hy_hi {
                                let fv = hy[g] as f64;
                                dft_hy_re[yoff + g] += ph_re * fv;
                                dft_hy_im[yoff + g] += ph_im * fv;
                            }
                        }
                    }

                    // =====================================================
                    // (6) Energy diagnostic (worker 0, serial). Only used by
                    //     a small test grid -- not worth a parallel reduction.
                    // =====================================================
                    if rec_energy && t == 0 {
                        let out_energy = out_energy_p.m(n_steps);
                        let cell = dx * dy;
                        let mut ue = zero;
                        for g in 0..ez_n {
                            ue += eps[g] * ez[g] * ez[g];
                        }
                        let mut uh = zero;
                        for &v in hx.iter() {
                            uh += v * v;
                        }
                        for &v in hy.iter() {
                            uh += v * v;
                        }
                        out_energy[n] = half * ue * cell + half * mu0 * uh * cell;
                    }

                    // Barrier C: only needed when a DFT / energy pass ran after
                    // barrier B -- it read hx/hy/ez and the next step's H sweep
                    // will overwrite hx/hy, so order them. Without DFT/energy,
                    // barrier B already closed the step (the next H sweep only
                    // needs ez, finalized before B).
                    if n_freq > 0 || rec_energy {
                        barrier.wait(&mut sense);
                    }
                }
            };

            if nthreads <= 1 {
                // Single-thread: a SpinBarrier of 1 falls straight through (the
                // sole arriver is always the "last"), so we reuse the same body
                // -- the small-grid fast path stays identical to the parallel
                // code (no separate implementation to drift).
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

make_kernel!(f64, gradenna_tm_run_f64);
make_kernel!(f32, gradenna_tm_run_f32);
