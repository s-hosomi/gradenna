//! Shared worker-pool primitives for the fused FDTD kernels.
//!
//! The 2D (`lib.rs`) and 3D (`lib3d.rs`) kernels use the identical resident
//! worker-pool machinery: a lightweight spin barrier, a raw shared pointer
//! wrapper, a static row-block partition and a measured thread-count heuristic.
//! These were byte-identical copies in both files; they live here so there is
//! one definition. Behavior is unchanged -- in particular `choose_threads` is
//! parameterized only by its (per-dimension, measured) cell thresholds, which
//! each kernel passes in via [`ThreadTuning`].

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
pub(crate) struct SpinBarrier {
    count: AtomicUsize,
    sense: AtomicUsize,
    n: usize,
}
impl SpinBarrier {
    pub(crate) fn new(n: usize) -> Self {
        SpinBarrier { count: AtomicUsize::new(0), sense: AtomicUsize::new(0), n }
    }
    /// `local` is this thread's private sense, toggled each call.
    #[inline]
    pub(crate) fn wait(&self, local: &mut usize) {
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

/// A raw `*mut T` that we promise is safe to share across the worker pool
/// because the workers only ever touch disjoint regions of the pointee at any
/// given barrier phase (rows are statically partitioned), and the few shared
/// reads happen after a barrier orders them. Standard pattern for a
/// hand-partitioned SIMD/threaded kernel.
#[derive(Clone, Copy)]
pub(crate) struct Shared<T>(pub(crate) *mut T);
unsafe impl<T> Send for Shared<T> {}
unsafe impl<T> Sync for Shared<T> {}
impl<T> Shared<T> {
    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    pub(crate) unsafe fn m(&self, len: usize) -> &'static mut [T] {
        slice::from_raw_parts_mut(self.0, len)
    }
}

/// Contiguous row block `[r0, r1)` assigned to worker `t` of `nthreads`.
#[inline]
pub(crate) fn row_block(nx: usize, nthreads: usize, t: usize) -> (usize, usize) {
    let base = nx / nthreads;
    let rem = nx % nthreads;
    let r0 = t * base + t.min(rem);
    let r1 = r0 + base + if t < rem { 1 } else { 0 };
    (r0, r1)
}

/// Measured cell thresholds for [`choose_threads`]. Each kernel supplies its
/// own (these were tuned on an M1 Pro per dimension; do not retune): below
/// `small_below` cells use 2 threads, below `medium_below` use 4, otherwise the
/// 6 P-cores.
#[derive(Clone, Copy)]
pub(crate) struct ThreadTuning {
    pub(crate) small_below: usize,
    pub(crate) medium_below: usize,
}

/// Pick a worker count for a grid of `cells` total cells and `nx` rows. Tuned
/// by measurement on an M1 Pro (6 P-cores + 2 E-cores); override with
/// `GRADENNA_NTHREADS` for other machines.
///
/// With the spin barrier the per-step sync is cheap, so even a cache-resident
/// grid scales to all 6 performance cores. Spilling onto the 2 E-cores (8
/// threads) is a large net loss -- they run the kernel several times slower and
/// every barrier then waits on them -- so we cap at 6 regardless of `hw`. Tiny
/// grids fall back to fewer threads where the row blocks would otherwise be a
/// handful of rows each.
pub(crate) fn choose_threads(cells: usize, nx: usize, tuning: ThreadTuning) -> usize {
    let hw = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    // Manual override for tuning / unusual machines.
    if let Ok(v) = std::env::var("GRADENNA_NTHREADS") {
        if let Ok(n) = v.parse::<usize>() {
            if n >= 1 {
                return n.min(hw).min(nx.max(1)).max(1);
            }
        }
    }
    let p_cores = 6;
    let want = if cells < tuning.small_below {
        2
    } else if cells < tuning.medium_below {
        4
    } else {
        p_cores
    };
    want.min(hw).min(nx.max(1)).max(1)
}
