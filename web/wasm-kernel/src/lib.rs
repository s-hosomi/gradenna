//! Educational single-threaded 2D TM-mode FDTD with CPML absorbing boundaries.
//!
//! Non-zero field components (Yee staggering, matching
//! `src/gradenna/fdtd2d.py`):
//!
//! ```text
//! Ez(i, j)        integer grid, shape (nx, ny)        -- outer ring is PEC
//! Hx(i, j+1/2)    half grid in y, shape (nx, ny-1)
//! Hy(i+1/2, j)    half grid in x, shape (nx-1, ny)
//! ```
//!
//! All field arrays are row-major with flat index `i * ny + j` (for Ez) so
//! that JavaScript can map a `Float32Array` view straight onto the wasm
//! linear memory returned by [`Fdtd2D::ez_ptr`].
//!
//! Update scheme (Schneider, *Understanding the FDTD Method*, Ch. 8/11;
//! identical discretization to the gradenna reference solver):
//!
//! ```text
//! Hx^{n+1/2} = Hx - (dt/mu) [ dEz/dy / kappa_y + psi_Hx,y ]
//! Hy^{n+1/2} = Hy + (dt/mu) [ dEz/dx / kappa_x + psi_Hy,x ]
//! Ez^{n+1}   = Ca Ez + Cb [ dHy/dx / kappa_x + psi_Ez,x
//!                           - dHx/dy / kappa_y - psi_Ez,y ]
//! ```
//!
//! with `Ca = (1 - sigma dt/2eps)/(1 + sigma dt/2eps)` and
//! `Cb = (dt/eps)/(1 + sigma dt/2eps)`. A soft source adds `src_val` onto the
//! Ez cell each step. This is a simplified, f32, single-threaded educational
//! port; it is deliberately independent of the performance kernel in `rust/`.

#![allow(clippy::needless_range_loop)]

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

// --- Physical constants (SI), f32 (educational demo precision) ----------------

const C0: f32 = 299_792_458.0; // speed of light [m/s]
const MU0: f32 = 1.256_637e-6; // vacuum permeability [H/m] (f32-exact, 0x35a8a9b8)
const EPS0: f32 = 1.0 / (MU0 * C0 * C0); // vacuum permittivity [F/m]
const ETA0: f32 = MU0 * C0; // free-space impedance [ohm]

// --- CPML parameters (Roden & Gedney 2000 / Taflove reference defaults) -------
//
// Mirrors `CPMLSpec` in `src/gradenna/cpml.py`. alpha_max is left at 0 (no
// complex-frequency-shift term) for the simple broadband demo, matching the
// gradenna default.

const PML_M: f32 = 3.0; // polynomial grading order for sigma and kappa
const PML_KAPPA_MAX: f32 = 5.0;
const PML_ALPHA_MAX: f32 = 0.0;
const PML_SIGMA_FACTOR: f32 = 0.75; // sigma_max as a fraction of sigma_opt

/// CPML auxiliary tables along one axis: `b`, `c` and `1/kappa` per position.
///
/// Outside the PML layer `c == 0` and `inv_kappa == 1`, so the psi recursion
/// stays identically zero and the update reduces to the plain Yee scheme.
struct AxisCoeffs {
    b: Vec<f32>,
    c: Vec<f32>,
    inv_kappa: Vec<f32>,
}

impl AxisCoeffs {
    /// Build the tables for an axis with `n` integer (Ez) positions.
    ///
    /// `half == false` evaluates at integer positions (length `n`, for the
    /// E-field psi); `half == true` evaluates at `i + 1/2` (length `n - 1`,
    /// for the H-field psi). `delta` is the cell size, `dt` the time step.
    fn new(n: usize, npml: usize, delta: f32, dt: f32, half: bool) -> AxisCoeffs {
        let len = if half { n - 1 } else { n };
        let mut b = vec![1.0f32; len];
        let mut c = vec![0.0f32; len];
        let mut inv_kappa = vec![1.0f32; len];

        if npml == 0 {
            return AxisCoeffs { b, c, inv_kappa };
        }

        let np = npml as f32;
        // sigma_max = sigma_factor * 0.8 (m+1) / (eta0 dx sqrt(eps_r_bg)),
        // with the background relative permittivity == 1.
        let sigma_max = PML_SIGMA_FACTOR * 0.8 * (PML_M + 1.0) / (ETA0 * delta);

        for idx in 0..len {
            let pos = if half { idx as f32 + 0.5 } else { idx as f32 };
            // rho: 0 at the PML/interior interface, 1 at the outer edge.
            let depth_left = (np - pos) / np;
            let depth_right = (pos - (n as f32 - 1.0 - np)) / np;
            let rho = depth_left.max(depth_right).clamp(0.0, 1.0);

            let rho_m = rho.powf(PML_M);
            let sigma = sigma_max * rho_m;
            let kappa = 1.0 + (PML_KAPPA_MAX - 1.0) * rho_m;
            let alpha = if rho > 0.0 {
                PML_ALPHA_MAX * (1.0 - rho).powf(1.0) // ma = 1
            } else {
                0.0
            };

            b[idx] = (-(sigma / kappa + alpha) * dt / EPS0).exp();
            let denom = sigma + kappa * alpha;
            c[idx] = if denom > 0.0 {
                sigma * (b[idx] - 1.0) / (kappa * denom)
            } else {
                0.0
            };
            inv_kappa[idx] = 1.0 / kappa;
        }

        AxisCoeffs { b, c, inv_kappa }
    }
}

/// Single-threaded 2D TM FDTD solver core (no wasm dependency).
///
/// The struct is plain Rust so it can be exercised by native `cargo test`;
/// the wasm-bindgen wrapper below simply re-exports its methods.
pub struct Solver {
    nx: usize,
    ny: usize,
    dt: f32,

    // Material-derived per-cell update coefficients (length nx*ny).
    ca: Vec<f32>,
    cb: Vec<f32>,

    // Fields.
    ez: Vec<f32>, // (nx, ny)
    hx: Vec<f32>, // (nx, ny-1)
    hy: Vec<f32>, // (nx-1, ny)

    // CPML auxiliary psi variables (full-size for clarity; zero outside PML).
    psi_ezx: Vec<f32>, // (nx, ny), x-stretched contribution to Ez
    psi_ezy: Vec<f32>, // (nx, ny), y-stretched contribution to Ez
    psi_hyx: Vec<f32>, // (nx-1, ny), x-stretched contribution to Hy
    psi_hxy: Vec<f32>, // (nx, ny-1), y-stretched contribution to Hx

    // CPML coefficient tables.
    cx_e: AxisCoeffs, // integer x, for Ez psi (dHy/dx)
    cy_e: AxisCoeffs, // integer y, for Ez psi (dHx/dy)
    cx_h: AxisCoeffs, // half x, for Hy psi (dEz/dx)
    cy_h: AxisCoeffs, // half y, for Hx psi (dEz/dy)

    inv_dx: f32,
}

impl Solver {
    /// Create a vacuum-filled solver on a square-cell grid (`dy == dx`).
    pub fn new(nx: usize, ny: usize, dx_m: f32, npml: usize) -> Solver {
        assert!(nx >= 3 && ny >= 3, "grid must be at least 3x3");
        assert!(dx_m > 0.0, "dx must be positive");
        assert!(
            nx > 2 * npml + 2 && ny > 2 * npml + 2,
            "grid too small for the requested PML thickness"
        );

        // Square cell: dt = 0.99 * dx / (c * sqrt(2)).
        let dt = 0.99 * dx_m / (C0 * std::f32::consts::SQRT_2);
        let inv_dx = 1.0 / dx_m;

        // Vacuum: eps = EPS0, sigma = 0 -> Ca = 1, Cb = dt/eps0.
        let n = nx * ny;
        let ca = vec![1.0f32; n];
        let cb = vec![dt / EPS0; n];

        Solver {
            nx,
            ny,
            dt,
            ca,
            cb,
            ez: vec![0.0; n],
            hx: vec![0.0; nx * (ny - 1)],
            hy: vec![0.0; (nx - 1) * ny],
            psi_ezx: vec![0.0; n],
            psi_ezy: vec![0.0; n],
            psi_hyx: vec![0.0; (nx - 1) * ny],
            psi_hxy: vec![0.0; nx * (ny - 1)],
            cx_e: AxisCoeffs::new(nx, npml, dx_m, dt, false),
            cy_e: AxisCoeffs::new(ny, npml, dx_m, dt, false),
            cx_h: AxisCoeffs::new(nx, npml, dx_m, dt, true),
            cy_h: AxisCoeffs::new(ny, npml, dx_m, dt, true),
            inv_dx,
        }
    }

    /// Rebuild Ca/Cb from a per-cell conductivity map (S/m), row-major
    /// `i*ny + j`, relative permittivity fixed at 1 (vacuum background).
    pub fn set_sigma(&mut self, sigma: &[f32]) {
        assert_eq!(sigma.len(), self.nx * self.ny, "sigma length mismatch");
        for k in 0..sigma.len() {
            let half_loss = sigma[k] * self.dt / (2.0 * EPS0);
            self.ca[k] = (1.0 - half_loss) / (1.0 + half_loss);
            self.cb[k] = (self.dt / EPS0) / (1.0 + half_loss);
        }
    }

    /// Reset the material to vacuum (Ca = 1, Cb = dt/eps0).
    pub fn clear_sigma(&mut self) {
        let cb = self.dt / EPS0;
        for k in 0..self.ca.len() {
            self.ca[k] = 1.0;
            self.cb[k] = cb;
        }
    }

    /// Zero all fields and CPML state; keeps the material coefficients.
    pub fn reset(&mut self) {
        self.ez.iter_mut().for_each(|v| *v = 0.0);
        self.hx.iter_mut().for_each(|v| *v = 0.0);
        self.hy.iter_mut().for_each(|v| *v = 0.0);
        self.psi_ezx.iter_mut().for_each(|v| *v = 0.0);
        self.psi_ezy.iter_mut().for_each(|v| *v = 0.0);
        self.psi_hyx.iter_mut().for_each(|v| *v = 0.0);
        self.psi_hxy.iter_mut().for_each(|v| *v = 0.0);
    }

    /// Advance one FDTD time step, soft-sourcing Ez at `(src_i, src_j)`.
    pub fn step(&mut self, src_i: usize, src_j: usize, src_val: f32) {
        let nx = self.nx;
        let ny = self.ny;
        let dt_mu = self.dt / MU0;
        let inv_dx = self.inv_dx; // square cells: inv_dy == inv_dx

        // --- Hx update: Hx(i, j+1/2), shape (nx, ny-1) -----------------------
        // mu dHx/dt = -dEz/dy  ->  Hx -= (dt/mu)(dEz/dy / kappa_y + psi_Hx,y)
        for i in 0..nx {
            let row_e = i * ny;
            let row_h = i * (ny - 1);
            for j in 0..ny - 1 {
                let diff = (self.ez[row_e + j + 1] - self.ez[row_e + j]) * inv_dx;
                let k = row_h + j;
                let psi = self.cy_h.b[j] * self.psi_hxy[k] + self.cy_h.c[j] * diff;
                self.psi_hxy[k] = psi;
                let term = diff * self.cy_h.inv_kappa[j] + psi;
                self.hx[k] -= dt_mu * term;
            }
        }

        // --- Hy update: Hy(i+1/2, j), shape (nx-1, ny) -----------------------
        // mu dHy/dt = +dEz/dx  ->  Hy += (dt/mu)(dEz/dx / kappa_x + psi_Hy,x)
        for i in 0..nx - 1 {
            let row_e0 = i * ny;
            let row_e1 = (i + 1) * ny;
            let row_h = i * ny;
            let bx = self.cx_h.b[i];
            let cxx = self.cx_h.c[i];
            let ikx = self.cx_h.inv_kappa[i];
            for j in 0..ny {
                let diff = (self.ez[row_e1 + j] - self.ez[row_e0 + j]) * inv_dx;
                let k = row_h + j;
                let psi = bx * self.psi_hyx[k] + cxx * diff;
                self.psi_hyx[k] = psi;
                let term = diff * ikx + psi;
                self.hy[k] += dt_mu * term;
            }
        }

        // --- Ez update on the interior (1..nx-1, 1..ny-1) --------------------
        // Outer ring stays PEC (zero). curl_z = dHy/dx - dHx/dy.
        for i in 1..nx - 1 {
            let row_e = i * ny;
            let row_hy0 = (i - 1) * ny; // Hy(i-1/2, j)
            let row_hy1 = i * ny; // Hy(i+1/2, j)
            let row_hx = i * (ny - 1); // Hx(i, j+/-1/2)
            let bx = self.cx_e.b[i];
            let cxx = self.cx_e.c[i];
            let ikx = self.cx_e.inv_kappa[i];
            for j in 1..ny - 1 {
                let dhy_dx = (self.hy[row_hy1 + j] - self.hy[row_hy0 + j]) * inv_dx;
                let dhx_dy = (self.hx[row_hx + j] - self.hx[row_hx + j - 1]) * inv_dx;

                let k = row_e + j;
                // x-stretched psi for Ez (from dHy/dx).
                let psi_x = bx * self.psi_ezx[k] + cxx * dhy_dx;
                self.psi_ezx[k] = psi_x;
                let term_x = dhy_dx * ikx + psi_x;
                // y-stretched psi for Ez (from dHx/dy).
                let psi_y = self.cy_e.b[j] * self.psi_ezy[k] + self.cy_e.c[j] * dhx_dy;
                self.psi_ezy[k] = psi_y;
                let term_y = dhx_dy * self.cy_e.inv_kappa[j] + psi_y;

                let curl = term_x - term_y;
                self.ez[k] = self.ca[k] * self.ez[k] + self.cb[k] * curl;
            }
        }

        // --- Soft source on Ez ----------------------------------------------
        if src_i < nx && src_j < ny {
            self.ez[src_i * ny + src_j] += src_val;
        }
    }

    #[inline]
    pub fn ez(&self) -> &[f32] {
        &self.ez
    }
}

// ============================================================================
// wasm-bindgen wrapper (only compiled for the wasm32 target).
// ============================================================================

/// Browser-facing 2D TM FDTD solver.
///
/// The exported API is frozen; see the crate README for usage from JS.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct Fdtd2D {
    inner: Solver,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl Fdtd2D {
    /// Vacuum-initialized solver. `dx_m` is the (square) cell size in metres,
    /// `npml` the CPML thickness in cells per side.
    #[wasm_bindgen(constructor)]
    pub fn new(nx: u32, ny: u32, dx_m: f32, npml: u32) -> Fdtd2D {
        Fdtd2D {
            inner: Solver::new(nx as usize, ny as usize, dx_m, npml as usize),
        }
    }

    /// Set the conductivity map (S/m), row-major `i*ny + j`, length `nx*ny`.
    pub fn set_sigma(&mut self, sigma: &[f32]) {
        self.inner.set_sigma(sigma);
    }

    /// Reset the material to vacuum.
    pub fn clear_sigma(&mut self) {
        self.inner.clear_sigma();
    }

    /// Advance one time step, soft-sourcing Ez at `(src_i, src_j)`.
    pub fn step(&mut self, src_i: u32, src_j: u32, src_val: f32) {
        self.inner.step(src_i as usize, src_j as usize, src_val);
    }

    /// Pointer to the row-major `nx*ny` Ez array in wasm linear memory.
    pub fn ez_ptr(&self) -> *const f32 {
        self.inner.ez.as_ptr()
    }

    pub fn nx(&self) -> u32 {
        self.inner.nx as u32
    }

    pub fn ny(&self) -> u32 {
        self.inner.ny as u32
    }

    /// Time step in seconds.
    pub fn dt_seconds(&self) -> f64 {
        self.inner.dt as f64
    }

    /// Zero all fields (keeps the conductivity map).
    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

// ============================================================================
// Native tests (cargo test, no wasm dependency).
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// A Ricker-wavelet soft source value at integer step `n`.
    fn ricker(n: usize, n0: f32, width: f32) -> f32 {
        let t = (n as f32 - n0) / width;
        let a = std::f32::consts::PI * std::f32::consts::PI * t * t;
        (1.0 - 2.0 * a) * (-a).exp()
    }

    /// Run a point source and return (final solver, peak |Ez| seen at center).
    fn run_point_source(nx: usize, ny: usize, npml: usize, steps: usize) -> Solver {
        let dx = 1e-3_f32;
        let mut s = Solver::new(nx, ny, dx, npml);
        let ci = nx / 2;
        let cj = ny / 2;
        for n in 0..steps {
            let src = ricker(n, 30.0, 12.0);
            s.step(ci, cj, src);
        }
        s
    }

    #[test]
    fn stability_uniform_grid_1000_steps() {
        // 1000 steps on a uniform grid must not blow up.
        let dx = 1e-3_f32;
        let mut s = Solver::new(80, 80, dx, 10);
        let mut peak = 0.0f32;
        for n in 0..1000 {
            let src = ricker(n, 30.0, 12.0);
            s.step(40, 40, src);
            let m = s.ez().iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            peak = peak.max(m);
            assert!(m.is_finite(), "field became non-finite at step {n}");
        }
        // Bounded: the late-time field must be far below the driven peak.
        let late = s.ez().iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        assert!(
            peak.is_finite() && peak < 1e3,
            "unbounded growth: peak {peak}"
        );
        assert!(late < peak, "field not decaying after the pulse");
    }

    #[test]
    fn cylindrical_wave_is_isotropic() {
        // Symmetric square grid + center source -> 4-fold symmetric |Ez|.
        let n = 101; // odd so the center cell is unique
        let s = run_point_source(n, n, 12, 60);
        let ci = n / 2;
        let cj = n / 2;
        let ny = s.ny;
        let r = 20; // sample radius (inside the PML-free interior)
        let at = |i: usize, j: usize| s.ez()[i * ny + j];
        let east = at(ci + r, cj);
        let west = at(ci - r, cj);
        let north = at(ci, cj + r);
        let south = at(ci, cj - r);
        let max_abs = east.abs().max(west.abs()).max(north.abs()).max(south.abs());
        assert!(max_abs > 1e-6, "no wave reached the sample radius");
        for (a, b) in [(east, west), (east, north), (east, south)] {
            assert!(
                (a - b).abs() < 1e-5,
                "anisotropic wavefront: {a} vs {b} (diff {})",
                (a - b).abs()
            );
        }
    }

    #[test]
    fn cpml_absorbs_outgoing_wave() {
        // After the pulse has left the interior, the residual at the center
        // must be well below the peak (CPML, no PEC reflection of note).
        let n = 81;
        let dx = 1e-3_f32;
        let mut s = Solver::new(n, n, dx, 12);
        let ci = n / 2;
        let cj = n / 2;
        // Track the peak |Ez| at the center cell while the pulse is present.
        let mut peak_center = 0.0f32;
        for step in 0..120 {
            let src = ricker(step, 30.0, 12.0);
            s.step(ci, cj, src);
            peak_center = peak_center.max(s.ez()[ci * n + cj].abs());
        }
        // Let the wave fully exit through the CPML.
        for _ in 0..400 {
            s.step(0, 0, 0.0); // no-op source (cell (0,0) is PEC anyway)
        }
        let residual = s.ez().iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let db = 20.0 * (residual / peak_center).log10();
        assert!(
            db < -40.0,
            "CPML reflection {db:.1} dB too large (residual {residual}, peak {peak_center})"
        );
    }

    #[test]
    fn conductor_reflects() {
        // A high-sigma wall traps the wave: with CPML the vacuum field is
        // absorbed and decays to near zero, but a reflecting conductor bounces
        // energy back so the late-time interior field stays much larger.
        let nx = 81;
        let ny = 81;
        let dx = 1e-3_f32;
        let npml = 10;

        let make = |with_wall: bool| -> Solver {
            let mut s = Solver::new(nx, ny, dx, npml);
            if with_wall {
                // Full PEC-like box just inside the PML: a high-conductivity
                // frame around the whole interior reflects the wave back in.
                let mut sigma = vec![0.0f32; nx * ny];
                let lo = npml + 2;
                let hi_i = nx - npml - 3;
                let hi_j = ny - npml - 3;
                for i in 0..nx {
                    for j in 0..ny {
                        let on_frame = i == lo || i == hi_i || j == lo || j == hi_j;
                        let inside = i >= lo && i <= hi_i && j >= lo && j <= hi_j;
                        if on_frame && inside {
                            sigma[i * ny + j] = 1e6;
                        }
                    }
                }
                s.set_sigma(&sigma);
            }
            // Drive the pulse, then let it propagate / get absorbed.
            for n in 0..400 {
                let src = ricker(n, 30.0, 12.0);
                s.step(nx / 2, ny / 2, src);
            }
            s
        };

        let vac = make(false);
        let wall = make(true);

        // Late-time interior energy (inside the conductor frame).
        let interior_energy = |s: &Solver| -> f32 {
            let mut e = 0.0f32;
            for i in npml + 3..nx - npml - 3 {
                for j in npml + 3..ny - npml - 3 {
                    let v = s.ez()[i * ny + j];
                    e += v * v;
                }
            }
            e
        };

        let e_vac = interior_energy(&vac);
        let e_wall = interior_energy(&wall);
        assert!(
            e_wall > 10.0 * e_vac,
            "expected the conductor to trap energy: vac {e_vac}, wall {e_wall}"
        );
    }

    #[test]
    fn conductor_field_inside_is_small() {
        // Field deep inside a strong conductor must be heavily attenuated.
        let nx = 81;
        let ny = 81;
        let dx = 1e-3_f32;
        let mut s = Solver::new(nx, ny, dx, 10);
        let mut sigma = vec![0.0f32; nx * ny];
        // Conductor block covering the eastern third of the interior.
        for i in nx / 2 + 5..nx - 11 {
            for j in 11..ny - 11 {
                sigma[i * ny + j] = 1e5;
            }
        }
        s.set_sigma(&sigma);
        for n in 0..200 {
            let src = ricker(n, 30.0, 12.0);
            s.step(nx / 2, ny / 2, src);
        }
        // Center of the conductor block.
        let ic = (nx / 2 + 5 + nx - 11) / 2;
        let jc = ny / 2;
        let inside = s.ez()[ic * ny + jc].abs();
        let at_source = s.ez()[(nx / 2) * ny + ny / 2].abs().max(1e-12);
        assert!(
            inside < 0.1 * at_source,
            "conductor did not attenuate field: inside {inside}, source {at_source}"
        );
    }
}
