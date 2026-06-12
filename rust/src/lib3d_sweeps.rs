// Inner-sweep macros for the 3D kernel (included into lib3d.rs).
//
// Each curl is computed on the fly with a k (z, stride-1) inner loop. The CPML
// psi recursion runs only on the two PML slabs of the relevant stretched axis;
// the interior is the plain Yee difference. The arithmetic mirrors
// gradenna.fdtd3d.simulate_3d term by term:
//   Hx += dt_mu*(t_hxz - t_hxy),  Hy += dt_mu*(t_hyx - t_hyz),
//   Hz += dt_mu*(t_hzy - t_hzx);
//   curl_x = t_exy - t_exz,  curl_y = t_eyz - t_eyx,  curl_z = t_ezx - t_ezy;
//   E = ca*E + cb*curl.
// where t_* = diff/kappa + psi on the stretched-axis PML slab, diff otherwise.

// ---------------------------------------------------------------------------
// H sweep. Field strides: <f>_si (i), <f>_sj (j), k is contiguous.
// ---------------------------------------------------------------------------
macro_rules! h_sweep {
    (
        $scalar:ty,
        $r0:expr, $r1:expr, $nx:expr, $ny:expr, $nz:expr, $npml:expr, $pml:expr,
        $dt_mu:expr, $inv_dx:expr, $inv_dy:expr, $inv_dz:expr,
        $ex:ident, $ey:ident, $ez:ident, $hx:ident, $hy:ident, $hz:ident,
        $ex_si:expr, $ex_sj:expr, $ey_si:expr, $ey_sj:expr, $ez_si:expr, $ez_sj:expr,
        $hx_si:expr, $hx_sj:expr, $hy_si:expr, $hy_sj:expr, $hz_si:expr, $hz_sj:expr,
        $ikx_h:ident, $iky_h:ident, $ikz_h:ident,
        $sx_eh:expr, $sy_eh:expr, $sz_eh:expr,
        $hxz_lo:ident, $hxz_hi:ident, $hxz_b_lo:ident, $hxz_c_lo:ident, $hxz_b_hi:ident, $hxz_c_hi:ident,
        $hxy_lo:ident, $hxy_hi:ident, $hxy_b_lo:ident, $hxy_c_lo:ident, $hxy_b_hi:ident, $hxy_c_hi:ident,
        $hyx_lo:ident, $hyx_hi:ident, $hyx_b_lo:ident, $hyx_c_lo:ident, $hyx_b_hi:ident, $hyx_c_hi:ident,
        $hyz_lo:ident, $hyz_hi:ident, $hyz_b_lo:ident, $hyz_c_lo:ident, $hyz_b_hi:ident, $hyz_c_hi:ident,
        $hzy_lo:ident, $hzy_hi:ident, $hzy_b_lo:ident, $hzy_c_lo:ident, $hzy_b_hi:ident, $hzy_c_hi:ident,
        $hzx_lo:ident, $hzx_hi:ident, $hzx_b_lo:ident, $hzx_c_lo:ident, $hzx_b_hi:ident, $hzx_c_hi:ident
    ) => {{
        let (r0, r1) = ($r0, $r1);
        let ny = $ny; let nz = $nz; let nx = $nx;
        let npml = $npml; let pml = $pml;
        let dt_mu = $dt_mu;
        // z slab boundaries (half grid, length nz-1) for hxz/hyz.
        let (zlo1, zhi0) = if pml { ($sz_eh.lo1, $sz_eh.hi0) } else { (0usize, nz - 1) };

        // ---- Hx (nx, ny-1, nz-1) ----
        // dey_dz (axis z -> hxz), dez_dy (axis y -> hxy); hx += dt_mu*(t_hxz - t_hxy)
        for i in r0..r1 {
            for j in 0..(ny - 1) {
                // y-slab membership for hxy is constant across this (i,j) row.
                let y_lo = pml && j < $sy_eh.lo1;
                let y_hi = pml && j >= $sy_eh.hi0;
                let ey_base = i * $ey_si + j * $ey_sj;
                let ez_lo_base = i * $ez_si + j * $ez_sj;
                let ez_hi_base = i * $ez_si + (j + 1) * $ez_sj;
                let hx_base = i * $hx_si + j * $hx_sj;
                // y psi setup
                let (iky_y, hxy_slab, hxy_b, hxy_c, ybase): (
                    $scalar, Option<&mut [$scalar]>, $scalar, $scalar, usize,
                ) = if y_lo {
                    let s = j;
                    ($iky_h[j], Some(&mut $hxy_lo[..]), $hxy_b_lo[s], $hxy_c_lo[s],
                     i * npml * (nz - 1) + s * (nz - 1))
                } else if y_hi {
                    let s = j - $sy_eh.hi0;
                    ($iky_h[j], Some(&mut $hxy_hi[..]), $hxy_b_hi[s], $hxy_c_hi[s],
                     i * npml * (nz - 1) + s * (nz - 1))
                } else {
                    (1.0 as $scalar, None, 0.0 as $scalar, 0.0 as $scalar, 0)
                };
                match hxy_slab {
                    None => {
                        // y interior: t_hxy = dez_dy. Split z for hxz psi.
                        for k in 0..zlo1 {
                            let dez_dy = ($ez[ez_hi_base + k] - $ez[ez_lo_base + k]) * $inv_dy;
                            let dey_dz = ($ey[ey_base + k + 1] - $ey[ey_base + k]) * $inv_dz;
                            let pidx = i * (ny - 1) * npml + j * npml + k;
                            $hxz_lo[pidx] = $hxz_b_lo[k] * $hxz_lo[pidx] + $hxz_c_lo[k] * dey_dz;
                            let t_hxz = dey_dz * $ikz_h[k] + $hxz_lo[pidx];
                            $hx[hx_base + k] += dt_mu * (t_hxz - dez_dy);
                        }
                        for k in zlo1..zhi0 {
                            let dez_dy = ($ez[ez_hi_base + k] - $ez[ez_lo_base + k]) * $inv_dy;
                            let dey_dz = ($ey[ey_base + k + 1] - $ey[ey_base + k]) * $inv_dz;
                            $hx[hx_base + k] += dt_mu * (dey_dz - dez_dy);
                        }
                        for k in zhi0..(nz - 1) {
                            let dez_dy = ($ez[ez_hi_base + k] - $ez[ez_lo_base + k]) * $inv_dy;
                            let dey_dz = ($ey[ey_base + k + 1] - $ey[ey_base + k]) * $inv_dz;
                            let s = k - $sz_eh.hi0;
                            let pidx = i * (ny - 1) * npml + j * npml + s;
                            $hxz_hi[pidx] = $hxz_b_hi[s] * $hxz_hi[pidx] + $hxz_c_hi[s] * dey_dz;
                            let t_hxz = dey_dz * $ikz_h[k] + $hxz_hi[pidx];
                            $hx[hx_base + k] += dt_mu * (t_hxz - dez_dy);
                        }
                    }
                    Some(hxy) => {
                        // y-slab row: y-psi (hxy) element-wise; z-psi (hxz)
                        // corners only. Corners first, then a vector middle.
                        let hxzrow = i * (ny - 1) * npml + j * npml;
                        for k in 0..zlo1 {
                            let dez_dy = ($ez[ez_hi_base + k] - $ez[ez_lo_base + k]) * $inv_dy;
                            let dey_dz = ($ey[ey_base + k + 1] - $ey[ey_base + k]) * $inv_dz;
                            let pidx = ybase + k;
                            hxy[pidx] = hxy_b * hxy[pidx] + hxy_c * dez_dy;
                            let t_hxy = dez_dy * iky_y + hxy[pidx];
                            let p = hxzrow + k;
                            $hxz_lo[p] = $hxz_b_lo[k] * $hxz_lo[p] + $hxz_c_lo[k] * dey_dz;
                            let t_hxz = dey_dz * $ikz_h[k] + $hxz_lo[p];
                            $hx[hx_base + k] += dt_mu * (t_hxz - t_hxy);
                        }
                        for k in zhi0..(nz - 1) {
                            let dez_dy = ($ez[ez_hi_base + k] - $ez[ez_lo_base + k]) * $inv_dy;
                            let dey_dz = ($ey[ey_base + k + 1] - $ey[ey_base + k]) * $inv_dz;
                            let pidx = ybase + k;
                            hxy[pidx] = hxy_b * hxy[pidx] + hxy_c * dez_dy;
                            let t_hxy = dez_dy * iky_y + hxy[pidx];
                            let s = k - $sz_eh.hi0;
                            let p = hxzrow + s;
                            $hxz_hi[p] = $hxz_b_hi[s] * $hxz_hi[p] + $hxz_c_hi[s] * dey_dz;
                            let t_hxz = dey_dz * $ikz_h[k] + $hxz_hi[p];
                            $hx[hx_base + k] += dt_mu * (t_hxz - t_hxy);
                        }
                        // Middle: t_hxz == dey_dz; y-psi element-wise.
                        let mid = zhi0 - zlo1;
                        let hx_m = &mut $hx[hx_base + zlo1..hx_base + zlo1 + mid];
                        let pym = &mut hxy[ybase + zlo1..ybase + zlo1 + mid];
                        for m in 0..mid {
                            let k = zlo1 + m;
                            let dez_dy = ($ez[ez_hi_base + k] - $ez[ez_lo_base + k]) * $inv_dy;
                            let dey_dz = ($ey[ey_base + k + 1] - $ey[ey_base + k]) * $inv_dz;
                            pym[m] = hxy_b * pym[m] + hxy_c * dez_dy;
                            let t_hxy = dez_dy * iky_y + pym[m];
                            hx_m[m] += dt_mu * (dey_dz - t_hxy);
                        }
                    }
                }
            }
        }

        // ---- Hy (nx-1, ny, nz-1) ----
        // dez_dx (axis x -> hyx), dex_dz (axis z -> hyz); hy += dt_mu*(t_hyx - t_hyz)
        let nz1 = nz - 1;
        for i in r0..r1.min(nx - 1) {
            let x_lo = pml && i < $sx_eh.lo1;
            let x_hi = pml && i >= $sx_eh.hi0;
            let in_xslab = x_lo || x_hi;
            let ikx_x = $ikx_h[i];
            let xs_hyx = if x_hi { i - $sx_eh.hi0 } else { i };
            let bx = if x_lo { $hyx_b_lo[xs_hyx] } else if x_hi { $hyx_b_hi[xs_hyx] } else { 0.0 as $scalar };
            let cx = if x_lo { $hyx_c_lo[xs_hyx] } else if x_hi { $hyx_c_hi[xs_hyx] } else { 0.0 as $scalar };
            for j in 0..ny {
                let ez_lo_base = i * $ez_si + j * $ez_sj;
                let ez_hi_base = (i + 1) * $ez_si + j * $ez_sj;
                let ex_base = i * $ex_si + j * $ex_sj;
                let hy_base = i * $hy_si + j * $hy_sj;
                let hyzb = i * ny * npml + j * npml; // hyz row base (z slabs)
                if !in_xslab {
                    // t_hyx == dez_dx. z-psi (hyz) only in the two z slabs.
                    for k in 0..zlo1 {
                        let dez_dx = ($ez[ez_hi_base + k] - $ez[ez_lo_base + k]) * $inv_dx;
                        let dex_dz = ($ex[ex_base + k + 1] - $ex[ex_base + k]) * $inv_dz;
                        let p = hyzb + k;
                        $hyz_lo[p] = $hyz_b_lo[k] * $hyz_lo[p] + $hyz_c_lo[k] * dex_dz;
                        let t_hyz = dex_dz * $ikz_h[k] + $hyz_lo[p];
                        $hy[hy_base + k] += dt_mu * (dez_dx - t_hyz);
                    }
                    for k in zlo1..zhi0 {
                        let dez_dx = ($ez[ez_hi_base + k] - $ez[ez_lo_base + k]) * $inv_dx;
                        let dex_dz = ($ex[ex_base + k + 1] - $ex[ex_base + k]) * $inv_dz;
                        $hy[hy_base + k] += dt_mu * (dez_dx - dex_dz);
                    }
                    for k in zhi0..nz1 {
                        let dez_dx = ($ez[ez_hi_base + k] - $ez[ez_lo_base + k]) * $inv_dx;
                        let dex_dz = ($ex[ex_base + k + 1] - $ex[ex_base + k]) * $inv_dz;
                        let s = k - $sz_eh.hi0;
                        let p = hyzb + s;
                        $hyz_hi[p] = $hyz_b_hi[s] * $hyz_hi[p] + $hyz_c_hi[s] * dex_dz;
                        let t_hyz = dex_dz * $ikz_h[k] + $hyz_hi[p];
                        $hy[hy_base + k] += dt_mu * (dez_dx - t_hyz);
                    }
                } else {
                    // x-slab row: x-psi (hyx) element-wise; z-psi (hyz) corners.
                    let hyxb = xs_hyx * ny * nz1 + j * nz1;
                    for k in 0..zlo1 {
                        let dez_dx = ($ez[ez_hi_base + k] - $ez[ez_lo_base + k]) * $inv_dx;
                        let dex_dz = ($ex[ex_base + k + 1] - $ex[ex_base + k]) * $inv_dz;
                        let p = hyxb + k;
                        let t_hyx = if x_lo {
                            $hyx_lo[p] = bx * $hyx_lo[p] + cx * dez_dx; dez_dx * ikx_x + $hyx_lo[p]
                        } else {
                            $hyx_hi[p] = bx * $hyx_hi[p] + cx * dez_dx; dez_dx * ikx_x + $hyx_hi[p]
                        };
                        let pz = hyzb + k;
                        $hyz_lo[pz] = $hyz_b_lo[k] * $hyz_lo[pz] + $hyz_c_lo[k] * dex_dz;
                        let t_hyz = dex_dz * $ikz_h[k] + $hyz_lo[pz];
                        $hy[hy_base + k] += dt_mu * (t_hyx - t_hyz);
                    }
                    for k in zhi0..nz1 {
                        let dez_dx = ($ez[ez_hi_base + k] - $ez[ez_lo_base + k]) * $inv_dx;
                        let dex_dz = ($ex[ex_base + k + 1] - $ex[ex_base + k]) * $inv_dz;
                        let p = hyxb + k;
                        let t_hyx = if x_lo {
                            $hyx_lo[p] = bx * $hyx_lo[p] + cx * dez_dx; dez_dx * ikx_x + $hyx_lo[p]
                        } else {
                            $hyx_hi[p] = bx * $hyx_hi[p] + cx * dez_dx; dez_dx * ikx_x + $hyx_hi[p]
                        };
                        let s = k - $sz_eh.hi0;
                        let pz = hyzb + s;
                        $hyz_hi[pz] = $hyz_b_hi[s] * $hyz_hi[pz] + $hyz_c_hi[s] * dex_dz;
                        let t_hyz = dex_dz * $ikz_h[k] + $hyz_hi[pz];
                        $hy[hy_base + k] += dt_mu * (t_hyx - t_hyz);
                    }
                    // Middle: t_hyz == dex_dz; x-psi element-wise.
                    let mid = zhi0 - zlo1;
                    let hy_m = &mut $hy[hy_base + zlo1..hy_base + zlo1 + mid];
                    let pxm = if x_lo { &mut $hyx_lo[hyxb + zlo1..hyxb + zlo1 + mid] }
                              else { &mut $hyx_hi[hyxb + zlo1..hyxb + zlo1 + mid] };
                    for m in 0..mid {
                        let k = zlo1 + m;
                        let dez_dx = ($ez[ez_hi_base + k] - $ez[ez_lo_base + k]) * $inv_dx;
                        let dex_dz = ($ex[ex_base + k + 1] - $ex[ex_base + k]) * $inv_dz;
                        pxm[m] = bx * pxm[m] + cx * dez_dx;
                        let t_hyx = dez_dx * ikx_x + pxm[m];
                        hy_m[m] += dt_mu * (t_hyx - dex_dz);
                    }
                }
            }
        }

        // ---- Hz (nx-1, ny-1, nz) ----
        // dex_dy (axis y -> hzy), dey_dx (axis x -> hzx); hz += dt_mu*(t_hzy - t_hzx)
        for i in r0..r1.min(nx - 1) {
            let x_lo = pml && i < $sx_eh.lo1;
            let x_hi = pml && i >= $sx_eh.hi0;
            let in_xslab = x_lo || x_hi;
            let ikx_x = $ikx_h[i];
            let xs_hzx = if x_hi { i - $sx_eh.hi0 } else { i };
            let bxz = if x_lo { $hzx_b_lo[xs_hzx] } else if x_hi { $hzx_b_hi[xs_hzx] } else { 0.0 as $scalar };
            let cxz = if x_lo { $hzx_c_lo[xs_hzx] } else if x_hi { $hzx_c_hi[xs_hzx] } else { 0.0 as $scalar };
            for j in 0..(ny - 1) {
                let y_lo = pml && j < $sy_eh.lo1;
                let y_hi = pml && j >= $sy_eh.hi0;
                let in_yslab = y_lo || y_hi;
                let iky_y = $iky_h[j];
                let ex_lo_base = i * $ex_si + j * $ex_sj;
                let ex_hi_base = i * $ex_si + (j + 1) * $ex_sj;
                let ey_lo_base = i * $ey_si + j * $ey_sj;
                let ey_hi_base = (i + 1) * $ey_si + j * $ey_sj;
                let hz_base = i * $hz_si + j * $hz_sj;
                let ys_hzy = if y_hi { j - $sy_eh.hi0 } else { j };
                if !in_xslab && !in_yslab {
                    // Hot path: plain curl, branch-free.
                    for k in 0..nz {
                        let dex_dy = ($ex[ex_hi_base + k] - $ex[ex_lo_base + k]) * $inv_dy;
                        let dey_dx = ($ey[ey_hi_base + k] - $ey[ey_lo_base + k]) * $inv_dx;
                        $hz[hz_base + k] += dt_mu * (dex_dy - dey_dx);
                    }
                } else {
                    // Hz has no z-psi -> each sub-case is a branch-free
                    // element-wise recurrence over k once slabs are bound.
                    let hz_r = &mut $hz[hz_base..hz_base + nz];
                    let hzyb = i * npml * nz + ys_hzy * nz;
                    let hzxb = xs_hzx * (ny - 1) * nz + j * nz;
                    if in_xslab && !in_yslab {
                        let psi = if x_lo { &mut $hzx_lo[hzxb..hzxb + nz] } else { &mut $hzx_hi[hzxb..hzxb + nz] };
                        for k in 0..nz {
                            let dex_dy = ($ex[ex_hi_base + k] - $ex[ex_lo_base + k]) * $inv_dy;
                            let dey_dx = ($ey[ey_hi_base + k] - $ey[ey_lo_base + k]) * $inv_dx;
                            psi[k] = bxz * psi[k] + cxz * dey_dx;
                            let t_hzx = dey_dx * ikx_x + psi[k];
                            hz_r[k] += dt_mu * (dex_dy - t_hzx);
                        }
                    } else if in_yslab && !in_xslab {
                        let byz = if y_lo { $hzy_b_lo[ys_hzy] } else { $hzy_b_hi[ys_hzy] };
                        let cyz = if y_lo { $hzy_c_lo[ys_hzy] } else { $hzy_c_hi[ys_hzy] };
                        let psi = if y_lo { &mut $hzy_lo[hzyb..hzyb + nz] } else { &mut $hzy_hi[hzyb..hzyb + nz] };
                        for k in 0..nz {
                            let dex_dy = ($ex[ex_hi_base + k] - $ex[ex_lo_base + k]) * $inv_dy;
                            let dey_dx = ($ey[ey_hi_base + k] - $ey[ey_lo_base + k]) * $inv_dx;
                            psi[k] = byz * psi[k] + cyz * dex_dy;
                            let t_hzy = dex_dy * iky_y + psi[k];
                            hz_r[k] += dt_mu * (t_hzy - dey_dx);
                        }
                    } else {
                        let byz = if y_lo { $hzy_b_lo[ys_hzy] } else { $hzy_b_hi[ys_hzy] };
                        let cyz = if y_lo { $hzy_c_lo[ys_hzy] } else { $hzy_c_hi[ys_hzy] };
                        let py = if y_lo { &mut $hzy_lo[hzyb..hzyb + nz] } else { &mut $hzy_hi[hzyb..hzyb + nz] };
                        let px = if x_lo { &mut $hzx_lo[hzxb..hzxb + nz] } else { &mut $hzx_hi[hzxb..hzxb + nz] };
                        for k in 0..nz {
                            let dex_dy = ($ex[ex_hi_base + k] - $ex[ex_lo_base + k]) * $inv_dy;
                            let dey_dx = ($ey[ey_hi_base + k] - $ey[ey_lo_base + k]) * $inv_dx;
                            py[k] = byz * py[k] + cyz * dex_dy;
                            px[k] = bxz * px[k] + cxz * dey_dx;
                            let t_hzy = dex_dy * iky_y + py[k];
                            let t_hzx = dey_dx * ikx_x + px[k];
                            hz_r[k] += dt_mu * (t_hzy - t_hzx);
                        }
                    }
                }
            }
        }
    }};
}

// ---------------------------------------------------------------------------
// E sweep. Interior updates only (PEC shell stays zero).
//   Ex[:, 1:-1, 1:-1] over (nx-1, ny-2, nz-2)
//   Ey[1:-1, :, 1:-1] over (nx-2, ny-1, nz-2)
//   Ez[1:-1, 1:-1, :] over (nx-2, ny-2, nz-1)
// ---------------------------------------------------------------------------
macro_rules! e_sweep {
    (
        $scalar:ty,
        $r0:expr, $r1:expr, $nx:expr, $ny:expr, $nz:expr, $npml:expr, $pml:expr,
        $inv_dx:expr, $inv_dy:expr, $inv_dz:expr,
        $ex:ident, $ey:ident, $ez:ident, $hx:ident, $hy:ident, $hz:ident,
        $ex_si:expr, $ex_sj:expr, $ey_si:expr, $ey_sj:expr, $ez_si:expr, $ez_sj:expr,
        $hx_si:expr, $hx_sj:expr, $hy_si:expr, $hy_sj:expr, $hz_si:expr, $hz_sj:expr,
        $ca_ex:ident, $cb_ex:ident, $ca_ey:ident, $cb_ey:ident, $ca_ez:ident, $cb_ez:ident,
        $ikx_e:ident, $iky_e:ident, $ikz_e:ident,
        $nim:expr, $njm:expr, $nkm:expr, $sx_ei:expr, $sy_ei:expr, $sz_ei:expr,
        $exy_lo:ident, $exy_hi:ident, $exy_b_lo:ident, $exy_c_lo:ident, $exy_b_hi:ident, $exy_c_hi:ident,
        $exz_lo:ident, $exz_hi:ident, $exz_b_lo:ident, $exz_c_lo:ident, $exz_b_hi:ident, $exz_c_hi:ident,
        $eyz_lo:ident, $eyz_hi:ident, $eyz_b_lo:ident, $eyz_c_lo:ident, $eyz_b_hi:ident, $eyz_c_hi:ident,
        $eyx_lo:ident, $eyx_hi:ident, $eyx_b_lo:ident, $eyx_c_lo:ident, $eyx_b_hi:ident, $eyx_c_hi:ident,
        $ezx_lo:ident, $ezx_hi:ident, $ezx_b_lo:ident, $ezx_c_lo:ident, $ezx_b_hi:ident, $ezx_c_hi:ident,
        $ezy_lo:ident, $ezy_hi:ident, $ezy_b_lo:ident, $ezy_c_lo:ident, $ezy_b_hi:ident, $ezy_c_hi:ident
    ) => {{
        let (r0, r1) = ($r0, $r1);
        let nx = $nx; let ny = $ny; let nz = $nz; let npml = $npml; let pml = $pml;
        let _nim = $nim; let njm = $njm; let nkm = $nkm;
        // Interior k slab bounds (E interior length nkm = nz-2).
        let (kzlo1, kzhi0) = if pml { ($sz_ei.lo1, $sz_ei.hi0) } else { (0usize, nkm) };

        // ---- Ex[:, 1:-1, 1:-1] (rows i in 0..nx-1) ----
        // dhz_dy (axis y -> exy), dhy_dz (axis z -> exz); curl_x = t_exy - t_exz
        // dhz_dy = (hz[:,1:,:]-hz[:,:-1,:])[:,:,1:-1]   over (nx-1, ny-2, nz-2)
        // dhy_dz = (hy[:,:,1:]-hy[:,:,:-1])[:,1:-1,:]
        // exy shape (nx-1, npml, nz-2); exz shape (nx-1, ny-2, npml).
        for i in r0..r1.min(nx - 1) {
            for jb in 0..njm {
                let j = jb + 1; // ez/hz j index (interior y row jb -> field j=jb+1)
                let y_lo = pml && jb < $sy_ei.lo1;
                let y_hi = pml && jb >= $sy_ei.hi0;
                let hz_hi_base = i * $hz_si + j * $hz_sj;
                let hz_lo_base = i * $hz_si + (j - 1) * $hz_sj;
                let hy_base = i * $hy_si + j * $hy_sj; // dhy_dz at j=jb+1
                let ex_base = i * $ex_si + j * $ex_sj;
                let coff = i * njm * nkm + jb * nkm;
                let exz_row = i * njm * npml + jb * npml; // exz (i, jb, s)
                let y_in_slab = y_lo || y_hi;
                if !y_in_slab {
                    // t_exy == dhz_dy. z-psi (exz) only in the two z slabs.
                    for kb in 0..kzlo1 {
                        let k = kb + 1;
                        let dhz_dy = ($hz[hz_hi_base + k] - $hz[hz_lo_base + k]) * $inv_dy;
                        let dhy_dz = ($hy[hy_base + k] - $hy[hy_base + k - 1]) * $inv_dz;
                        let p = exz_row + kb;
                        $exz_lo[p] = $exz_b_lo[kb] * $exz_lo[p] + $exz_c_lo[kb] * dhy_dz;
                        let t_exz = dhy_dz * $ikz_e[kb] + $exz_lo[p];
                        let cidx = coff + kb;
                        $ex[ex_base + k] =
                            $ca_ex[cidx] * $ex[ex_base + k] + $cb_ex[cidx] * (dhz_dy - t_exz);
                    }
                    {
                        // Plain middle [kzlo1, kzhi0) over contiguous slices.
                        let mid = kzhi0 - kzlo1;
                        let kk = kzlo1 + 1;
                        let ex_m = &mut $ex[ex_base + kk..ex_base + kk + mid];
                        let ca_m = &$ca_ex[coff + kzlo1..coff + kzlo1 + mid];
                        let cb_m = &$cb_ex[coff + kzlo1..coff + kzlo1 + mid];
                        let hza = &$hz[hz_hi_base + kk..hz_hi_base + kk + mid];
                        let hzb = &$hz[hz_lo_base + kk..hz_lo_base + kk + mid];
                        let hya = &$hy[hy_base + kk..hy_base + kk + mid];
                        let hyb = &$hy[hy_base + kk - 1..hy_base + kk - 1 + mid];
                        for m in 0..mid {
                            let dhz_dy = (hza[m] - hzb[m]) * $inv_dy;
                            let dhy_dz = (hya[m] - hyb[m]) * $inv_dz;
                            ex_m[m] = ca_m[m] * ex_m[m] + cb_m[m] * (dhz_dy - dhy_dz);
                        }
                    }
                    for kb in kzhi0..nkm {
                        let k = kb + 1;
                        let dhz_dy = ($hz[hz_hi_base + k] - $hz[hz_lo_base + k]) * $inv_dy;
                        let dhy_dz = ($hy[hy_base + k] - $hy[hy_base + k - 1]) * $inv_dz;
                        let s = kb - $sz_ei.hi0;
                        let p = exz_row + s;
                        $exz_hi[p] = $exz_b_hi[s] * $exz_hi[p] + $exz_c_hi[s] * dhy_dz;
                        let t_exz = dhy_dz * $ikz_e[kb] + $exz_hi[p];
                        let cidx = coff + kb;
                        $ex[ex_base + k] =
                            $ca_ex[cidx] * $ex[ex_base + k] + $cb_ex[cidx] * (dhz_dy - t_exz);
                    }
                } else {
                    // y-slab row: y-psi (exy) element-wise; z-psi (exz) at the
                    // two z corners only. Corners first, then a vector middle.
                    let ys = if y_lo { jb } else { jb - $sy_ei.hi0 };
                    let exy_b = if y_lo { $exy_b_lo[ys] } else { $exy_b_hi[ys] };
                    let exy_c = if y_lo { $exy_c_lo[ys] } else { $exy_c_hi[ys] };
                    let iky_y = $iky_e[jb];
                    let exy_ybase = i * npml * nkm + ys * nkm;
                    for kb in 0..kzlo1 {
                        let k = kb + 1;
                        let dhz_dy = ($hz[hz_hi_base + k] - $hz[hz_lo_base + k]) * $inv_dy;
                        let dhy_dz = ($hy[hy_base + k] - $hy[hy_base + k - 1]) * $inv_dz;
                        let pp = exy_ybase + kb;
                        let t_exy = if y_lo {
                            $exy_lo[pp] = exy_b * $exy_lo[pp] + exy_c * dhz_dy; dhz_dy * iky_y + $exy_lo[pp]
                        } else {
                            $exy_hi[pp] = exy_b * $exy_hi[pp] + exy_c * dhz_dy; dhz_dy * iky_y + $exy_hi[pp]
                        };
                        let pz = exz_row + kb;
                        $exz_lo[pz] = $exz_b_lo[kb] * $exz_lo[pz] + $exz_c_lo[kb] * dhy_dz;
                        let t_exz = dhy_dz * $ikz_e[kb] + $exz_lo[pz];
                        let cidx = coff + kb;
                        $ex[ex_base + k] = $ca_ex[cidx] * $ex[ex_base + k] + $cb_ex[cidx] * (t_exy - t_exz);
                    }
                    for kb in kzhi0..nkm {
                        let k = kb + 1;
                        let dhz_dy = ($hz[hz_hi_base + k] - $hz[hz_lo_base + k]) * $inv_dy;
                        let dhy_dz = ($hy[hy_base + k] - $hy[hy_base + k - 1]) * $inv_dz;
                        let pp = exy_ybase + kb;
                        let t_exy = if y_lo {
                            $exy_lo[pp] = exy_b * $exy_lo[pp] + exy_c * dhz_dy; dhz_dy * iky_y + $exy_lo[pp]
                        } else {
                            $exy_hi[pp] = exy_b * $exy_hi[pp] + exy_c * dhz_dy; dhz_dy * iky_y + $exy_hi[pp]
                        };
                        let s = kb - $sz_ei.hi0;
                        let pz = exz_row + s;
                        $exz_hi[pz] = $exz_b_hi[s] * $exz_hi[pz] + $exz_c_hi[s] * dhy_dz;
                        let t_exz = dhy_dz * $ikz_e[kb] + $exz_hi[pz];
                        let cidx = coff + kb;
                        $ex[ex_base + k] = $ca_ex[cidx] * $ex[ex_base + k] + $cb_ex[cidx] * (t_exy - t_exz);
                    }
                    // Middle: t_exz == dhy_dz; y-psi element-wise.
                    let mid = kzhi0 - kzlo1;
                    let kk = kzlo1 + 1;
                    let ex_m = &mut $ex[ex_base + kk..ex_base + kk + mid];
                    let ca_m = &$ca_ex[coff + kzlo1..coff + kzlo1 + mid];
                    let cb_m = &$cb_ex[coff + kzlo1..coff + kzlo1 + mid];
                    let pym = if y_lo { &mut $exy_lo[exy_ybase + kzlo1..exy_ybase + kzlo1 + mid] }
                              else { &mut $exy_hi[exy_ybase + kzlo1..exy_ybase + kzlo1 + mid] };
                    for m in 0..mid {
                        let k = kzlo1 + m + 1;
                        let dhz_dy = ($hz[hz_hi_base + k] - $hz[hz_lo_base + k]) * $inv_dy;
                        let dhy_dz = ($hy[hy_base + k] - $hy[hy_base + k - 1]) * $inv_dz;
                        pym[m] = exy_b * pym[m] + exy_c * dhz_dy;
                        let t_exy = dhz_dy * iky_y + pym[m];
                        ex_m[m] = ca_m[m] * ex_m[m] + cb_m[m] * (t_exy - dhy_dz);
                    }
                }
            }
        }

        // ---- Ey[1:-1, :, 1:-1] (rows i in 1..nx-1) ----
        // dhx_dz (axis z -> eyz), dhz_dx (axis x -> eyx); curl_y = t_eyz - t_eyx
        // dhx_dz = (hx[:,:,1:]-hx[:,:,:-1])[1:-1,:,:]  over (nx-2, ny-1, nz-2)
        // dhz_dx = (hz[1:,:,:]-hz[:-1,:,:])[:,:,1:-1]
        let ie0 = if r0 == 0 { 1 } else { r0 };
        let ie1 = r1.min(nx - 1);
        for i in ie0..ie1 {
            let ia = i - 1; // interior x index for eyx / ca_ey
            let x_lo = pml && ia < $sx_ei.lo1;
            let x_hi = pml && ia >= $sx_ei.hi0;
            let in_xslab = x_lo || x_hi;
            let ikx = $ikx_e[ia];
            let xs_eyx = if x_hi { ia - $sx_ei.hi0 } else { ia };
            for j in 0..(ny - 1) {
                let hx_base = i * $hx_si + j * $hx_sj;
                let hz_hi_base = i * $hz_si + j * $hz_sj;
                let hz_lo_base = (i - 1) * $hz_si + j * $hz_sj;
                let ey_base = i * $ey_si + j * $ey_sj;
                let coff = ia * (ny - 1) * nkm + j * nkm;
                let ezb = ia * (ny - 1) * npml + j * npml; // eyz row base (z slab)
                if !in_xslab {
                    // z-PML corners first (read the pre-update ey), then the
                    // plain-curl middle as one contiguous slice loop -- this
                    // form vectorizes even with runtime slab bounds, unlike a
                    // three-way k split with a short variable-length middle.
                    for kb in 0..kzlo1 {
                        let k = kb + 1;
                        let dhx_dz = ($hx[hx_base + k] - $hx[hx_base + k - 1]) * $inv_dz;
                        let dhz_dx = ($hz[hz_hi_base + k] - $hz[hz_lo_base + k]) * $inv_dx;
                        let p = ezb + kb;
                        $eyz_lo[p] = $eyz_b_lo[kb] * $eyz_lo[p] + $eyz_c_lo[kb] * dhx_dz;
                        let t_eyz = dhx_dz * $ikz_e[kb] + $eyz_lo[p];
                        let cidx = coff + kb;
                        $ey[ey_base + k] =
                            $ca_ey[cidx] * $ey[ey_base + k] + $cb_ey[cidx] * (t_eyz - dhz_dx);
                    }
                    for kb in kzhi0..nkm {
                        let k = kb + 1;
                        let dhx_dz = ($hx[hx_base + k] - $hx[hx_base + k - 1]) * $inv_dz;
                        let dhz_dx = ($hz[hz_hi_base + k] - $hz[hz_lo_base + k]) * $inv_dx;
                        let s = kb - $sz_ei.hi0;
                        let p = ezb + s;
                        $eyz_hi[p] = $eyz_b_hi[s] * $eyz_hi[p] + $eyz_c_hi[s] * dhx_dz;
                        let t_eyz = dhx_dz * $ikz_e[kb] + $eyz_hi[p];
                        let cidx = coff + kb;
                        $ey[ey_base + k] =
                            $ca_ey[cidx] * $ey[ey_base + k] + $cb_ey[cidx] * (t_eyz - dhz_dx);
                    }
                    // Plain middle [kzlo1, kzhi0) over contiguous slices.
                    let mid = kzhi0 - kzlo1;
                    let kk = kzlo1 + 1;
                    let ey_m = &mut $ey[ey_base + kk..ey_base + kk + mid];
                    let ca_m = &$ca_ey[coff + kzlo1..coff + kzlo1 + mid];
                    let cb_m = &$cb_ey[coff + kzlo1..coff + kzlo1 + mid];
                    let hxa = &$hx[hx_base + kk..hx_base + kk + mid];
                    let hxb = &$hx[hx_base + kk - 1..hx_base + kk - 1 + mid];
                    let hza = &$hz[hz_hi_base + kk..hz_hi_base + kk + mid];
                    let hzb = &$hz[hz_lo_base + kk..hz_lo_base + kk + mid];
                    for m in 0..mid {
                        let dhx_dz = (hxa[m] - hxb[m]) * $inv_dz;
                        let dhz_dx = (hza[m] - hzb[m]) * $inv_dx;
                        ey_m[m] = ca_m[m] * ey_m[m] + cb_m[m] * (dhx_dz - dhz_dx);
                    }
                } else {
                    // x-slab row: x-psi (eyx) is element-wise over k; z-psi
                    // (eyz) only at the two z corners. Corners first (reading
                    // pre-update ey + eyx), then a vectorizable middle slice.
                    let xbase = xs_eyx * (ny - 1) * nkm + j * nkm;
                    let bx = if x_lo { $eyx_b_lo[xs_eyx] } else { $eyx_b_hi[xs_eyx] };
                    let cx = if x_lo { $eyx_c_lo[xs_eyx] } else { $eyx_c_hi[xs_eyx] };
                    for kb in 0..kzlo1 {
                        let k = kb + 1;
                        let dhx_dz = ($hx[hx_base + k] - $hx[hx_base + k - 1]) * $inv_dz;
                        let dhz_dx = ($hz[hz_hi_base + k] - $hz[hz_lo_base + k]) * $inv_dx;
                        let pz = ezb + kb;
                        $eyz_lo[pz] = $eyz_b_lo[kb] * $eyz_lo[pz] + $eyz_c_lo[kb] * dhx_dz;
                        let t_eyz = dhx_dz * $ikz_e[kb] + $eyz_lo[pz];
                        let p = xbase + kb;
                        let t_eyx = if x_lo {
                            $eyx_lo[p] = bx * $eyx_lo[p] + cx * dhz_dx; dhz_dx * ikx + $eyx_lo[p]
                        } else {
                            $eyx_hi[p] = bx * $eyx_hi[p] + cx * dhz_dx; dhz_dx * ikx + $eyx_hi[p]
                        };
                        let cidx = coff + kb;
                        $ey[ey_base + k] = $ca_ey[cidx] * $ey[ey_base + k] + $cb_ey[cidx] * (t_eyz - t_eyx);
                    }
                    for kb in kzhi0..nkm {
                        let k = kb + 1;
                        let dhx_dz = ($hx[hx_base + k] - $hx[hx_base + k - 1]) * $inv_dz;
                        let dhz_dx = ($hz[hz_hi_base + k] - $hz[hz_lo_base + k]) * $inv_dx;
                        let s = kb - $sz_ei.hi0;
                        let pz = ezb + s;
                        $eyz_hi[pz] = $eyz_b_hi[s] * $eyz_hi[pz] + $eyz_c_hi[s] * dhx_dz;
                        let t_eyz = dhx_dz * $ikz_e[kb] + $eyz_hi[pz];
                        let p = xbase + kb;
                        let t_eyx = if x_lo {
                            $eyx_lo[p] = bx * $eyx_lo[p] + cx * dhz_dx; dhz_dx * ikx + $eyx_lo[p]
                        } else {
                            $eyx_hi[p] = bx * $eyx_hi[p] + cx * dhz_dx; dhz_dx * ikx + $eyx_hi[p]
                        };
                        let cidx = coff + kb;
                        $ey[ey_base + k] = $ca_ey[cidx] * $ey[ey_base + k] + $cb_ey[cidx] * (t_eyz - t_eyx);
                    }
                    // Middle [kzlo1, kzhi0): t_eyz == dhx_dz, x-psi element-wise.
                    let mid = kzhi0 - kzlo1;
                    let kk = kzlo1 + 1;
                    let ey_m = &mut $ey[ey_base + kk..ey_base + kk + mid];
                    let ca_m = &$ca_ey[coff + kzlo1..coff + kzlo1 + mid];
                    let cb_m = &$cb_ey[coff + kzlo1..coff + kzlo1 + mid];
                    let pxm = if x_lo { &mut $eyx_lo[xbase + kzlo1..xbase + kzlo1 + mid] }
                              else { &mut $eyx_hi[xbase + kzlo1..xbase + kzlo1 + mid] };
                    for m in 0..mid {
                        let k = kzlo1 + m + 1;
                        let dhx_dz = ($hx[hx_base + k] - $hx[hx_base + k - 1]) * $inv_dz;
                        let dhz_dx = ($hz[hz_hi_base + k] - $hz[hz_lo_base + k]) * $inv_dx;
                        pxm[m] = bx * pxm[m] + cx * dhz_dx;
                        let t_eyx = dhz_dx * ikx + pxm[m];
                        ey_m[m] = ca_m[m] * ey_m[m] + cb_m[m] * (dhx_dz - t_eyx);
                    }
                }
            }
        }

        // ---- Ez[1:-1, 1:-1, :] (rows i in 1..nx-1) ----
        // dhy_dx (axis x -> ezx), dhx_dy (axis y -> ezy); curl_z = t_ezx - t_ezy
        // dhy_dx = (hy[1:,:,:]-hy[:-1,:,:])[:,1:-1,:]  over (nx-2, ny-2, nz-1)
        // dhx_dy = (hx[:,1:,:]-hx[:,:-1,:])[1:-1,:,:]
        let nz1 = nz - 1;
        for i in ie0..ie1 {
            let ia = i - 1;
            let x_lo = pml && ia < $sx_ei.lo1;
            let x_hi = pml && ia >= $sx_ei.hi0;
            let in_xslab = x_lo || x_hi;
            let ikx = $ikx_e[ia];
            let xs_ezx = if x_hi { ia - $sx_ei.hi0 } else { ia };
            for jb in 0..njm {
                let j = jb + 1;
                let y_lo = pml && jb < $sy_ei.lo1;
                let y_hi = pml && jb >= $sy_ei.hi0;
                let in_yslab = y_lo || y_hi;
                let iky = $iky_e[jb];
                let ys_ezy = if y_hi { jb - $sy_ei.hi0 } else { jb };
                let hy_hi_base = i * $hy_si + j * $hy_sj;
                let hy_lo_base = (i - 1) * $hy_si + j * $hy_sj;
                let hx_hi_base = i * $hx_si + j * $hx_sj;
                let hx_lo_base = i * $hx_si + (j - 1) * $hx_sj;
                let ez_base = i * $ez_si + j * $ez_sj;
                let coff = ia * njm * nz1 + jb * nz1;
                if !in_xslab && !in_yslab {
                    // Hot path: plain Yee curl, branch-free, vectorizable.
                    for k in 0..nz1 {
                        let dhy_dx = ($hy[hy_hi_base + k] - $hy[hy_lo_base + k]) * $inv_dx;
                        let dhx_dy = ($hx[hx_hi_base + k] - $hx[hx_lo_base + k]) * $inv_dy;
                        let cidx = coff + k;
                        $ez[ez_base + k] =
                            $ca_ez[cidx] * $ez[ez_base + k] + $cb_ez[cidx] * (dhy_dx - dhx_dy);
                    }
                } else {
                    // Slow path: an x and/or y psi is live on this row. Ez has
                    // no z-psi, so each sub-case is a branch-free element-wise
                    // recurrence over k -> still vectorizable once the slab
                    // array and its b/c are bound outside the loop.
                    let ca_r = &$ca_ez[coff..coff + nz1];
                    let cb_r = &$cb_ez[coff..coff + nz1];
                    let ez_r = &mut $ez[ez_base..ez_base + nz1];
                    if in_xslab && !in_yslab {
                        let xbase = xs_ezx * njm * nz1 + jb * nz1;
                        let bx = if x_lo { $ezx_b_lo[xs_ezx] } else { $ezx_b_hi[xs_ezx] };
                        let cx = if x_lo { $ezx_c_lo[xs_ezx] } else { $ezx_c_hi[xs_ezx] };
                        let psi = if x_lo { &mut $ezx_lo[xbase..xbase + nz1] } else { &mut $ezx_hi[xbase..xbase + nz1] };
                        for k in 0..nz1 {
                            let dhy_dx = ($hy[hy_hi_base + k] - $hy[hy_lo_base + k]) * $inv_dx;
                            let dhx_dy = ($hx[hx_hi_base + k] - $hx[hx_lo_base + k]) * $inv_dy;
                            psi[k] = bx * psi[k] + cx * dhy_dx;
                            let t_ezx = dhy_dx * ikx + psi[k];
                            ez_r[k] = ca_r[k] * ez_r[k] + cb_r[k] * (t_ezx - dhx_dy);
                        }
                    } else if in_yslab && !in_xslab {
                        let ybase = ia * npml * nz1 + ys_ezy * nz1;
                        let by = if y_lo { $ezy_b_lo[ys_ezy] } else { $ezy_b_hi[ys_ezy] };
                        let cy = if y_lo { $ezy_c_lo[ys_ezy] } else { $ezy_c_hi[ys_ezy] };
                        let psi = if y_lo { &mut $ezy_lo[ybase..ybase + nz1] } else { &mut $ezy_hi[ybase..ybase + nz1] };
                        for k in 0..nz1 {
                            let dhy_dx = ($hy[hy_hi_base + k] - $hy[hy_lo_base + k]) * $inv_dx;
                            let dhx_dy = ($hx[hx_hi_base + k] - $hx[hx_lo_base + k]) * $inv_dy;
                            psi[k] = by * psi[k] + cy * dhx_dy;
                            let t_ezy = dhx_dy * iky + psi[k];
                            ez_r[k] = ca_r[k] * ez_r[k] + cb_r[k] * (dhy_dx - t_ezy);
                        }
                    } else {
                        // x and y slab corner (only ~npml^2 rows).
                        let xbase = xs_ezx * njm * nz1 + jb * nz1;
                        let ybase = ia * npml * nz1 + ys_ezy * nz1;
                        let bx = if x_lo { $ezx_b_lo[xs_ezx] } else { $ezx_b_hi[xs_ezx] };
                        let cx = if x_lo { $ezx_c_lo[xs_ezx] } else { $ezx_c_hi[xs_ezx] };
                        let by = if y_lo { $ezy_b_lo[ys_ezy] } else { $ezy_b_hi[ys_ezy] };
                        let cy = if y_lo { $ezy_c_lo[ys_ezy] } else { $ezy_c_hi[ys_ezy] };
                        let px = if x_lo { &mut $ezx_lo[xbase..xbase + nz1] } else { &mut $ezx_hi[xbase..xbase + nz1] };
                        let py = if y_lo { &mut $ezy_lo[ybase..ybase + nz1] } else { &mut $ezy_hi[ybase..ybase + nz1] };
                        for k in 0..nz1 {
                            let dhy_dx = ($hy[hy_hi_base + k] - $hy[hy_lo_base + k]) * $inv_dx;
                            let dhx_dy = ($hx[hx_hi_base + k] - $hx[hx_lo_base + k]) * $inv_dy;
                            px[k] = bx * px[k] + cx * dhy_dx;
                            py[k] = by * py[k] + cy * dhx_dy;
                            let t_ezx = dhy_dx * ikx + px[k];
                            let t_ezy = dhx_dy * iky + py[k];
                            ez_r[k] = ca_r[k] * ez_r[k] + cb_r[k] * (t_ezx - t_ezy);
                        }
                    }
                }
            }
        }
    }};
}
