"""Near-to-far-field (NTFF) transformation for the 2D TM and 3D solvers.

Surface-equivalence NTFF (research note 13; J. B. Schneider, *Understanding
the FDTD Method*, Ch. 14): the running-DFT phasors of the tangential fields
on a closed contour (2D) / box (3D) inside the PML are converted to
equivalent surface currents

    Js = n x H,    Ms = -n x E        (n = outward normal),

whose free-space radiation gives the far field. Everything in this module
is a fixed complex-linear map of the DFT phasors (plus smooth quotients for
directivity/gain), so it is exactly differentiable with `jax.grad`.

Conventions (shared by all functions here)
------------------------------------------
* Time convention: engineering e^{+j omega t}. The solvers accumulate the
  DFT with the kernel exp(-j omega t), so outward-travelling waves carry
  e^{-jk rho} (2D) / e^{-jkr} (3D) and the Schneider/Balanis far-field
  formulas apply verbatim.
* DFT normalization: the solver phasors are dt-scaled sums
  X(f) = dt sum_n x^n e^{-j omega t_n}. Directivity is a ratio and does not
  depend on this constant; `radiated_power_*` keeps it, so radiated power
  is directly comparable with source/port powers computed from spectra that
  use the *same* dt-scaled DFT (e.g. `gradenna.fdtd3d.time_series_dft`).
* Pattern amplitudes: the radial spreading and propagation phase are
  removed from the returned far fields,

      2D:  E_far(phi)         = lim_{rho->inf} sqrt(rho) e^{+jk rho} Ez(rho, phi)
      3D:  E_theta/phi(th,ph) = lim_{r->inf}   r         e^{+jk r}   E(r, th, ph)

  so |E_far|^2 / (2 eta0) is the radiation intensity per unit angle (2D,
  per unit length in z) / per unit solid angle (3D).
* Phase reference (origin of r'): the geometric center of the grid.
* Yee staggering: H is averaged spatially onto the E-aligned surface
  points (two adjacent samples in 2D, four in 3D); in 3D the tangential E
  components are likewise averaged onto the face-patch centers. The E/H
  half-*time*-step offset needs no correction here, because the solvers
  accumulate the running DFT with the exact sample times (n+1) dt for E
  and (n+1/2) dt for H — the phasors already refer to a common t = 0.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from gradenna.constants import C0, ETA0
from gradenna.fdtd3d import DFTMonitor, Grid3D
from gradenna.grid import Grid2D

__all__ = [
    "directivity_2d",
    "directivity_3d",
    "gain",
    "ntff_2d",
    "ntff_3d",
    "radiated_power_2d",
    "radiated_power_3d",
]


# ---------------------------------------------------------------------------
# Quadrature weights
# ---------------------------------------------------------------------------


def _periodic_trapezoid_weights(angles, period: float):
    """Trapezoid weights for samples of a periodic function.

    Each sample gets half the distance to its two (cyclic) neighbours, so a
    uniform grid `linspace(0, period, n, endpoint=False)` gets period/n per
    point, and a grid that duplicates the endpoint gets half weight at the
    duplicated pair — both reproduce the periodic trapezoid rule. The
    samples must be sorted and cover one full period.
    """
    a = jnp.asarray(angles)
    if a.ndim != 1 or a.shape[0] < 2:
        raise ValueError("need a 1D array of at least 2 angles")
    d_next = jnp.mod(jnp.roll(a, -1) - a, period)
    d_prev = jnp.mod(a - jnp.roll(a, 1), period)
    return 0.5 * (d_next + d_prev)


def _trapezoid_weights(x):
    """Plain (non-periodic) trapezoid weights for sorted 1D samples."""
    x = jnp.asarray(x)
    if x.ndim != 1 or x.shape[0] < 2:
        raise ValueError("need a 1D array of at least 2 samples")
    d = jnp.diff(x)
    return 0.5 * jnp.concatenate([d[:1], d[:-1] + d[1:], d[-1:]])


# ---------------------------------------------------------------------------
# 2D TM
# ---------------------------------------------------------------------------


def ntff_2d(dft_ez, dft_hx, dft_hy, grid: Grid2D, contour_margin: int, freqs, angles):
    """2D TM near-to-far-field transform on a rectangular contour.

    The contour is the rectangle of Ez nodes `contour_margin` cells inside
    the grid boundary (choose it a few cells inside the CPML interface,
    i.e. contour_margin > cpml.thickness). On each side the equivalent
    currents are (Schneider eqs. 14.57/14.60-14.63, e^{+j omega t})

        Jz    = n'_x Hy - n'_y Hx,
        M_phi = (n'_x cos(phi) + n'_y sin(phi)) Ez,

    and the far-field pattern amplitude (sqrt(rho) e^{+jk rho} removed) is

        E_far(phi) = -sqrt(j k / (8 pi)) *
                     closed-integral (eta0 Jz - M_phi) e^{+jk(x' cos phi + y' sin phi)} dl'

    discretized with trapezoid weights along each side (corner nodes are
    shared by two sides at half weight each, with the side's own normal).

    Yee staggering: Hx/Hy are half a cell off the Ez contour, so the two
    samples straddling each contour node are averaged onto it (e.g. on the
    right side Hy(i1-1/2, j) and Hy(i1+1/2, j) -> Hy(i1, j)). The half
    *time*-step offset of H is already compensated by the solver's
    exact-sample-time DFT ((n+1) dt for Ez, (n+1/2) dt for Hx/Hy), so no
    e^{+j omega dt/2} factor is applied here.

    Args:
        dft_ez: (n_freq, nx, ny) Ez phasors (`SimResult.dft_ez`).
        dft_hx: (n_freq, nx, ny-1) Hx phasors.
        dft_hy: (n_freq, nx-1, ny) Hy phasors.
        grid: the 2D Yee grid of the simulation.
        contour_margin: contour distance from the grid boundary in cells
            (>= 1; in practice >= cpml.thickness + a few cells).
        freqs: (n_freq,) frequencies [Hz] of the phasor axis, in order.
        angles: (n_angles,) observation angles phi [rad] from the +x axis.

    Returns:
        Complex (n_freq, n_angles) pattern amplitude E_far(f, phi); units
        are [V/m * s * sqrt(m)] for dt-scaled DFT inputs. The phase is
        referenced to the grid center.
    """
    dft_ez = jnp.asarray(dft_ez)
    dft_hx = jnp.asarray(dft_hx)
    dft_hy = jnp.asarray(dft_hy)
    freqs = np.atleast_1d(np.asarray(freqs, np.float64))
    if dft_ez.shape[0] != freqs.shape[0]:
        raise ValueError(
            f"freqs has {freqs.shape[0]} entries but dft_ez has {dft_ez.shape[0]} rows"
        )
    angles = jnp.atleast_1d(jnp.asarray(angles))

    nx, ny = grid.nx, grid.ny
    m = int(contour_margin)
    if m < 1:
        raise ValueError("contour_margin must be >= 1 (H averaging needs one cell)")
    i0, i1 = m, nx - 1 - m
    j0, j1 = m, ny - 1 - m
    if i1 - i0 < 2 or j1 - j0 < 2:
        raise ValueError(f"contour_margin {m} leaves no contour on a {nx}x{ny} grid")

    # Node coordinates relative to the grid center (phase reference).
    xn = (np.arange(nx) - 0.5 * (nx - 1)) * grid.dx
    yn = (np.arange(ny) - 0.5 * (ny - 1)) * grid.dy

    def side_weights(n_pts: int, dl: float) -> np.ndarray:
        w = np.full(n_pts, dl)
        w[0] *= 0.5
        w[-1] *= 0.5
        return w

    xs, ys, nxs, nys, ws, jzs, ezs = [], [], [], [], [], [], []

    def add_side(x, y, n_x, n_y, w, jz, ez):
        n_pts = w.shape[0]
        xs.append(np.broadcast_to(x, (n_pts,)))
        ys.append(np.broadcast_to(y, (n_pts,)))
        nxs.append(np.full(n_pts, float(n_x)))
        nys.append(np.full(n_pts, float(n_y)))
        ws.append(w)
        jzs.append(jz)
        ezs.append(ez)

    jr = slice(j0, j1 + 1)
    ir = slice(i0, i1 + 1)
    # Right side, n' = +x: Jz = +Hy averaged onto x = i1.
    add_side(
        xn[i1], yn[jr], 1.0, 0.0, side_weights(j1 - j0 + 1, grid.dy),
        0.5 * (dft_hy[:, i1 - 1, jr] + dft_hy[:, i1, jr]),
        dft_ez[:, i1, jr],
    )
    # Left side, n' = -x: Jz = -Hy averaged onto x = i0.
    add_side(
        xn[i0], yn[jr], -1.0, 0.0, side_weights(j1 - j0 + 1, grid.dy),
        -0.5 * (dft_hy[:, i0 - 1, jr] + dft_hy[:, i0, jr]),
        dft_ez[:, i0, jr],
    )
    # Top side, n' = +y: Jz = -Hx averaged onto y = j1.
    add_side(
        xn[ir], yn[j1], 0.0, 1.0, side_weights(i1 - i0 + 1, grid.dx),
        -0.5 * (dft_hx[:, ir, j1 - 1] + dft_hx[:, ir, j1]),
        dft_ez[:, ir, j1],
    )
    # Bottom side, n' = -y: Jz = +Hx averaged onto y = j0.
    add_side(
        xn[ir], yn[j0], 0.0, -1.0, side_weights(i1 - i0 + 1, grid.dx),
        0.5 * (dft_hx[:, ir, j0 - 1] + dft_hx[:, ir, j0]),
        dft_ez[:, ir, j0],
    )

    x_all = jnp.asarray(np.concatenate(xs))
    y_all = jnp.asarray(np.concatenate(ys))
    nx_all = jnp.asarray(np.concatenate(nxs))
    ny_all = jnp.asarray(np.concatenate(nys))
    w_all = jnp.asarray(np.concatenate(ws))
    jz_all = jnp.concatenate(jzs, axis=1)  # (n_freq, n_pts)
    ez_all = jnp.concatenate(ezs, axis=1)

    k = jnp.asarray(2.0 * np.pi * freqs / C0)
    cos_a, sin_a = jnp.cos(angles), jnp.sin(angles)
    # k rho' cos(psi) = k (x' cos phi + y' sin phi).
    proj = x_all[None, :] * cos_a[:, None] + y_all[None, :] * sin_a[:, None]
    m_phi_dir = nx_all[None, :] * cos_a[:, None] + ny_all[None, :] * sin_a[:, None]
    phase = jnp.exp(1j * k[:, None, None] * proj[None, :, :])  # (n_freq, n_ang, n_pts)
    integrand = ETA0 * jz_all[:, None, :] - m_phi_dir[None, :, :] * ez_all[:, None, :]
    contour_sum = jnp.sum(w_all * integrand * phase, axis=-1)  # (n_freq, n_ang)
    # E_far = -sqrt(j/(8 pi k)) * k * integral  =  -sqrt(j k / 8 pi) * integral.
    prefactor = -jnp.sqrt(1j * k / (8.0 * jnp.pi))
    return prefactor[:, None] * contour_sum


def radiated_power_2d(e_far, angles):
    """Radiated power per unit length, P' = closed-integral |E_far|^2/(2 eta0) dphi.

    `angles` must sample the full circle (periodic trapezoid weights).
    Returns (n_freq,) [W/m for absolute phasors; for dt-scaled DFT inputs
    the value is consistent with source powers from the same DFT].
    """
    w = _periodic_trapezoid_weights(angles, 2.0 * math.pi)
    return jnp.sum(w * jnp.abs(jnp.asarray(e_far)) ** 2, axis=-1) / (2.0 * ETA0)


def directivity_2d(e_far, angles):
    """2D directivity D(phi) = 2 pi |E_far(phi)|^2 / closed-integral |E_far|^2 dphi.

    Dimensionless (DFT normalization cancels); shape (n_freq, n_angles).
    """
    e2 = jnp.abs(jnp.asarray(e_far)) ** 2
    w = _periodic_trapezoid_weights(angles, 2.0 * math.pi)
    total = jnp.sum(w * e2, axis=-1, keepdims=True)
    return 2.0 * jnp.pi * e2 / total


# ---------------------------------------------------------------------------
# 3D
# ---------------------------------------------------------------------------


def ntff_3d(dft_monitor: DFTMonitor, grid3d: Grid3D, box_margin: int, freqs, thetas, phis):
    """3D near-to-far-field transform on a closed rectangular box.

    The box faces lie on the integer-node planes `box_margin` cells inside
    the grid boundary (choose box_margin > cpml.thickness). Each face is
    sampled at its dy*dz / dx*dz / dx*dy patch centers (a midpoint-rule
    surface integral); the Yee components are averaged onto those centers
    (tangential E: 2-point average, tangential H: 4-point average — the
    half-time-step offset is already handled by the solver's exact-phase
    DFT). With Js = n x H and Ms = -n x E, the radiation vectors

        N(th, ph) = sum Js e^{+jk rhat.r'} dA,   L(th, ph) = sum Ms e^{...} dA

    give the pattern amplitudes (r e^{+jkr} removed; Schneider Ch. 14 /
    Balanis, e^{+j omega t} convention)

        E_theta = -jk/(4 pi) (L_phi + eta0 N_theta),
        E_phi   = +jk/(4 pi) (L_theta - eta0 N_phi).

    Args:
        dft_monitor: `SimResult3D.dft` with all six components.
        grid3d: the 3D Yee grid of the simulation.
        box_margin: box distance from the grid boundary in cells (>= 1; in
            practice >= cpml.thickness + a few cells).
        freqs: (n_freq,) frequencies [Hz] matching the monitor's first axis.
        thetas: (n_theta,) polar angles [rad] from the +z axis.
        phis: (n_phi,) azimuth angles [rad] from the +x axis.

    Returns:
        Complex (n_freq, n_theta, n_phi, 2) array; [..., 0] is E_theta and
        [..., 1] is E_phi. Units are [V/m * s * m] for dt-scaled DFT
        inputs; the phase is referenced to the grid center.
    """
    freqs = np.atleast_1d(np.asarray(freqs, np.float64))
    ex, ey, ez = dft_monitor.ex, dft_monitor.ey, dft_monitor.ez
    hx, hy, hz = dft_monitor.hx, dft_monitor.hy, dft_monitor.hz
    if ex.shape[0] != freqs.shape[0]:
        raise ValueError(
            f"freqs has {freqs.shape[0]} entries but the monitor has {ex.shape[0]} rows"
        )
    thetas = jnp.atleast_1d(jnp.asarray(thetas))
    phis = jnp.atleast_1d(jnp.asarray(phis))

    nx, ny, nz = grid3d.nx, grid3d.ny, grid3d.nz
    dx, dy, dz = grid3d.dx, grid3d.dy, grid3d.dz
    m = int(box_margin)
    if m < 1:
        raise ValueError("box_margin must be >= 1 (H averaging needs one cell)")
    i0, i1 = m, nx - 1 - m
    j0, j1 = m, ny - 1 - m
    k0, k1 = m, nz - 1 - m
    if i1 - i0 < 1 or j1 - j0 < 1 or k1 - k0 < 1:
        raise ValueError(f"box_margin {m} leaves no box on a {nx}x{ny}x{nz} grid")

    # Cell-range / shifted-cell-range slices used by the staggering averages.
    ic, icp = slice(i0, i1), slice(i0 + 1, i1 + 1)
    jc, jcp = slice(j0, j1), slice(j0 + 1, j1 + 1)
    kc, kcp = slice(k0, k1), slice(k0 + 1, k1 + 1)

    # Patch-center / node coordinates relative to the grid center.
    xn = (np.arange(nx) - 0.5 * (nx - 1)) * dx
    yn = (np.arange(ny) - 0.5 * (ny - 1)) * dy
    zn = (np.arange(nz) - 0.5 * (nz - 1)) * dz
    xc = xn[i0:i1] + 0.5 * dx
    yc = yn[j0:j1] + 0.5 * dy
    zc = zn[k0:k1] + 0.5 * dz

    pos_list, w_list, j_list, m_list = [], [], [], []

    def add_face(pos_mesh, area, j_comps, m_comps):
        """Append one face: pos_mesh = (X, Y, Z) arrays, J/M component triples."""
        n_pts = pos_mesh[0].size
        pos_list.append(np.stack([p.reshape(-1) for p in pos_mesh], axis=-1))
        w_list.append(np.full(n_pts, area))
        flat = lambda c: c.reshape(c.shape[0], -1)  # noqa: E731
        j_list.append(jnp.stack([flat(c) for c in j_comps], axis=-1))
        m_list.append(jnp.stack([flat(c) for c in m_comps], axis=-1))

    zero = lambda ref: jnp.zeros_like(ref)  # noqa: E731

    # --- Faces x = const (normal +-x): tangential Ey, Ez, Hy, Hz ---------
    for i, sgn in ((i1, 1.0), (i0, -1.0)):
        ey_f = 0.5 * (ey[:, i, jc, kc] + ey[:, i, jc, kcp])
        ez_f = 0.5 * (ez[:, i, jc, kc] + ez[:, i, jcp, kc])
        hy_f = 0.25 * (
            hy[:, i - 1, jc, kc] + hy[:, i, jc, kc]
            + hy[:, i - 1, jcp, kc] + hy[:, i, jcp, kc]
        )
        hz_f = 0.25 * (
            hz[:, i - 1, jc, kc] + hz[:, i, jc, kc]
            + hz[:, i - 1, jc, kcp] + hz[:, i, jc, kcp]
        )
        yy, zz = np.meshgrid(yc, zc, indexing="ij")
        # J = sgn x_hat x H = sgn (0, -Hz, Hy); M = -sgn x_hat x E = sgn (0, Ez, -Ey).
        add_face(
            (np.full_like(yy, xn[i]), yy, zz), dy * dz,
            (zero(hz_f), -sgn * hz_f, sgn * hy_f),
            (zero(ez_f), sgn * ez_f, -sgn * ey_f),
        )

    # --- Faces y = const (normal +-y): tangential Ex, Ez, Hx, Hz ---------
    for j, sgn in ((j1, 1.0), (j0, -1.0)):
        ex_f = 0.5 * (ex[:, ic, j, kc] + ex[:, ic, j, kcp])
        ez_f = 0.5 * (ez[:, ic, j, kc] + ez[:, icp, j, kc])
        hx_f = 0.25 * (
            hx[:, ic, j - 1, kc] + hx[:, ic, j, kc]
            + hx[:, icp, j - 1, kc] + hx[:, icp, j, kc]
        )
        hz_f = 0.25 * (
            hz[:, ic, j - 1, kc] + hz[:, ic, j, kc]
            + hz[:, ic, j - 1, kcp] + hz[:, ic, j, kcp]
        )
        xx, zz = np.meshgrid(xc, zc, indexing="ij")
        # J = sgn y_hat x H = sgn (Hz, 0, -Hx); M = -sgn y_hat x E = sgn (-Ez, 0, Ex).
        add_face(
            (xx, np.full_like(xx, yn[j]), zz), dx * dz,
            (sgn * hz_f, zero(hz_f), -sgn * hx_f),
            (-sgn * ez_f, zero(ez_f), sgn * ex_f),
        )

    # --- Faces z = const (normal +-z): tangential Ex, Ey, Hx, Hy ---------
    for kk, sgn in ((k1, 1.0), (k0, -1.0)):
        ex_f = 0.5 * (ex[:, ic, jc, kk] + ex[:, ic, jcp, kk])
        ey_f = 0.5 * (ey[:, ic, jc, kk] + ey[:, icp, jc, kk])
        hx_f = 0.25 * (
            hx[:, ic, jc, kk - 1] + hx[:, ic, jc, kk]
            + hx[:, icp, jc, kk - 1] + hx[:, icp, jc, kk]
        )
        hy_f = 0.25 * (
            hy[:, ic, jc, kk - 1] + hy[:, ic, jc, kk]
            + hy[:, ic, jcp, kk - 1] + hy[:, ic, jcp, kk]
        )
        xx, yy = np.meshgrid(xc, yc, indexing="ij")
        # J = sgn z_hat x H = sgn (-Hy, Hx, 0); M = -sgn z_hat x E = sgn (Ey, -Ex, 0).
        add_face(
            (xx, yy, np.full_like(xx, zn[kk])), dx * dy,
            (-sgn * hy_f, sgn * hx_f, zero(hx_f)),
            (sgn * ey_f, -sgn * ex_f, zero(ex_f)),
        )

    pos = jnp.asarray(np.concatenate(pos_list))  # (n_pts, 3)
    w = jnp.asarray(np.concatenate(w_list))  # (n_pts,)
    j_all = jnp.concatenate(j_list, axis=1)  # (n_freq, n_pts, 3)
    m_all = jnp.concatenate(m_list, axis=1)

    k_wave = jnp.asarray(2.0 * np.pi * freqs / C0)  # (n_freq,)
    coef = 1j * k_wave / (4.0 * jnp.pi)

    def one_direction(ct, st, cp, sp):
        rhat = jnp.stack([st * cp, st * sp, ct])
        # Quadrature-weighted phase e^{+jk rhat.r'} dA per frequency/point.
        ph = w * jnp.exp(1j * k_wave[:, None] * (pos @ rhat)[None, :])
        nvec = jnp.einsum("fp,fpc->fc", ph, j_all)  # radiation vector N
        lvec = jnp.einsum("fp,fpc->fc", ph, m_all)  # radiation vector L
        n_th = (nvec[:, 0] * cp + nvec[:, 1] * sp) * ct - nvec[:, 2] * st
        n_ph = -nvec[:, 0] * sp + nvec[:, 1] * cp
        l_th = (lvec[:, 0] * cp + lvec[:, 1] * sp) * ct - lvec[:, 2] * st
        l_ph = -lvec[:, 0] * sp + lvec[:, 1] * cp
        e_th = -coef * (l_ph + ETA0 * n_th)
        e_ph = coef * (l_th - ETA0 * n_ph)
        return jnp.stack([e_th, e_ph], axis=-1)  # (n_freq, 2)

    cp, sp = jnp.cos(phis), jnp.sin(phis)

    def one_theta(args):
        ct, st = args
        return jax.vmap(one_direction, in_axes=(None, None, 0, 0))(ct, st, cp, sp)

    # lax.map over theta keeps memory at O(n_phi * n_pts) per step.
    out = jax.lax.map(one_theta, (jnp.cos(thetas), jnp.sin(thetas)))
    return jnp.transpose(out, (2, 0, 1, 3))  # (n_freq, n_theta, n_phi, 2)


def _intensity(e_theta, e_phi):
    return (jnp.abs(jnp.asarray(e_theta)) ** 2 + jnp.abs(jnp.asarray(e_phi)) ** 2) / (
        2.0 * ETA0
    )


def radiated_power_3d(e_theta, e_phi, thetas, phis):
    """Total radiated power P = closed-integral U(th, ph) sin(th) dth dph.

    U = (|E_theta|^2 + |E_phi|^2) / (2 eta0) is the radiation intensity of
    the pattern amplitudes. Quadrature: sin(theta)-weighted trapezoid in
    theta (thetas should span [0, pi]) and periodic trapezoid in phi (phis
    must sample the full circle). Returns (n_freq,).
    """
    thetas = jnp.asarray(thetas)
    u = _intensity(e_theta, e_phi)  # (n_freq, n_theta, n_phi)
    u_phi = jnp.sum(u * _periodic_trapezoid_weights(phis, 2.0 * math.pi), axis=-1)
    w_th = _trapezoid_weights(thetas) * jnp.sin(thetas)
    return jnp.sum(u_phi * w_th, axis=-1)


def directivity_3d(e_theta, e_phi, thetas, phis):
    """Directivity D(th, ph) = 4 pi U(th, ph) / P_rad.

    Dimensionless (DFT normalization cancels); shape (n_freq, n_theta, n_phi).
    """
    u = _intensity(e_theta, e_phi)
    p_rad = radiated_power_3d(e_theta, e_phi, thetas, phis)
    return 4.0 * jnp.pi * u / p_rad[:, None, None]


def gain(directivity, p_rad, p_in):
    """Gain G = e_rad * D with the radiation efficiency e_rad = P_rad / P_in.

    `p_rad` and `p_in` must use the same DFT normalization (the dt-scaled
    spectra of this package cancel in the ratio). With P_in the accepted
    port power this is the IEEE gain; with the incident power it is the
    realized gain. `p_rad`/`p_in` are broadcast over the trailing
    (angle) axes of `directivity`.
    """
    d = jnp.asarray(directivity)
    eff = jnp.asarray(p_rad) / jnp.asarray(p_in)
    eff = jnp.reshape(eff, jnp.shape(eff) + (1,) * (d.ndim - eff.ndim))
    return d * eff
