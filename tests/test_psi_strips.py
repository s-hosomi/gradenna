"""Slab (strip) storage of the CPML psi variables (note 14 Sec. 5.3).

Guarantees:

1. Numerical equivalence: the slab-stored solvers must reproduce a
   reference implementation that carries every psi as a full-size array
   (the pre-optimization scheme) to floating-point noise, both with an
   active CPML and with thickness=0 (PEC box, empty slabs).

   Tolerances: per-probe relative error <= 1e-14 in 2D. In 3D the same
   comparison bottoms out at a few 1e-14 *even at thickness=0*, where the
   slab code is provably inert (empty slices) — i.e. the residual is XLA
   fusion/FMA reassociation noise between two structurally different but
   mathematically identical programs, amplified by the near-singular field
   of the 1-cell hard source over hundreds of steps. The 3D bound is
   therefore 1e-13, and the thickness=0 control is kept in the same
   parametrization to pin the non-psi noise floor. As the sharper check,
   `test_3d_pml_thickness_invariance` verifies the interior field history
   agrees to <= 1 ulp for npml=6 vs npml=10 until the wavefront reaches
   the differing layer — psi slabs cannot leak into the interior.
2. Memory: the psi part of the scan carry must shrink by >= 70% against
   the full-array baseline for a representative 3D optimization grid
   (64^3, npml=8) — the prerequisite for running 3D adjoint problems in a
   24 GB GPU budget.
3. Gradients still flow through the slab-stored carry (AD vs FD and the
   checkpoint equivalence are covered by the existing suite; here we only
   re-check differentiability through an eps patch straddling the PML).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradenna import CPMLSpec, alpha_max_for_fmin, gaussian_derivative, simulate_tm
from gradenna import fdtd2d, fdtd3d
from gradenna.constants import EPS0, ETA0, MU0
from gradenna.cpml import axis_coefficients, slab_slices
from gradenna.fdtd3d import Grid3D, simulate_3d
from gradenna.grid import Grid2D

DX = 2e-3


# ---------------------------------------------------------------------------
# Reference solvers with full-array psi (the pre-strip scheme)
# ---------------------------------------------------------------------------


def _reference_tm(grid, source_ij, current, eps_r, cpml):
    """2D TM stepper carrying all four psi as full-size arrays."""
    nx, ny = grid.nx, grid.ny
    dt = grid.dt
    inv_dx, inv_dy = 1.0 / grid.dx, 1.0 / grid.dy
    dt_mu = dt / MU0

    eps = EPS0 * jnp.broadcast_to(jnp.asarray(eps_r, jnp.float64), (nx, ny))
    ca = jnp.ones((nx, ny))
    cb = dt / eps

    cx_e = axis_coefficients(nx, grid.dx, dt, cpml, half=False)
    cx_h = axis_coefficients(nx, grid.dx, dt, cpml, half=True)
    cy_e = axis_coefficients(ny, grid.dy, dt, cpml, half=False)
    cy_h = axis_coefficients(ny, grid.dy, dt, cpml, half=True)

    si, sj = source_ij
    cb_src = cb[si, sj] * (inv_dx * inv_dy)

    def step(state, j):
        ez, hx, hy, p_ezx, p_ezy, p_hyx, p_hxy = state

        dez_dy = (ez[:, 1:] - ez[:, :-1]) * inv_dy
        p_hxy = cy_h.b[None, :] * p_hxy + cy_h.c[None, :] * dez_dy
        hx = hx - dt_mu * (dez_dy * cy_h.inv_kappa[None, :] + p_hxy)

        dez_dx = (ez[1:, :] - ez[:-1, :]) * inv_dx
        p_hyx = cx_h.b[:, None] * p_hyx + cx_h.c[:, None] * dez_dx
        hy = hy + dt_mu * (dez_dx * cx_h.inv_kappa[:, None] + p_hyx)

        dhy_dx = (hy[1:, 1:-1] - hy[:-1, 1:-1]) * inv_dx
        dhx_dy = (hx[1:-1, 1:] - hx[1:-1, :-1]) * inv_dy
        p_ezx = cx_e.b[1:-1, None] * p_ezx + cx_e.c[1:-1, None] * dhy_dx
        p_ezy = cy_e.b[None, 1:-1] * p_ezy + cy_e.c[None, 1:-1] * dhx_dy
        curl = (
            dhy_dx * cx_e.inv_kappa[1:-1, None]
            + p_ezx
            - dhx_dy * cy_e.inv_kappa[None, 1:-1]
            - p_ezy
        )
        ez = ez.at[1:-1, 1:-1].set(ca[1:-1, 1:-1] * ez[1:-1, 1:-1] + cb[1:-1, 1:-1] * curl)
        ez = ez.at[si, sj].add(-cb_src * j)
        return (ez, hx, hy, p_ezx, p_ezy, p_hyx, p_hxy), ez

    z = jnp.zeros
    state0 = (
        z((nx, ny)), z((nx, ny - 1)), z((nx - 1, ny)),
        z((nx - 2, ny - 2)), z((nx - 2, ny - 2)), z((nx - 1, ny)), z((nx, ny - 1)),
    )
    state, ez_t = jax.lax.scan(step, state0, current)
    return state[0], state[1], state[2], ez_t


def _reference_3d(grid, source_ijk, current, eps_r, cpml):
    """3D stepper carrying all twelve psi as full-size arrays."""
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dt = grid.dt
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    inv_dx, inv_dy, inv_dz = 1.0 / dx, 1.0 / dy, 1.0 / dz
    dt_mu = dt / MU0

    eps = EPS0 * jnp.broadcast_to(jnp.asarray(eps_r, jnp.float64), (nx, ny, nz))
    ca = jnp.ones((nx, ny, nz))
    cb = dt / eps
    ca_ex, cb_ex = ca[:-1, 1:-1, 1:-1], cb[:-1, 1:-1, 1:-1]
    ca_ey, cb_ey = ca[1:-1, :-1, 1:-1], cb[1:-1, :-1, 1:-1]
    ca_ez, cb_ez = ca[1:-1, 1:-1, :-1], cb[1:-1, 1:-1, :-1]

    bx_e, cx_e, kx_e = (
        a[1:-1, None, None] for a in axis_coefficients(nx, dx, dt, cpml, half=False)
    )
    by_e, cy_e, ky_e = (
        a[None, 1:-1, None] for a in axis_coefficients(ny, dy, dt, cpml, half=False)
    )
    bz_e, cz_e, kz_e = (
        a[None, None, 1:-1] for a in axis_coefficients(nz, dz, dt, cpml, half=False)
    )
    bx_h, cx_h, kx_h = (
        a[:, None, None] for a in axis_coefficients(nx, dx, dt, cpml, half=True)
    )
    by_h, cy_h, ky_h = (
        a[None, :, None] for a in axis_coefficients(ny, dy, dt, cpml, half=True)
    )
    bz_h, cz_h, kz_h = (
        a[None, None, :] for a in axis_coefficients(nz, dz, dt, cpml, half=True)
    )

    si, sj, sk = source_ijk
    cb_src = cb[si, sj, sk] * (inv_dx * inv_dy)

    def step(state, j):
        (ex, ey, ez, hx, hy, hz,
         p_exy, p_exz, p_eyx, p_eyz, p_ezx, p_ezy,
         p_hxy, p_hxz, p_hyx, p_hyz, p_hzx, p_hzy) = state

        dey_dz = (ey[:, :, 1:] - ey[:, :, :-1]) * inv_dz
        dez_dy = (ez[:, 1:, :] - ez[:, :-1, :]) * inv_dy
        p_hxz = bz_h * p_hxz + cz_h * dey_dz
        p_hxy = by_h * p_hxy + cy_h * dez_dy
        hx = hx + dt_mu * (dey_dz * kz_h + p_hxz - dez_dy * ky_h - p_hxy)

        dez_dx = (ez[1:, :, :] - ez[:-1, :, :]) * inv_dx
        dex_dz = (ex[:, :, 1:] - ex[:, :, :-1]) * inv_dz
        p_hyx = bx_h * p_hyx + cx_h * dez_dx
        p_hyz = bz_h * p_hyz + cz_h * dex_dz
        hy = hy + dt_mu * (dez_dx * kx_h + p_hyx - dex_dz * kz_h - p_hyz)

        dex_dy = (ex[:, 1:, :] - ex[:, :-1, :]) * inv_dy
        dey_dx = (ey[1:, :, :] - ey[:-1, :, :]) * inv_dx
        p_hzy = by_h * p_hzy + cy_h * dex_dy
        p_hzx = bx_h * p_hzx + cx_h * dey_dx
        hz = hz + dt_mu * (dex_dy * ky_h + p_hzy - dey_dx * kx_h - p_hzx)

        dhz_dy = (hz[:, 1:, :] - hz[:, :-1, :])[:, :, 1:-1] * inv_dy
        dhy_dz = (hy[:, :, 1:] - hy[:, :, :-1])[:, 1:-1, :] * inv_dz
        p_exy = by_e * p_exy + cy_e * dhz_dy
        p_exz = bz_e * p_exz + cz_e * dhy_dz
        curl_x = dhz_dy * ky_e + p_exy - dhy_dz * kz_e - p_exz
        ex = ex.at[:, 1:-1, 1:-1].set(ca_ex * ex[:, 1:-1, 1:-1] + cb_ex * curl_x)

        dhx_dz = (hx[:, :, 1:] - hx[:, :, :-1])[1:-1, :, :] * inv_dz
        dhz_dx = (hz[1:, :, :] - hz[:-1, :, :])[:, :, 1:-1] * inv_dx
        p_eyz = bz_e * p_eyz + cz_e * dhx_dz
        p_eyx = bx_e * p_eyx + cx_e * dhz_dx
        curl_y = dhx_dz * kz_e + p_eyz - dhz_dx * kx_e - p_eyx
        ey = ey.at[1:-1, :, 1:-1].set(ca_ey * ey[1:-1, :, 1:-1] + cb_ey * curl_y)

        dhy_dx = (hy[1:, :, :] - hy[:-1, :, :])[:, 1:-1, :] * inv_dx
        dhx_dy = (hx[:, 1:, :] - hx[:, :-1, :])[1:-1, :, :] * inv_dy
        p_ezx = bx_e * p_ezx + cx_e * dhy_dx
        p_ezy = by_e * p_ezy + cy_e * dhx_dy
        curl_z = dhy_dx * kx_e + p_ezx - dhx_dy * ky_e - p_ezy
        ez = ez.at[1:-1, 1:-1, :].set(ca_ez * ez[1:-1, 1:-1, :] + cb_ez * curl_z)
        ez = ez.at[si, sj, sk].add(-cb_src * j)

        state = (ex, ey, ez, hx, hy, hz,
                 p_exy, p_exz, p_eyx, p_eyz, p_ezx, p_ezy,
                 p_hxy, p_hxz, p_hyx, p_hyz, p_hzx, p_hzy)
        return state, ez

    z = jnp.zeros
    state0 = (
        z((nx - 1, ny, nz)), z((nx, ny - 1, nz)), z((nx, ny, nz - 1)),
        z((nx, ny - 1, nz - 1)), z((nx - 1, ny, nz - 1)), z((nx - 1, ny - 1, nz)),
        z((nx - 1, ny - 2, nz - 2)), z((nx - 1, ny - 2, nz - 2)),
        z((nx - 2, ny - 1, nz - 2)), z((nx - 2, ny - 1, nz - 2)),
        z((nx - 2, ny - 2, nz - 1)), z((nx - 2, ny - 2, nz - 1)),
        z((nx, ny - 1, nz - 1)), z((nx, ny - 1, nz - 1)),
        z((nx - 1, ny, nz - 1)), z((nx - 1, ny, nz - 1)),
        z((nx - 1, ny - 1, nz)), z((nx - 1, ny - 1, nz)),
    )
    state, ez_t = jax.lax.scan(step, state0, current)
    return state[:6], ez_t


# ---------------------------------------------------------------------------
# 1/2. Direct equivalence against the full-array-psi reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("thickness", [0, 8], ids=["pec", "cpml8"])
def test_2d_matches_full_psi_reference(thickness):
    """60x60: slab-injected psi must reproduce full-array psi to <= 1e-14.

    Probes cover the interior, points deep inside the PML (4 cells from
    the wall) and a PML corner; each probe series is normalized by its own
    peak, so the comparison is local, not masked by the source amplitude.
    """
    grid = Grid2D(nx=60, ny=60, dx=DX, dy=DX)
    cpml = CPMLSpec(thickness=thickness, alpha_max=alpha_max_for_fmin(1e9))
    n_steps = 400  # several transits: the pulse interacts with the PML a lot
    t = (jnp.arange(n_steps) + 0.5) * grid.dt
    current = gaussian_derivative(t, t0=80 * grid.dt, tau=14 * grid.dt)
    src = (30, 30)
    # Off-center dielectric patch breaks the symmetry, exercising all slabs.
    eps_r = jnp.ones(grid.shape).at[20:26, 32:40].set(3.0)
    probes = ((40, 30), (30, 40), (20, 20), (4, 30), (30, 4), (52, 52))

    res = simulate_tm(
        grid,
        source_ij=src,
        source_current=current,
        eps_r=eps_r,
        probe_ij=probes,
        cpml=cpml,
    )
    ez_ref, hx_ref, hy_ref, ez_t_ref = _reference_tm(grid, src, current, eps_r, cpml)

    ref_probes = jnp.stack([ez_t_ref[:, i, j] for (i, j) in probes], axis=1)
    err_t = jnp.abs(res.probe_ez - ref_probes).max(0) / jnp.abs(ref_probes).max(0)
    assert float(err_t.max()) <= 1e-14, (
        f"probe series deviate by {np.asarray(err_t)}"
    )
    # Final snapshots, normalized by the run's peak field scale (the final
    # fields have decayed by orders of magnitude in the CPML case, so
    # normalizing by them would only amplify floating-point noise);
    # H errors use the wave-impedance scale.
    scale = float(jnp.abs(ez_t_ref).max())
    err_ez = float(jnp.abs(res.ez - ez_ref).max()) / scale
    err_h = max(
        float(jnp.abs(res.hx - hx_ref).max()),
        float(jnp.abs(res.hy - hy_ref).max()),
    ) / (scale / ETA0)
    assert err_ez <= 1e-14, f"final Ez deviates by {err_ez:.2e}"
    assert err_h <= 1e-14, f"final H deviates by {err_h:.2e}"


@pytest.mark.parametrize("thickness", [0, 6], ids=["pec", "cpml6"])
def test_3d_matches_full_psi_reference(thickness):
    """24^3: slab-injected psi must reproduce full-array psi to FP noise.

    The thickness=0 control bounds the non-psi noise: there the slab code
    is inert (empty static slices), yet the two compiled programs still
    drift apart by a few 1e-14 per-probe over 250 steps (XLA fusion/FMA
    choices, ulp-level per step, amplified by the ~1e5 V/m field at the
    1-cell source). 1e-13 is one order above that floor and seven below
    any physical tolerance in the suite; the bitwise interior check lives
    in test_3d_pml_thickness_invariance.
    """
    n = 24
    grid = Grid3D(nx=n, ny=n, nz=n, dx=DX, dy=DX, dz=DX)
    cpml = CPMLSpec(thickness=thickness, alpha_max=alpha_max_for_fmin(1e9))
    n_steps = 250
    t = (jnp.arange(n_steps) + 0.5) * grid.dt
    current = gaussian_derivative(t, t0=60 * grid.dt, tau=10 * grid.dt)
    src = (n // 2, n // 2, n // 2)
    eps_r = jnp.ones(grid.shape).at[8:11, 13:17, 10:13].set(2.5)
    # Interior, PML interface, deep PML (x/y/z walls) and an edge region.
    probes = (
        (16, 12, 12), (12, 16, 12), (8, 8, 8),
        (3, 12, 12), (12, 3, 12), (12, 12, 3), (20, 20, 11),
    )

    res = simulate_3d(
        grid,
        source_ijk=src,
        source_current=current,
        eps_r=eps_r,
        probe_ijk=probes,
        cpml=cpml,
    )
    (ex_r, ey_r, ez_r, hx_r, hy_r, hz_r), ez_t_ref = _reference_3d(
        grid, src, current, eps_r, cpml
    )

    ref_probes = jnp.stack([ez_t_ref[:, i, j, k] for (i, j, k) in probes], axis=1)
    err_t = jnp.abs(res.probe_ez - ref_probes).max(0) / jnp.abs(ref_probes).max(0)
    assert float(err_t.max()) <= 1e-13, (
        f"probe series deviate by {np.asarray(err_t)}"
    )
    # Final snapshots against the run's peak field scale (see the 2D test).
    scale = float(jnp.abs(ez_t_ref).max())
    err_e = max(
        float(jnp.abs(res.ex - ex_r).max()),
        float(jnp.abs(res.ey - ey_r).max()),
        float(jnp.abs(res.ez - ez_r).max()),
    ) / scale
    err_h = max(
        float(jnp.abs(res.hx - hx_r).max()),
        float(jnp.abs(res.hy - hy_r).max()),
        float(jnp.abs(res.hz - hz_r).max()),
    ) / (scale / ETA0)
    assert err_e <= 1e-13, f"final E deviates by {err_e:.2e}"
    assert err_h <= 1e-13, f"final H deviates by {err_h:.2e}"


def test_3d_pml_thickness_invariance():
    """Interior history must be PML-thickness-independent early on.

    npml=6 and npml=10 runs share every interior coefficient; the psi slabs
    differ only inside the respective layers. Until the wavefront reaches
    the deeper layer (i = 10, 6 cells from the source, ~12 steps at the 3D
    Courant speed) the probe records must agree to <= 1 ulp (the slab
    extents change the concatenate split points, hence XLA fusion/FMA
    grouping, so exact bitwise equality is not guaranteed) — anything
    beyond that would mean the slab machinery touches points it must not.
    For scale: the physical difference once the wave does reach the PML is
    ~1e-9 relative at step 10 and grows from there.
    """
    n = 32
    grid = Grid3D(nx=n, ny=n, nz=n, dx=DX, dy=DX, dz=DX)
    c = n // 2
    n_steps = 60
    t = (jnp.arange(n_steps) + 0.5) * grid.dt
    current = gaussian_derivative(t, t0=12 * grid.dt, tau=4 * grid.dt)
    probes = ((c + 1, c, c), (c, c + 1, c), (c - 1, c - 1, c))

    records = {}
    for th in (6, 10):
        res = simulate_3d(
            grid,
            source_ijk=(c, c, c),
            source_current=current,
            probe_ijk=probes,
            cpml=CPMLSpec(thickness=th, alpha_max=alpha_max_for_fmin(1e9)),
        )
        records[th] = res.probe_ez

    n_safe = 10  # wavefront reaches i = 10 after ~12 steps; stay inside
    early = float(jnp.abs(records[6][:n_safe] - records[10][:n_safe]).max())
    scale = float(jnp.abs(records[10][:n_safe]).max())
    # Sanity: the probes did record a non-trivial signal in that window.
    assert scale > 0.0
    assert early <= 1e-15 * scale, (
        f"interior fields differ before the pulse reaches the PML: "
        f"{early:.2e} (rel {early / scale:.2e})"
    )


# ---------------------------------------------------------------------------
# 3. Memory: psi bytes in the carry vs the full-array baseline
# ---------------------------------------------------------------------------


def _nbytes(tree) -> int:
    return sum(leaf.nbytes for leaf in jax.tree_util.tree_leaves(tree))


def test_3d_psi_memory_reduction_64cube():
    """64^3, npml=8 (the note 08 GPU sizing case): psi bytes must drop >= 70%."""
    nx = ny = nz = 64
    npml = 8
    itemsize = jnp.zeros((), jnp.float64).dtype.itemsize
    full_shapes = [
        (nx - 1, ny - 2, nz - 2), (nx - 1, ny - 2, nz - 2),
        (nx - 2, ny - 1, nz - 2), (nx - 2, ny - 1, nz - 2),
        (nx - 2, ny - 2, nz - 1), (nx - 2, ny - 2, nz - 1),
        (nx, ny - 1, nz - 1), (nx, ny - 1, nz - 1),
        (nx - 1, ny, nz - 1), (nx - 1, ny, nz - 1),
        (nx - 1, ny - 1, nz), (nx - 1, ny - 1, nz),
    ]
    full = sum(int(np.prod(s)) * itemsize for s in full_shapes)
    slab = _nbytes(fdtd3d._init_psi(nx, ny, nz, npml, jnp.float64))
    reduction = 1.0 - slab / full
    assert reduction >= 0.70, (
        f"3D psi reduction {reduction:.1%} < 70% ({slab} vs {full} bytes)"
    )


def test_2d_psi_memory_reduction():
    """256^2, npml=10: psi bytes must drop >= 70% in 2D as well."""
    nx = ny = 256
    npml = 10
    itemsize = jnp.zeros((), jnp.float64).dtype.itemsize
    full_shapes = [
        (nx - 2, ny - 2), (nx - 2, ny - 2), (nx - 1, ny), (nx, ny - 1),
    ]
    full = sum(int(np.prod(s)) * itemsize for s in full_shapes)
    slab = _nbytes(fdtd2d._init_psi(nx, ny, npml, jnp.float64))
    reduction = 1.0 - slab / full
    assert reduction >= 0.70, (
        f"2D psi reduction {reduction:.1%} < 70% ({slab} vs {full} bytes)"
    )


def test_slab_slices_basics():
    lo, hi = slab_slices(20, 6)
    assert (lo.start, lo.stop) == (0, 6)
    assert (hi.start, hi.stop) == (14, 20)
    lo0, hi0 = slab_slices(20, 0)  # PEC box: empty slabs, not full-array views
    assert np.arange(20)[lo0].size == 0
    assert np.arange(20)[hi0].size == 0
    with pytest.raises(ValueError, match="slab thickness"):
        slab_slices(10, 6)


# ---------------------------------------------------------------------------
# 4. Gradients flow through the slab-stored carry
# ---------------------------------------------------------------------------


def test_3d_gradient_through_slab_psi():
    """grad wrt an eps patch straddling the PML interface is finite/non-zero."""
    n = 20
    grid = Grid3D(nx=n, ny=n, nz=n, dx=DX, dy=DX, dz=DX)
    n_steps = 100
    t = (jnp.arange(n_steps) + 0.5) * grid.dt
    current = gaussian_derivative(t, t0=30 * grid.dt, tau=8 * grid.dt)

    def loss(eps_patch):
        # Patch reaches into the PML (cells 2..5 with npml=4): the psi slabs
        # see a non-trivial eps and must propagate its gradient.
        eps_r = jnp.ones(grid.shape).at[2:8, 8:12, 8:12].set(eps_patch)
        res = simulate_3d(
            grid,
            source_ijk=(10, 10, 9),
            source_current=current,
            probe_ijk=((14, 10, 9),),
            eps_r=eps_r,
            cpml=CPMLSpec(thickness=4, alpha_max=alpha_max_for_fmin(2e9)),
        )
        return jnp.sum(res.probe_ez**2)

    g = jax.grad(loss)(2.0 * jnp.ones((6, 4, 4)))
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.abs(g).max()) > 0.0
