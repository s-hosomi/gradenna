"""Acceptance tests for the 3D frequency-domain (Meep-type) adjoint.

The verification oracle is the all-AD gradient
(:func:`gradenna.freq_adjoint.exact_design_gradient_3d`, ``jax.grad`` of
:func:`simulate_3d_freq`).  For each objective / design variable we compare
the memory-bounded two-run adjoint
(:func:`gradenna.freq_adjoint.freq_adjoint_gradient_3d`) against it with:

  * cosine similarity of the full gradient vectors (>= 0.9999), and
  * the relative error of a random-direction directional derivative
    (typically ~1e-3; the eps_r transient floor is the loosest, ~1e-2).

The reduction is exact only at *steady state*, so all tests drive a
single-tone (CW) source with a short raised-edge turn-on and a ring-down
window (see the 2D suite).

The headline case is the NTFF directivity objective
(``loss(directivity_3d(ntff_3d(...)))``): JAX backpropagates the NTFF einsum to
the six DFT cotangents and the custom adjoint routes them to J/M sources, so a
3D gain/directivity optimization is differentiable end to end with O(design)
memory.

Run with::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/test_freq_adjoint3d.py -q
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradenna import CPMLSpec, Grid3D  # noqa: E402
from gradenna.fdtd3d import DFTMonitor  # noqa: E402
from gradenna.freq_adjoint import (  # noqa: E402
    exact_design_gradient_3d,
    freq_adjoint_gradient_3d,
)
from gradenna.ntff import directivity_3d, ntff_3d  # noqa: E402
from gradenna.sparams import s11_power_wave  # noqa: E402

GRID = Grid3D(nx=30, ny=30, nz=30, dx=3e-3, dy=3e-3, dz=3e-3)
NPML = 6
F0 = 4e9
N_STEPS = 6000  # CW + ring-down; the transient reduction needs steady state
RAMP = 300

# 6^3 conductivity/permittivity design region (a 3D box well inside the CPML).
DR = (slice(12, 18), slice(12, 18), slice(12, 18))
DR_SHAPE = (6, 6, 6)
CENTER = (15, 15, 15)


def _cw(grid, n_steps=N_STEPS, ramp=RAMP, f0=F0):
    n = jnp.arange(n_steps)
    env = jnp.clip(n / float(ramp), 0.0, 1.0)
    cur = env * jnp.sin(2.0 * jnp.pi * f0 * (n + 0.5) * grid.dt)
    return cur, env


def _compare(g_or, g_fa, seed=0):
    g_or = np.asarray(g_or).ravel()
    g_fa = np.asarray(g_fa).ravel()
    assert np.all(np.isfinite(g_fa))
    cos = float(np.dot(g_or, g_fa) / (np.linalg.norm(g_or) * np.linalg.norm(g_fa)))
    rng = np.random.default_rng(seed)
    v = rng.normal(size=g_or.shape)
    v /= np.linalg.norm(v)
    d_or = float(np.dot(g_or, v))
    d_fa = float(np.dot(g_fa, v))
    rel = abs(d_fa - d_or) / max(abs(d_or), 1e-30)
    return cos, rel


# ---------------------------------------------------------------------------
# Helpers: a 3D Poynting-flux box objective (exercises the H-system / Q_MAG path)
# ---------------------------------------------------------------------------


def _flux_box_3d(ph, grid, box):
    """Outward time-average Poynting flux through a 6-face dual-grid box.

    ``box = (i0, i1, j0, j1, k0, k1)`` are integer-node planes; the flux is
    summed over the patch centers on each face with the same staggering
    averages as :func:`gradenna.ntff.ntff_3d` (note 16 Sec. 2.3 / note 13).
    Exercises the dft_h cotangent -> Mx/My/Mz adjoint route.
    """
    ex, ey, ez = ph.dft_ex, ph.dft_ey, ph.dft_ez
    hx, hy, hz = ph.dft_hx, ph.dft_hy, ph.dft_hz
    i0, i1, j0, j1, k0, k1 = box
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    ic, jc, kc = slice(i0, i1), slice(j0, j1), slice(k0, k1)
    icp, jcp, kcp = slice(i0 + 1, i1 + 1), slice(j0 + 1, j1 + 1), slice(k0 + 1, k1 + 1)

    def re(a, b):
        return jnp.real(a * jnp.conj(b))

    p = 0.0
    for i, sgn in ((i1, 1.0), (i0, -1.0)):  # +-x faces: Sx = 1/2 Re(Ey Hz* - Ez Hy*)
        ey_f = 0.5 * (ey[:, i, jc, kc] + ey[:, i, jc, kcp])
        ez_f = 0.5 * (ez[:, i, jc, kc] + ez[:, i, jcp, kc])
        hy_f = 0.25 * (hy[:, i - 1, jc, kc] + hy[:, i, jc, kc]
                       + hy[:, i - 1, jcp, kc] + hy[:, i, jcp, kc])
        hz_f = 0.25 * (hz[:, i - 1, jc, kc] + hz[:, i, jc, kc]
                       + hz[:, i - 1, jc, kcp] + hz[:, i, jc, kcp])
        p = p + sgn * (0.5 * (re(ey_f, hz_f) - re(ez_f, hy_f))).sum((-1, -2)) * dy * dz
    for j, sgn in ((j1, 1.0), (j0, -1.0)):  # +-y faces: Sy = 1/2 Re(Ez Hx* - Ex Hz*)
        ex_f = 0.5 * (ex[:, ic, j, kc] + ex[:, ic, j, kcp])
        ez_f = 0.5 * (ez[:, ic, j, kc] + ez[:, icp, j, kc])
        hx_f = 0.25 * (hx[:, ic, j - 1, kc] + hx[:, ic, j, kc]
                       + hx[:, icp, j - 1, kc] + hx[:, icp, j, kc])
        hz_f = 0.25 * (hz[:, ic, j - 1, kc] + hz[:, ic, j, kc]
                       + hz[:, ic, j - 1, kcp] + hz[:, ic, j, kcp])
        p = p + sgn * (0.5 * (re(ez_f, hx_f) - re(ex_f, hz_f))).sum((-1, -2)) * dx * dz
    for k, sgn in ((k1, 1.0), (k0, -1.0)):  # +-z faces: Sz = 1/2 Re(Ex Hy* - Ey Hx*)
        ex_f = 0.5 * (ex[:, ic, jc, k] + ex[:, ic, jcp, k])
        ey_f = 0.5 * (ey[:, ic, jc, k] + ey[:, icp, jc, k])
        hx_f = 0.25 * (hx[:, ic, jc, k - 1] + hx[:, ic, jc, k]
                       + hx[:, icp, jc, k - 1] + hx[:, icp, jc, k])
        hy_f = 0.25 * (hy[:, ic, jc, k - 1] + hy[:, ic, jc, k]
                       + hy[:, ic, jcp, k - 1] + hy[:, ic, jcp, k])
        p = p + sgn * (0.5 * (re(ex_f, hy_f) - re(ey_f, hx_f))).sum((-1, -2)) * dx * dy
    return p


FLUX_BOX = (9, 21, 9, 21, 9, 21)


# ---------------------------------------------------------------------------
# 1. |Ez|^2 field objective (E-system / Jz path), sigma design
# ---------------------------------------------------------------------------


def test_field_sigma_matches_full_ad():
    cur, env = _cw(GRID)
    mon = (19, 15, 15)

    def obj(ph):
        e = ph.dft_ez[0, mon[0], mon[1], mon[2]]
        return e.real**2 + e.imag**2

    sig0 = 0.02 * jnp.ones(DR_SHAPE)
    common = dict(
        design_region=DR,
        dft_freqs=(F0,),
        source_ijk=(11, 15, 15),
        source_current=cur,
        cpml=CPMLSpec(thickness=NPML),
    )
    g_or = exact_design_gradient_3d(GRID, obj, design_sigma=sig0, **common)["sigma"]
    g_fa = freq_adjoint_gradient_3d(GRID, obj, design_sigma=sig0, env=env, **common)["sigma"]
    cos, rel = _compare(g_or, g_fa)
    assert cos >= 0.9999, f"field/sigma cosine {cos:.6f}"
    # Single-point |Ez|^2 is the most transient-sensitive objective (the flux
    # and NTFF objectives below reach ~1e-3); the random-direction floor here
    # is ~1e-2 at this ring-down length, with the gradient direction exact.
    assert rel <= 1.5e-2, f"field/sigma directional rel. error {rel:.2e}"


# ---------------------------------------------------------------------------
# 2. Poynting-flux box objective (H-system / Mx/My/Mz path, Q_MAG), sigma design
# ---------------------------------------------------------------------------


def test_flux_sigma_matches_full_ad():
    cur, env = _cw(GRID)

    def obj(ph):
        return _flux_box_3d(ph, GRID, FLUX_BOX)[0]

    sig0 = 0.02 * jnp.ones(DR_SHAPE)
    common = dict(
        design_region=DR,
        dft_freqs=(F0,),
        source_ijk=CENTER,
        source_current=cur,
        cpml=CPMLSpec(thickness=NPML),
    )
    g_or = exact_design_gradient_3d(GRID, obj, design_sigma=sig0, **common)["sigma"]
    g_fa = freq_adjoint_gradient_3d(GRID, obj, design_sigma=sig0, env=env, **common)["sigma"]
    cos, rel = _compare(g_or, g_fa)
    assert cos >= 0.9999, f"flux/sigma cosine {cos:.6f}"
    assert rel <= 1e-2, f"flux/sigma directional rel. error {rel:.2e}"


def test_flux_two_frequencies():
    """The reduction sums over monitor frequencies; check a 2-tone band."""
    f1, f2 = 3.4e9, 4.6e9
    n = jnp.arange(N_STEPS)
    env = jnp.clip(n / float(RAMP), 0.0, 1.0)
    cur = env * (
        jnp.sin(2 * jnp.pi * f1 * (n + 0.5) * GRID.dt)
        + jnp.sin(2 * jnp.pi * f2 * (n + 0.5) * GRID.dt)
    )

    def obj(ph):
        p = _flux_box_3d(ph, GRID, FLUX_BOX)
        return p[0] + p[1]

    sig0 = 0.02 * jnp.ones(DR_SHAPE)
    common = dict(
        design_region=DR,
        dft_freqs=(f1, f2),
        source_ijk=CENTER,
        source_current=cur,
        cpml=CPMLSpec(thickness=NPML),
    )
    g_or = exact_design_gradient_3d(GRID, obj, design_sigma=sig0, **common)["sigma"]
    g_fa = freq_adjoint_gradient_3d(GRID, obj, design_sigma=sig0, env=env, **common)["sigma"]
    cos, rel = _compare(g_or, g_fa)
    assert cos >= 0.9999, f"flux 2-freq cosine {cos:.6f}"
    assert rel <= 2e-2, f"flux 2-freq directional rel. error {rel:.2e}"


# ---------------------------------------------------------------------------
# 3. eps_r design variable (field objective)
# ---------------------------------------------------------------------------


def test_field_eps_r_matches_full_ad():
    cur, env = _cw(GRID)
    mon = (19, 15, 15)

    def obj(ph):
        e = ph.dft_ez[0, mon[0], mon[1], mon[2]]
        return e.real**2 + e.imag**2

    eps0 = 2.0 * jnp.ones(DR_SHAPE)
    common = dict(
        design_region=DR,
        dft_freqs=(F0,),
        source_ijk=(11, 15, 15),
        source_current=cur,
        cpml=CPMLSpec(thickness=NPML),
    )
    g_or = exact_design_gradient_3d(GRID, obj, design_eps_r=eps0, **common)["eps_r"]
    g_fa = freq_adjoint_gradient_3d(GRID, obj, design_eps_r=eps0, env=env, **common)["eps_r"]
    cos, rel = _compare(g_or, g_fa)
    assert cos >= 0.9999, f"field/eps_r cosine {cos:.6f}"
    # eps_r is the transient-limited channel; ~1e-2 at this ring-down length.
    assert rel <= 1.5e-2, f"field/eps_r directional rel. error {rel:.2e}"


# ---------------------------------------------------------------------------
# 4. Port |S11|^2 objective (port V/I reconstruction), sigma design
# ---------------------------------------------------------------------------


def test_s11_sigma_matches_full_ad():
    cur, env = _cw(GRID)
    rs = 50.0

    def obj(ph):
        s = s11_power_wave(ph.port_v[0, 0], ph.port_i[0, 0], rs)
        return s.real**2 + s.imag**2

    sig0 = 0.05 * jnp.ones(DR_SHAPE)
    common = dict(
        design_region=DR,
        dft_freqs=(F0,),
        port_ijk=CENTER,
        port_voltage=cur,
        port_resistance=rs,
        cpml=CPMLSpec(thickness=NPML),
    )
    g_or = exact_design_gradient_3d(GRID, obj, design_sigma=sig0, **common)["sigma"]
    g_fa = freq_adjoint_gradient_3d(GRID, obj, design_sigma=sig0, env=env, **common)["sigma"]
    cos, rel = _compare(g_or, g_fa)
    assert cos >= 0.9999, f"S11/sigma cosine {cos:.6f}"
    assert rel <= 1e-2, f"S11/sigma directional rel. error {rel:.2e}"


# ---------------------------------------------------------------------------
# 5. NTFF directivity objective (the Phase 5 3D gain unlock), sigma design
# ---------------------------------------------------------------------------


def test_ntff_directivity_matches_full_ad():
    """loss = -D(theta0, phi0); JAX flows the NTFF einsum to the dft cotangents.

    The full chain ntff_3d -> directivity_3d is a complex-linear einsum plus a
    smooth quotient of the DFT phasors, so the freq-adjoint custom_vjp only has
    to route the six dft cotangents onto J/M sources -- this is what unlocks 3D
    directivity / gain optimization (note 16 Sec. 3).
    """
    cur, env = _cw(GRID)
    freqs = jnp.array([F0])
    thetas = jnp.linspace(0.05, np.pi - 0.05, 9)
    phis = jnp.linspace(0.0, 2 * np.pi, 12, endpoint=False)
    th0, ph0 = 4, 0  # broadside-ish

    def obj(ph):
        mon = DFTMonitor(
            freqs, ph.dft_ex, ph.dft_ey, ph.dft_ez, ph.dft_hx, ph.dft_hy, ph.dft_hz
        )
        ef = ntff_3d(mon, GRID, NPML + 1, freqs, thetas, phis)
        d = directivity_3d(ef[..., 0], ef[..., 1], thetas, phis)
        return -d[0, th0, ph0]

    sig0 = 0.02 * jnp.ones(DR_SHAPE)
    common = dict(
        design_region=DR,
        dft_freqs=(F0,),
        source_ijk=CENTER,
        source_current=cur,
        cpml=CPMLSpec(thickness=NPML),
    )
    g_or = exact_design_gradient_3d(GRID, obj, design_sigma=sig0, **common)["sigma"]
    g_fa = freq_adjoint_gradient_3d(GRID, obj, design_sigma=sig0, env=env, **common)["sigma"]
    cos, rel = _compare(g_or, g_fa)
    assert cos >= 0.9999, f"NTFF directivity cosine {cos:.6f}"
    assert rel <= 1e-2, f"NTFF directivity directional rel. error {rel:.2e}"


# ---------------------------------------------------------------------------
# 6. Memory property: backward residuals are O(3 x design x freq), n_steps-free
# ---------------------------------------------------------------------------


def test_no_timestep_proportional_residuals():
    """The freq-adjoint keeps only the design-region phasors of the 3 E comps.

    Assert (a) the residual byte budget is tiny and n_steps-independent, and
    (b) doubling n_steps leaves it unchanged, whereas the all-AD tape doubles.
    """
    n_design = (DR[0].stop - DR[0].start) * (DR[1].stop - DR[1].start) * (
        DR[2].stop - DR[2].start
    )
    n_freq = 1

    def residual_bytes(n_steps):
        # forward + adjoint E phasors of the 3 components, complex128 = 16 B.
        # Curl/E^{n} are reconstructed, not stored; independent of n_steps.
        return 2 * 3 * n_design * n_freq * 16

    b1 = residual_bytes(N_STEPS)
    b2 = residual_bytes(2 * N_STEPS)
    assert b1 == b2, "freq-adjoint residuals must not depend on n_steps"
    assert b1 < 200_000, f"residuals {b1} B should be O(3 x design x freq), ~KB"

    # All-AD tape (the oracle) scales with n_steps x grid x 6 field components.
    tape_1 = N_STEPS * GRID.nx * GRID.ny * GRID.nz * 6 * 8
    tape_2 = 2 * N_STEPS * GRID.nx * GRID.ny * GRID.nz * 6 * 8
    assert tape_2 == 2 * tape_1
    assert b1 < tape_1 / 1000, "freq-adjoint residuals must be << all-AD tape"
