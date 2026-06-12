r"""Meep-type frequency-domain adjoint for the 2D TM FDTD solver.

This module computes the gradient of a *frequency-domain* objective (a
function of the steady-state running-DFT phasors recorded by
:func:`gradenna.simulate_tm`) with respect to the design conductivity
``sigma`` and/or relative permittivity ``eps_r``, using **two forward
time-stepping runs** (one forward, one adjoint) and a backward-pass memory
footprint of only ``O(design cells x frequencies)`` -- independent of the
number of time steps.  This is the strategy of
docs/research/04-time-reversal-lossy.md (Strategy G) and
docs/research/09-adjoint-theory.md (Sec. 5.2), with the all-AD
``jax.grad(simulate_tm)`` as a machine-precision verification oracle
(:func:`exact_design_gradient`).

Excitation model
================
The objective phasors are taken at **steady state**: drive the structure
with a *single-tone* (CW) source per design frequency under a smooth
turn-on envelope ``env[n]`` (the gradient is exact only once the forward
field is steady and the adjoint has rung up, so a CW + ring-down window is
required -- a broadband pulse leaves a transient that the single-frequency
DFT cannot resolve, see "Limitations").  ``N_eff = sum_n env[n]^2`` is the
effective number of steady samples and sets the Parseval normalization.

Derivation of the discrete gradient (all conventions verified to
``cos = 1`` and scale ``-> 1`` against ``jax.grad`` of ``simulate_tm``)
======================================================================

Forward interior Ez update (one cell ``i``; the solver's exact-sample-time
running DFT uses ``(n+1) dt`` for Ez, ``(n+1/2) dt`` for H):

.. math::

    E_i^{n+1} = c_{a,i} E_i^{n} + c_{b,i}\,\mathrm{curl}_i^{n+1},\quad
    c_a=\frac{1-h}{1+h},\ c_b=\frac{\Delta t/\varepsilon}{1+h},\
    h=\frac{\sigma\Delta t}{2\varepsilon},\ \varepsilon=\varepsilon_0\varepsilon_r .

``sigma``/``eps_r`` enter only through ``c_a, c_b``.  The exact discrete
adjoint sensitivity (what reverse-mode AD computes, note 09 Sec. 2.2) is

.. math::

    \frac{\partial L}{\partial\theta_i}
    = \sum_n \lambda_i^{n+1}\Big(
        \partial_\theta c_{a,i}\,E_i^{n}
      + \partial_\theta c_{b,i}\,\mathrm{curl}_i^{n+1}\Big),
    \quad \lambda_i^{n+1}=\frac{\partial L}{\partial E_i^{n+1}},

with the closed-form coefficients (evaluated at the operating point)

.. math::

    \partial_\sigma c_a=\frac{-\Delta t/\varepsilon}{(1+h)^2},\quad
    \partial_\sigma c_b=\frac{-(\Delta t/\varepsilon)(\Delta t/2\varepsilon)}{(1+h)^2},

(``eps_r`` analogues obtained by :func:`jax.jacfwd`, see :func:`_dcoef`).
This time-domain form was checked against ``jax.grad`` to ``1e-15``
(lossless) and ``5e-3`` (lossy, the residual being the adjoint medium
feedback + transient).

**Adjoint source (verified exact).**  For a complex phasor cotangent
``g = conj(dL/dE_hat)`` (the value ``jax.grad`` returns for a *real* loss of
a complex phasor) at a monitor cell, the adjoint source is the CW waveform

.. math::

    a^{n} = \mathrm{env}[n]\,\mathrm{Re}\!\big[g\,e^{+i\omega(n+1)\Delta t}\big],

injected as a **Jz line current** ``J = a/(-c_{b,\mathrm{src}})``
(``c_{b,src}=(\Delta t/\varepsilon_0)/(dx\,dy)`` reproduces a unit additive
Ez increment).  H-system (flux) cotangents become **magnetic currents**
``Mx, My`` on the H update (the new ``magnetic_current_sources`` hook of
:func:`simulate_tm`), with the H sample phase ``(n+1/2)\Delta t`` and a unit
additive H injection ``M = a/(-\Delta t/\mu_0)`` scaled by the universal
Yee transpose constant :data:`Q_MAG` (the E<->H staggering factor that
couples an H cotangent into the adjoint Ez phasor; dx/frequency
independent, fitted once to ``cos = 1``).

**Frequency reduction (the memory win, exact at steady state).**  Because
the adjoint field is a pure tone, the exact pure-tone Parseval identity
``sum_n a[n] Re[B e^{i w (n+1)dt}] = (2/(N_eff dt^2)) Re[A_hat B_hat]``
(both dt-scaled running DFTs) collapses the time sum to a per-frequency
product of the **forward** design phasors and the **adjoint** design phasor:

.. math::

    \boxed{\;
    \frac{\partial L}{\partial\theta_i}
    = \sum_k \frac{2}{N_{\mathrm{eff}}\,\Delta t}\,
      \mathrm{Re}\!\big[\overline{\hat\Lambda_{k,i}}\;\overline{\hat G_{k,i}}\big],
    \quad
    \hat G_{k,i}=\partial_\theta c_a\,\hat E^{(n)}_{k,i}
                +\partial_\theta c_b\,\hat C_{k,i}\; }

where :math:`\hat E^{(n)}=\hat E/z`, :math:`\hat C=(\hat E-c_a\hat E^{(n)})/c_b`,
:math:`z=e^{i\omega\Delta t}` (the exact discrete relation between the Ez^{n}
and Ez^{n+1} steady phasors), and :math:`\hat\Lambda` is the adjoint Ez
running-DFT phasor on the design region.  The residuals kept for the
backward pass are only these design-region phasors -- ``O(N_design x
N_freq)`` complex numbers, with **no tape proportional to n_steps**.

Limitations
===========
The single approximation is the *transient*: the band-limited (single
DFT bin) assumption is exact only for a pure tone over the whole window.
With a CW excitation and adequate turn-on + ring-down the residual is
``<~ 5e-3`` (see ``tests/test_freq_adjoint.py``); a broadband pulse leaves
``O(10%)`` transient error and is *not* supported by this reduction (use
:func:`exact_design_gradient`, the all-AD path, for pulse objectives).

3D extension path
=================
Dimension-agnostic: record design-region DFTs of the three E components and
the curl driving them, inject Jz/Mx/My adjoint sources, and form the same
per-frequency ``Re[conj(Lambda) conj(G)]`` contraction.  ``fdtd3d`` is
intentionally untouched; the 2D mechanism (and the
``magnetic_current_sources`` hook) is the template to port.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gradenna.constants import EPS0, MU0
from gradenna.cpml import CPMLSpec
from gradenna.fdtd2d import simulate_tm
from gradenna.fdtd3d import simulate_3d, time_series_dft
from gradenna.grid import Grid2D, Grid3D

__all__ = [
    "FreqPhasors",
    "FreqPhasors3D",
    "Q_MAG",
    "simulate_tm_freq",
    "freq_adjoint_gradient",
    "exact_design_gradient",
    "simulate_3d_freq",
    "freq_adjoint_gradient_3d",
    "exact_design_gradient_3d",
]

#: Universal Yee transpose constant coupling a magnetic-current (H) cotangent
#: into the adjoint Ez running-DFT phasor used by the gradient contraction.
#: Closed form Q_MAG = -eps0/mu0 = -1/eta0^2 (the negative squared
#: free-space admittance): on the unweighted Euclidean inner product that
#: reverse-mode AD uses, this is the Yee symplectic metric ratio that
#: converts an H-array adjoint into an E-array adjoint (strip mu0, apply
#: eps0). Derived and verified grid-/Courant-/frequency-independent in
#: docs/research/16 (the earlier empirical fit -7.025e-6 was a slightly
#: biased estimate of the same constant); applied on the *unit-additive*
#: H injection basis ``M = a/(-dt/mu0)``.
Q_MAG = -EPS0 / MU0


class FreqPhasors(NamedTuple):
    """Frequency-domain phasors returned by :func:`simulate_tm_freq`.

    dft_ez/dft_hx/dft_hy: full-grid running-DFT phasors (same convention as
        :class:`gradenna.SimResult`), shape ``(n_freq, ...)``.
    port_v/port_i: port voltage/current phasors, shape ``(n_freq, n_ports)``
        or ``None`` if no ports were given.
    """

    dft_ez: jnp.ndarray
    dft_hx: jnp.ndarray
    dft_hy: jnp.ndarray
    port_v: jnp.ndarray | None = None
    port_i: jnp.ndarray | None = None


# ---------------------------------------------------------------------------
# Coefficient sensitivities  d c_a / d theta, d c_b / d theta
# ---------------------------------------------------------------------------


def _ca_cb(eps_r, sigma, dt):
    eps = EPS0 * eps_r
    h = sigma * dt / (2.0 * eps)
    ca = (1.0 - h) / (1.0 + h)
    cb = (dt / eps) / (1.0 + h)
    return ca, cb


def _dcoef_closed(eps_d, sig_d, dt):
    r"""Closed-form ``d c_a, c_b`` sensitivities at the operating point.

    Returns ``(dca_dsig, dcb_dsig, dca_deps, dcb_deps)`` (the last two w.r.t.
    physical ``eps``; multiply by ``EPS0`` for ``eps_r``).  Verified to
    machine precision against :func:`jax.jacfwd` of :func:`_ca_cb`.

    .. math::
        \partial_\sigma c_a = \frac{-\Delta t/\varepsilon}{(1+h)^2},\quad
        \partial_\sigma c_b = \frac{-(\Delta t/\varepsilon)(\Delta t/2\varepsilon)}{(1+h)^2},\\
        \partial_\varepsilon c_a = \frac{\sigma\Delta t/\varepsilon^2}{(1+h)^2},\quad
        \partial_\varepsilon c_b = \frac{-\Delta t/\varepsilon^2}{(1+h)^2},
        \quad h=\frac{\sigma\Delta t}{2\varepsilon}.
    """
    h = sig_d * dt / (2.0 * eps_d)
    denom = (1.0 + h) ** 2
    dca_dsig = (-dt / eps_d) / denom
    dcb_dsig = (-(dt / eps_d) * (dt / (2.0 * eps_d))) / denom
    dca_deps = (sig_d * dt / eps_d**2) / denom
    dcb_deps = (-(dt / eps_d**2)) / denom
    return dca_dsig, dcb_dsig, dca_deps, dcb_deps


# ---------------------------------------------------------------------------
# Forward primitive
# ---------------------------------------------------------------------------


def _full_eps_sigma(grid, design_sigma, design_eps_r, design_region):
    eps_r = jnp.ones(grid.shape)
    sigma = jnp.zeros(grid.shape)
    if design_eps_r is not None:
        eps_r = eps_r.at[design_region].set(design_eps_r)
    if design_sigma is not None:
        sigma = sigma.at[design_region].set(design_sigma)
    return eps_r, sigma


def simulate_tm_freq(
    grid: Grid2D,
    *,
    design_sigma=None,
    design_eps_r=None,
    design_region,
    dft_freqs,
    source_ij=None,
    source_current=None,
    ports=(),
    cpml: CPMLSpec = CPMLSpec(),
) -> FreqPhasors:
    """Forward run returning frequency-domain phasors (see :class:`FreqPhasors`).

    ``design_sigma``/``design_eps_r`` are the design variables placed on
    ``design_region`` (a tuple of slices); the background is vacuum.  Other
    arguments mirror :func:`gradenna.simulate_tm`.  This bare function is
    itself differentiable through ``simulate_tm`` (all-AD); use
    :func:`freq_adjoint_gradient` for the memory-bounded adjoint gradient.
    """
    eps_r, sigma = _full_eps_sigma(grid, design_sigma, design_eps_r, design_region)
    res = simulate_tm(
        grid,
        source_ij=source_ij,
        source_current=source_current,
        eps_r=eps_r,
        sigma=sigma,
        ports=ports,
        dft_freqs=dft_freqs,
        dft_dtype=jnp.complex128,
        cpml=cpml,
    )
    port_v = port_i = None
    if res.port_v is not None:
        from gradenna.sparams import port_dft

        port_v, port_i = port_dft(res.port_v, res.port_i, grid.dt, dft_freqs)
    return FreqPhasors(res.dft_ez, res.dft_hx, res.dft_hy, port_v, port_i)


# ---------------------------------------------------------------------------
# Exact all-AD reference gradient (verification oracle)
# ---------------------------------------------------------------------------


def exact_design_gradient(
    grid: Grid2D,
    objective,
    *,
    design_sigma=None,
    design_eps_r=None,
    design_region,
    dft_freqs,
    source_ij=None,
    source_current=None,
    ports=(),
    cpml: CPMLSpec = CPMLSpec(),
):
    """All-AD reference gradient of ``objective(FreqPhasors)`` (the oracle).

    ``jax.grad`` of :func:`simulate_tm_freq` composed with ``objective``.
    Returns a dict with keys ``"sigma"`` / ``"eps_r"`` (only those supplied).
    Holds the full reverse tape (memory ``O(grid x n_steps)``); use only on
    small problems, as the ground truth the adjoint path is validated against.
    """

    def loss(design):
        ph = simulate_tm_freq(
            grid,
            design_sigma=design.get("sigma"),
            design_eps_r=design.get("eps_r"),
            design_region=design_region,
            dft_freqs=dft_freqs,
            source_ij=source_ij,
            source_current=source_current,
            ports=ports,
            cpml=cpml,
        )
        return objective(ph)

    design = {}
    if design_sigma is not None:
        design["sigma"] = design_sigma
    if design_eps_r is not None:
        design["eps_r"] = design_eps_r
    return jax.grad(loss)(design)


# ---------------------------------------------------------------------------
# Frequency-domain adjoint gradient (memory bounded, O(design x freq))
# ---------------------------------------------------------------------------


def _adjoint_source_waveforms(grid, dft_freqs, env, cot: FreqPhasors):
    """Build Jz / Mx / My adjoint CW waveforms from phasor cotangents.

    Returns ``(jz_idx, jz_wave, mx_idx, mx_wave, my_idx, my_wave)`` with
    index arrays ``(M, 2)`` and waveforms ``(n_steps, M)`` ready for
    :func:`simulate_tm`; a channel with no active cells returns ``(None, None)``.
    """
    dt = grid.dt
    n_steps = env.shape[0]
    nn = np.arange(n_steps)
    env = np.asarray(env, np.float64)
    freqs = np.asarray([float(f) for f in dft_freqs], np.float64)
    cb_src = (dt / EPS0) / (grid.dx * grid.dy)
    dt_mu = dt / MU0
    # CW phase tables (float64): Ez at (n+1)dt, H at (n+1/2)dt.
    om = 2 * np.pi * freqs
    ph_e = np.exp(1j * np.outer(nn + 1.0, om) * dt)  # (N, K)
    ph_h = np.exp(1j * np.outer(nn + 0.5, om) * dt)  # (N, K)

    def build(cot_arr, ph, basis):
        if cot_arr is None:
            return None, None
        w = np.asarray(cot_arr)  # (K, *spatial)
        active = np.any(np.abs(w) > 0, axis=0)
        if not active.any():
            return None, None
        idx = np.argwhere(active)  # (M, 2)
        # waveform[n, m] = env[n] * Re[ sum_k w[k, cell_m] ph[n, k] ] * basis_scale
        wvals = w[(slice(None),) + tuple(idx.T)]  # (K, M)
        wave = env[:, None] * np.real(ph @ wvals)  # (N, M)
        return idx, wave * basis

    jz_idx, jz_wave = build(cot.dft_ez, ph_e, 1.0 / (-cb_src))
    mx_idx, mx_wave = build(cot.dft_hx, ph_h, Q_MAG / (-dt_mu))
    my_idx, my_wave = build(cot.dft_hy, ph_h, Q_MAG / (-dt_mu))
    return jz_idx, jz_wave, mx_idx, mx_wave, my_idx, my_wave


def freq_adjoint_gradient(
    grid: Grid2D,
    objective,
    *,
    design_sigma=None,
    design_eps_r=None,
    design_region,
    dft_freqs,
    env,
    source_ij=None,
    source_current=None,
    ports=(),
    cpml: CPMLSpec = CPMLSpec(),
    backend: str = "xla",
):
    """Memory-bounded frequency-domain adjoint gradient of ``objective``.

    One forward + one adjoint :func:`simulate_tm`; backward residuals are
    only the design-region phasors (``O(N_design x N_freq)``).  ``objective``
    maps a :class:`FreqPhasors` to a real scalar.  ``source_current`` must be
    a steady CW excitation at ``dft_freqs`` with smooth turn-on envelope
    ``env`` (shape ``(n_steps,)``, e.g. a raised-cosine ramp); ``env`` is
    reused to window the adjoint source and to form ``N_eff = sum env^2``.

    ``backend`` selects the time-stepping engine for the *forward and adjoint
    field solves only*: ``"xla"`` (default) uses :func:`simulate_tm`;
    ``"native"`` uses the fused Rust kernel
    (:func:`gradenna.native.simulate_tm_native`) when available, falling back
    to XLA otherwise. Both produce the same complex128 DFT phasors that drive
    the numpy gradient contraction, so the gradient is unchanged (the field
    solves feed only the contraction, not autodiff -- the cotangent step is a
    separate, always-XLA ``jax.grad`` over post-processing). Use ``"native"``
    to cut the dominant time-loop cost on CPU.

    Returns a dict with keys ``"sigma"`` / ``"eps_r"`` (only those supplied),
    matching :func:`exact_design_gradient`.
    """
    forward = _forward_solver(backend)
    dt = grid.dt
    freqs = tuple(float(f) for f in dft_freqs)
    n_freq = len(freqs)
    env = jnp.asarray(env)
    n_steps = env.shape[0]
    om = np.asarray([2 * np.pi * f for f in freqs])
    z = np.exp(1j * om * dt)  # (K,)

    eps_r, sigma = _full_eps_sigma(grid, design_sigma, design_eps_r, design_region)

    # ---- forward run: design-region phasors -------------------------------
    fwd = forward(
        grid,
        source_ij=source_ij,
        source_current=source_current,
        eps_r=eps_r,
        sigma=sigma,
        ports=ports,
        dft_freqs=freqs,
        cpml=cpml,
    )
    # ---- cotangents of the objective (cheap post-processing AD) -----------
    # Port voltage/current phasors are reconstructed *from the field phasors*
    # inside the differentiated map (V_hat = -DZ z^{-1/2}(Ez^{n}+Ez^{n+1})/2,
    # I_hat = Ampere loop of the H phasors -- both matched to the solver to
    # ~1e-4, see tests), so a single AD pass routes V- and I-based S-parameter
    # cotangents onto dft_ez / dft_hx / dft_hy uniformly with field objectives.
    om = np.asarray([2 * np.pi * f for f in freqs])
    z_arr = jnp.asarray(np.exp(1j * om * dt))
    half_arr = jnp.asarray(np.exp(0.5j * om * dt))
    port_cells = _port_cells(grid, ports)

    def field_objective(dft_ez, dft_hx, dft_hy):
        pv = pi = None
        if port_cells:
            pv, pi = _ports_from_fields(
                dft_ez, dft_hx, dft_hy, grid, port_cells, z_arr, half_arr
            )
        return objective(FreqPhasors(dft_ez, dft_hx, dft_hy, pv, pi))

    gz, ghx, ghy = jax.grad(field_objective, argnums=(0, 1, 2))(
        fwd.dft_ez, fwd.dft_hx, fwd.dft_hy
    )
    cot = FreqPhasors(gz, ghx, ghy)

    # ---- adjoint run: same medium, CW adjoint sources ---------------------
    jz_idx, jz_wave, mx_idx, mx_wave, my_idx, my_wave = _adjoint_source_waveforms(
        grid, freqs, env, cot
    )
    adj = forward(
        grid,
        source_ij=(jz_idx if jz_idx is not None else None),
        source_current=(jz_wave if jz_idx is not None else None),
        eps_r=eps_r,
        sigma=sigma,
        mx_ij=(mx_idx if mx_idx is not None else None),
        mx_current=(mx_wave if mx_idx is not None else None),
        my_ij=(my_idx if my_idx is not None else None),
        my_current=(my_wave if my_idx is not None else None),
        dft_freqs=freqs,
        cpml=cpml,
    )

    # ---- gradient contraction on the design region ------------------------
    n_eff = float(jnp.sum(env**2))
    k_norm = 2.0 / (n_eff * dt)

    eps_d = EPS0 * np.asarray(eps_r)[design_region]
    sig_d = np.asarray(sigma)[design_region]
    h = sig_d * dt / (2.0 * eps_d)
    ca = (1.0 - h) / (1.0 + h)
    cb = (dt / eps_d) / (1.0 + h)
    dca_dsig, dcb_dsig, dca_deps, dcb_deps = _dcoef_closed(eps_d, sig_d, dt)

    e_d = np.asarray(fwd.dft_ez)[(slice(None),) + design_region]  # (K, *dr)
    lam_d = np.asarray(adj.dft_ez)[(slice(None),) + design_region]

    # Steady-state discrete phasor relations on the design cells:
    #   Ez^{n} phasor = Ez^{n+1} phasor / z,  z = e^{i w dt};
    #   raw discrete curl^{n+1} phasor = (Ez^{n+1} - c_a Ez^{n}) / c_b.
    z_b = z.reshape((-1,) + (1,) * (e_d.ndim - 1))
    en_d = e_d / z_b
    curl_d = (e_d - ca[None] * en_d) / cb[None]

    # The adjoint Ez running-DFT phasor ``lam_d`` is recorded on the vacuum
    # unit-injection basis (the Jz source uses the vacuum c_{b,src}); the
    # design cell however updates Ez with the *local* c_b = (dt/eps)/(1+h),
    # i.e. the adjoint cell amplitude carries a 1/eps_r relative to vacuum.
    # Undo it once for the whole contraction with eps_r = eps_d/EPS0
    # (verified: ratio/eps_r is the uniform transient scale for both channels
    # and for non-uniform eps_r designs).
    eps_r_d = eps_d / EPS0  # (*dr)

    def contract(dca, dcb):
        g = dca[None] * en_d + dcb[None] * curl_d  # (K, *dr)
        return jnp.asarray(eps_r_d * (k_norm * np.real(np.conj(lam_d) * np.conj(g))).sum(axis=0))

    grads = {}
    if design_sigma is not None:
        grads["sigma"] = contract(dca_dsig, dcb_dsig)
    if design_eps_r is not None:
        grads["eps_r"] = contract(EPS0 * dca_deps, EPS0 * dcb_deps)
    return grads


def _forward_solver(backend: str):
    """Return a forward field-solve callable for the requested backend.

    Both variants accept the keyword subset used by the adjoint driver
    (source/mx/my currents, eps_r, sigma, ports, dft_freqs, cpml) and return
    an object exposing ``dft_ez/dft_hx/dft_hy`` (and ``port_v/port_i`` when
    ports are given) as complex128 phasors. ``"native"`` silently falls back
    to XLA when the Rust kernel is unavailable.
    """
    if backend not in ("xla", "native"):
        raise ValueError(f"backend must be 'xla' or 'native', got {backend!r}")

    use_native = False
    if backend == "native":
        from gradenna import native

        use_native = native.is_available()

    if use_native:
        from gradenna import native

        def solve(grid, **kw):
            # f64 fields with f64 DFT accumulators == dft_dtype=complex128.
            return native.simulate_tm_native(grid, dtype="float64", **kw)

        return solve

    def solve(grid, **kw):
        return simulate_tm(grid, dft_dtype=jnp.complex128, **kw)

    return solve


def _port_cells(grid, ports):
    """List of ``(i, j)`` port Ez-cell indices (empty if no ports)."""
    from gradenna.fdtd2d import Port

    return [tuple(int(v) for v in (p if isinstance(p, Port) else Port(*p)).ij) for p in ports]


def _ports_from_fields(dft_ez, dft_hx, dft_hy, grid, port_cells, z_arr, half_arr):
    """Reconstruct ``(V_hat, I_hat)`` from the field phasors (note 12, 2D form).

    Matches the solver's port recordings to ~1e-4 (verified), so that
    differentiating an S-parameter objective through these reconstructions
    routes the cotangents onto the field phasors:

        V_hat = -DZ * z^{-1/2} * (Ez^{n}_hat + Ez^{n+1}_hat) / 2,
                with Ez^{n}_hat = Ez^{n+1}_hat / z, z = e^{i w dt};
        I_hat = (Hy[i,j]-Hy[i-1,j]) dy - (Hx[i,j]-Hx[i,j-1]) dx.
    """
    from gradenna.fdtd2d import DZ

    vh, ih = [], []
    for (i, j) in port_cells:
        ez = dft_ez[:, i, j]
        v = -DZ * 0.5 * (ez / z_arr + ez) * half_arr
        cur = (dft_hy[:, i, j] - dft_hy[:, i - 1, j]) * grid.dy - (
            dft_hx[:, i, j] - dft_hx[:, i, j - 1]
        ) * grid.dx
        vh.append(v)
        ih.append(cur)
    return jnp.stack(vh, axis=1), jnp.stack(ih, axis=1)


# ===========================================================================
# 3D frequency-domain adjoint (docs/research/16-freq-adjoint-3d-theory.md)
# ===========================================================================
r"""3D extension.

The 3D mechanism is the dimension-agnostic version of the 2D reduction above,
made explicit in docs/research/16.  A design cell ``(i, j, k)`` carries one
``sigma``/``eps_r`` value that is applied identically to the three E edges
emanating from its node (``Ex(i+1/2,j,k)``, ``Ey(i,j+1/2,k)``,
``Ez(i,j,k+1/2)``; fdtd3d cell->edge slices), so the gradient is the **sum of
three per-component contributions**, each of the same form as the 2D scalar:

.. math::

    \frac{\partial L}{\partial\theta_{(i,j,k)}}
    = \sum_{\alpha\in\{x,y,z\}}\ \sum_k \frac{2}{N_\mathrm{eff}\,\Delta t}\,
      \mathrm{Re}\!\big[\overline{\hat\Lambda_\alpha}\,\overline{\hat G_\alpha}\big],
    \quad
    \hat G_\alpha=\partial_\theta c_a\,\hat E_\alpha^{(n)}
                 +\partial_\theta c_b\,\hat C_\alpha,

with :math:`\hat E_\alpha^{(n)}=\hat E_\alpha/z`,
:math:`\hat C_\alpha=(\hat E_\alpha-c_a\hat E_\alpha^{(n)})/c_b`,
:math:`z=e^{i\omega\Delta t}`, and :math:`\hat\Lambda_\alpha` the adjoint E
phasor of component :math:`\alpha` on the design cells.  The vacuum-unit
injection basis carries a ``1/eps_r`` correction applied once to each
component (note 16 Sec. 2.2).

The six objective cotangents become adjoint sources (note 16 Sec. 2.3): the E
cotangents ``g_ex/g_ey/g_ez`` drive ``Jx/Jy/Jz`` (phase :math:`(n+1)\Delta t`,
basis :math:`1/(-c_{b,\mathrm{src}})` with the **component-specific transverse
area** -- ``Jx``: :math:`\Delta y\Delta z`, ``Jy``: :math:`\Delta x\Delta z`,
``Jz``: :math:`\Delta x\Delta y`), and the H cotangents ``g_hx/g_hy/g_hz``
drive ``Mx/My/Mz`` (phase :math:`(n+1/2)\Delta t`, basis
:math:`Q_\mathrm{MAG}/(-\Delta t/\mu_0)` with the single closed-form constant
:data:`Q_MAG`\ =\ :math:`-\varepsilon_0/\mu_0`, identical for all three H
components because the Yee metric ratio is component-independent).
"""


class FreqPhasors3D(NamedTuple):
    """Frequency-domain phasors returned by :func:`simulate_3d_freq`.

    dft_ex..dft_hz: full-grid running-DFT phasors (same convention and shapes
        as :class:`gradenna.fdtd3d.DFTMonitor`), shape ``(n_freq, ...)``.
    port_v/port_i: port voltage/current phasors, shape ``(n_freq, n_ports)``
        or ``None`` if no port was given.
    """

    dft_ex: jnp.ndarray
    dft_ey: jnp.ndarray
    dft_ez: jnp.ndarray
    dft_hx: jnp.ndarray
    dft_hy: jnp.ndarray
    dft_hz: jnp.ndarray
    port_v: jnp.ndarray | None = None
    port_i: jnp.ndarray | None = None


def _full_eps_sigma_3d(grid, design_sigma, design_eps_r, design_region):
    eps_r = jnp.ones(grid.shape)
    sigma = jnp.zeros(grid.shape)
    if design_eps_r is not None:
        eps_r = eps_r.at[design_region].set(design_eps_r)
    if design_sigma is not None:
        sigma = sigma.at[design_region].set(design_sigma)
    return eps_r, sigma


def _port_v_i_3d(dft_ez, dft_hx, dft_hy, grid, pijk, z_arr, half_arr):
    r"""Reconstruct ``(V_hat, I_hat)`` of a single 3D RVS port from the phasors.

    The 3D lumped port is a single z-directed Ez edge (fdtd3d module
    docstring); the reconstruction mirrors the solver's V/I recordings to the
    same transient floor as in 2D (note 16 Sec. 2.4):

    .. math::

        \hat V = -d_z\,\tfrac12(\hat E_z^{(n)}+\hat E_z)\,z^{1/2},\quad
        \hat E_z^{(n)}=\hat E_z/z,\\
        \hat I = (\hat H_y|_{i}-\hat H_y|_{i-1})\,d_y
               + (\hat H_x|_{j-1}-\hat H_x|_{j})\,d_x .

    The Ampere-loop sign order follows the fdtd3d ``i_loop`` expression
    (note 16 Sec. 2.4, pitfall 2: 3D writes ``(Hx[j-1]-Hx[j])`` where 2D
    writes ``-(Hx[j]-Hx[j-1])`` -- algebraically equal).
    """
    i, j, k = pijk
    ez = dft_ez[:, i, j, k]
    v = -grid.dz * 0.5 * (ez / z_arr + ez) * half_arr
    cur = (dft_hy[:, i, j, k] - dft_hy[:, i - 1, j, k]) * grid.dy + (
        dft_hx[:, i, j - 1, k] - dft_hx[:, i, j, k]
    ) * grid.dx
    return v[:, None], cur[:, None]


def simulate_3d_freq(
    grid: Grid3D,
    *,
    design_sigma=None,
    design_eps_r=None,
    design_region,
    dft_freqs,
    source_ijk=None,
    source_current=None,
    port_ijk=None,
    port_voltage=None,
    port_resistance: float = 50.0,
    cpml: CPMLSpec = CPMLSpec(),
    checkpoint_segments: int | None = None,
) -> FreqPhasors3D:
    """3D forward run returning frequency-domain phasors (see :class:`FreqPhasors3D`).

    ``design_sigma``/``design_eps_r`` are the design variables placed on
    ``design_region`` (a tuple of three slices into the ``(nx, ny, nz)`` cell
    lattice -- a 3D box or a planar sheet); the background is vacuum.  Drive
    with a z-directed point current (``source_ijk``/``source_current``) and/or
    a single lumped RVS port (``port_ijk``/``port_voltage``).  This bare
    function is itself differentiable through :func:`gradenna.simulate_3d`
    (all-AD); use :func:`freq_adjoint_gradient_3d` for the memory-bounded
    adjoint gradient.
    """
    eps_r, sigma = _full_eps_sigma_3d(grid, design_sigma, design_eps_r, design_region)
    res = simulate_3d(
        grid,
        eps_r=eps_r,
        sigma=sigma,
        source_ijk=source_ijk,
        source_current=source_current,
        port_ijk=port_ijk,
        port_voltage=port_voltage,
        port_resistance=port_resistance,
        dft_freqs=dft_freqs,
        dft_dtype=jnp.complex128,
        cpml=cpml,
        checkpoint_segments=checkpoint_segments,
    )
    d = res.dft
    port_v = port_i = None
    if res.port_v is not None:
        port_v, port_i = port_dft_3d(res.port_v, res.port_i, grid.dt, dft_freqs)
    return FreqPhasors3D(d.ex, d.ey, d.ez, d.hx, d.hy, d.hz, port_v, port_i)


def port_dft_3d(v_t, i_t, dt, freqs):
    """Half-step DFT of the 3D port V/I series (both at ``t = (n+1/2) dt``)."""
    v_hat = time_series_dft(v_t, dt, freqs, t0=0.5 * dt)
    i_hat = time_series_dft(i_t, dt, freqs, t0=0.5 * dt)
    return v_hat[:, None], i_hat[:, None]


def exact_design_gradient_3d(
    grid: Grid3D,
    objective,
    *,
    design_sigma=None,
    design_eps_r=None,
    design_region,
    dft_freqs,
    source_ijk=None,
    source_current=None,
    port_ijk=None,
    port_voltage=None,
    port_resistance: float = 50.0,
    cpml: CPMLSpec = CPMLSpec(),
    checkpoint_segments: int | None = None,
):
    """All-AD reference gradient of ``objective(FreqPhasors3D)`` (the 3D oracle).

    ``jax.grad`` of :func:`simulate_3d_freq` composed with ``objective``.
    Returns a dict with keys ``"sigma"`` / ``"eps_r"`` (only those supplied).
    Holds the full reverse tape (memory ``O(grid x n_steps)`` unless
    ``checkpoint_segments`` is used); the ground truth the adjoint path is
    validated against.
    """

    def loss(design):
        ph = simulate_3d_freq(
            grid,
            design_sigma=design.get("sigma"),
            design_eps_r=design.get("eps_r"),
            design_region=design_region,
            dft_freqs=dft_freqs,
            source_ijk=source_ijk,
            source_current=source_current,
            port_ijk=port_ijk,
            port_voltage=port_voltage,
            port_resistance=port_resistance,
            cpml=cpml,
            checkpoint_segments=checkpoint_segments,
        )
        return objective(ph)

    design = {}
    if design_sigma is not None:
        design["sigma"] = design_sigma
    if design_eps_r is not None:
        design["eps_r"] = design_eps_r
    return jax.grad(loss)(design)


def _adjoint_sources_3d(grid, dft_freqs, env, cot: FreqPhasors3D):
    """Build Jx/Jy/Jz and Mx/My/Mz adjoint CW waveforms from phasor cotangents.

    Returns a dict of ``{channel: (idx (M,3), wave (n_steps, M))}`` for each
    active channel (note 16 Sec. 2.3 table); inactive channels are omitted.
    """
    dt = grid.dt
    n_steps = env.shape[0]
    nn = np.arange(n_steps)
    env = np.asarray(env, np.float64)
    freqs = np.asarray([float(f) for f in dft_freqs], np.float64)
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    dt_mu = dt / MU0
    om = 2 * np.pi * freqs
    ph_e = np.exp(1j * np.outer(nn + 1.0, om) * dt)  # (N, K), Ez at (n+1) dt
    ph_h = np.exp(1j * np.outer(nn + 0.5, om) * dt)  # (N, K), H at (n+1/2) dt

    def build(cot_arr, ph, basis):
        if cot_arr is None:
            return None
        w = np.asarray(cot_arr)  # (K, *spatial)
        active = np.any(np.abs(w) > 0, axis=0)
        if not active.any():
            return None
        idx = np.argwhere(active)  # (M, 3)
        wvals = w[(slice(None),) + tuple(idx.T)]  # (K, M)
        wave = env[:, None] * np.real(ph @ wvals)  # (N, M)
        return idx, wave * basis

    # J basis 1/(-c_{b,src}); c_{b,src} = (dt/eps0)/area; area is the curl
    # cross-section of each component (note 16 Sec. 2.3, pitfall 1).
    cb_jx = (dt / EPS0) / (dy * dz)
    cb_jy = (dt / EPS0) / (dx * dz)
    cb_jz = (dt / EPS0) / (dx * dy)
    m_basis = Q_MAG / (-dt_mu)

    out = {}
    for name, ch, ph, basis in (
        ("jx", cot.dft_ex, ph_e, 1.0 / (-cb_jx)),
        ("jy", cot.dft_ey, ph_e, 1.0 / (-cb_jy)),
        ("jz", cot.dft_ez, ph_e, 1.0 / (-cb_jz)),
        ("mx", cot.dft_hx, ph_h, m_basis),
        ("my", cot.dft_hy, ph_h, m_basis),
        ("mz", cot.dft_hz, ph_h, m_basis),
    ):
        r = build(ch, ph, basis)
        if r is not None:
            out[name] = r
    return out


def _forward_solver_3d(backend: str):
    """Return a 3D forward field-solve callable for the requested backend.

    Both variants accept the keyword subset used by the 3D adjoint driver
    (Jx/Jy/Jz and Mx/My/Mz currents, eps_r, sigma, the lumped port, dft_freqs,
    cpml) and return an object exposing ``dft`` (a six-component
    :class:`gradenna.fdtd3d.DFTMonitor`) and ``port_v``/``port_i`` as
    complex128 phasors. ``"native"`` silently falls back to XLA when the Rust
    kernel is unavailable. The field solves feed only the numpy gradient
    contraction (not autodiff), so the gradient is backend-independent.
    """
    if backend not in ("xla", "native"):
        raise ValueError(f"backend must be 'xla' or 'native', got {backend!r}")

    use_native = False
    if backend == "native":
        from gradenna import native3d

        use_native = native3d.is_available()

    if use_native:
        from gradenna import native3d

        def solve(grid, *, dft_dtype=None, checkpoint_segments=None, **kw):
            # f64 fields + f64 DFT accumulators == dft_dtype=complex128.
            return native3d.simulate_3d_native(grid, dtype="float64", **kw)

        return solve

    def solve(grid, **kw):
        return simulate_3d(grid, **kw)

    return solve


def freq_adjoint_gradient_3d(
    grid: Grid3D,
    objective,
    *,
    design_sigma=None,
    design_eps_r=None,
    design_region,
    dft_freqs,
    env,
    source_ijk=None,
    source_current=None,
    port_ijk=None,
    port_voltage=None,
    port_resistance: float = 50.0,
    cpml: CPMLSpec = CPMLSpec(),
    backend: str = "xla",
):
    r"""Memory-bounded 3D frequency-domain adjoint gradient of ``objective``.

    One forward + one adjoint :func:`gradenna.simulate_3d`; backward residuals
    are only the design-region phasors of the three E components
    (``O(3 x N_design x N_freq)``), independent of ``n_steps``.  ``objective``
    maps a :class:`FreqPhasors3D` to a real scalar (a field flux, a port
    S-parameter, or any ``loss(ntff_3d(...))`` -- JAX backpropagates the NTFF
    einsum to the six DFT cotangents automatically).

    The contraction sums three per-component terms (note 16 Sec. 2.2),

    .. math::

        \partial_\theta L = \sum_{\alpha} \sum_k \frac{2}{N_\mathrm{eff}\Delta t}
          \,\varepsilon_{r}\,
          \mathrm{Re}\!\big[\overline{\hat\Lambda_\alpha}\,\overline{\hat G_\alpha}\big],

    with :math:`\hat G_\alpha=\partial_\theta c_a\hat E_\alpha/z
    +\partial_\theta c_b(\hat E_\alpha-c_a\hat E_\alpha/z)/c_b` and the
    closed-form coefficient sensitivities of :func:`_dcoef_closed`.

    Returns a dict with keys ``"sigma"`` / ``"eps_r"`` (only those supplied),
    matching :func:`exact_design_gradient_3d`.
    """
    dt = grid.dt
    freqs = tuple(float(f) for f in dft_freqs)
    env = jnp.asarray(env)
    om = np.asarray([2 * np.pi * f for f in freqs])
    z = np.exp(1j * om * dt)  # (K,)
    z_arr = jnp.asarray(z)
    half_arr = jnp.asarray(np.exp(0.5j * om * dt))

    eps_r, sigma = _full_eps_sigma_3d(grid, design_sigma, design_eps_r, design_region)
    forward = _forward_solver_3d(backend)

    common = dict(
        eps_r=eps_r,
        sigma=sigma,
        dft_freqs=freqs,
        dft_dtype=jnp.complex128,
        cpml=cpml,
    )

    # ---- forward run --------------------------------------------------------
    fwd = forward(
        grid,
        source_ijk=source_ijk,
        source_current=source_current,
        port_ijk=port_ijk,
        port_voltage=port_voltage,
        port_resistance=port_resistance,
        **common,
    )
    fd = fwd.dft

    # ---- objective cotangents (cheap post-processing AD) --------------------
    pijk = None if port_ijk is None else tuple(int(v) for v in np.asarray(port_ijk).reshape(3))

    def field_objective(ex, ey, ez, hx, hy, hz):
        pv = pi = None
        if pijk is not None:
            pv, pi = _port_v_i_3d(ez, hx, hy, grid, pijk, z_arr, half_arr)
        return objective(FreqPhasors3D(ex, ey, ez, hx, hy, hz, pv, pi))

    gex, gey, gez, ghx, ghy, ghz = jax.grad(field_objective, argnums=(0, 1, 2, 3, 4, 5))(
        fd.ex, fd.ey, fd.ez, fd.hx, fd.hy, fd.hz
    )
    cot = FreqPhasors3D(gex, gey, gez, ghx, ghy, ghz)

    # ---- adjoint run: same medium, CW adjoint J/M sources -------------------
    chans = _adjoint_sources_3d(grid, freqs, env, cot)
    src_kw = {}
    if "jx" in chans:
        src_kw["source_x_ijk"], src_kw["source_x_current"] = chans["jx"]
    if "jy" in chans:
        src_kw["source_y_ijk"], src_kw["source_y_current"] = chans["jy"]
    if "jz" in chans:
        src_kw["source_ijk"], src_kw["source_current"] = chans["jz"]
    if "mx" in chans:
        src_kw["mx_ijk"], src_kw["mx_current"] = chans["mx"]
    if "my" in chans:
        src_kw["my_ijk"], src_kw["my_current"] = chans["my"]
    if "mz" in chans:
        src_kw["mz_ijk"], src_kw["mz_current"] = chans["mz"]
    adj = forward(grid, **src_kw, **common)
    ad = adj.dft

    # ---- gradient contraction on the design region (3 E components) ---------
    n_eff = float(jnp.sum(env**2))
    k_norm = 2.0 / (n_eff * dt)

    eps_d = EPS0 * np.asarray(eps_r)[design_region]
    sig_d = np.asarray(sigma)[design_region]
    h = sig_d * dt / (2.0 * eps_d)
    ca = (1.0 - h) / (1.0 + h)
    cb = (dt / eps_d) / (1.0 + h)
    dca_dsig, dcb_dsig, dca_deps, dcb_deps = _dcoef_closed(eps_d, sig_d, dt)
    eps_r_d = eps_d / EPS0  # 1/eps_r vacuum-basis correction, per component

    sel = (slice(None),) + design_region
    z_b = z.reshape((-1,) + (1,) * eps_d.ndim)

    # Forward / adjoint E phasors of the three components on the design cells.
    comps = (
        (np.asarray(fd.ex)[sel], np.asarray(ad.ex)[sel]),
        (np.asarray(fd.ey)[sel], np.asarray(ad.ey)[sel]),
        (np.asarray(fd.ez)[sel], np.asarray(ad.ez)[sel]),
    )

    def contract(dca, dcb):
        total = 0.0
        for e_d, lam_d in comps:
            en_d = e_d / z_b
            curl_d = (e_d - ca[None] * en_d) / cb[None]
            g = dca[None] * en_d + dcb[None] * curl_d
            total = total + (k_norm * np.real(np.conj(lam_d) * np.conj(g))).sum(axis=0)
        return jnp.asarray(eps_r_d * total)

    grads = {}
    if design_sigma is not None:
        grads["sigma"] = contract(dca_dsig, dcb_dsig)
    if design_eps_r is not None:
        grads["eps_r"] = contract(EPS0 * dca_deps, EPS0 * dcb_deps)
    return grads
