"""B4: CPML reflection benchmark (enlarged-domain reference method).

The test domain (with CPML) and a much larger PEC box share the same
source/probe layout. Within the observation window no reflection from the
big box's walls can geometrically reach the probe, so the difference
between the two records is purely the CPML reflection error:

    R_dB = 20 log10( max_t |E(t) - E_ref(t)| / max_t |E_ref(t)| )
"""

import jax.numpy as jnp
import pytest

from gradenna import CPMLSpec, Grid2D, alpha_max_for_fmin, gaussian_derivative, simulate_tm
from gradenna.cpml import axis_coefficients

DX = 2e-3
NPML = 10
# Differentiated Gaussian peaking at 2.5 GHz keeps the spectrum within the
# well-resolved band (lambda/dx >= 20 up to 7.5 GHz).
TAU = 1.0 / (2 * jnp.pi * 2.5e9)


def _record(n: int, n_steps: int, cpml: CPMLSpec, src_offset, probe_offset):
    grid = Grid2D(nx=n, ny=n, dx=DX, dy=DX)
    c = n // 2
    src = (c + src_offset[0], c + src_offset[1])
    probe = (c + probe_offset[0], c + probe_offset[1])
    t = (jnp.arange(n_steps) + 0.5) * grid.dt
    current = gaussian_derivative(t, t0=6 * float(TAU), tau=float(TAU))
    res = simulate_tm(grid, source_ij=src, source_current=current, probe_ij=(probe,), cpml=cpml)
    return res.probe_ez[:, 0], res


# Probe 3 cells inside the PML interface: edge-normal and corner-diagonal cases.
@pytest.mark.parametrize(
    "probe_offset", [(0, 41), (32, 32)], ids=["edge", "corner"]
)
def test_cpml_reflection_below_minus_60db(probe_offset):
    n_test = 110  # interior 90 cells, interface 41 cells from center
    n_steps = 600  # ~ 420 cells of travel: several round trips in the test box
    cpml = CPMLSpec(thickness=NPML, alpha_max=alpha_max_for_fmin(0.5e9))
    e_test, _ = _record(n_test, n_steps, cpml, (0, 0), probe_offset)

    # Reference: PEC box large enough that source->wall->probe > c * T_obs.
    e_ref, _ = _record(560, n_steps, CPMLSpec(thickness=0), (0, 0), probe_offset)

    err = jnp.max(jnp.abs(e_test - e_ref)) / jnp.max(jnp.abs(e_ref))
    r_db = 20.0 * jnp.log10(err)
    assert float(r_db) <= -60.0, f"CPML reflection {float(r_db):.1f} dB > -60 dB"


def test_late_time_energy_decay():
    """After the pulse leaves, total energy must decay to a deep floor."""
    cpml = CPMLSpec(thickness=NPML, alpha_max=alpha_max_for_fmin(0.5e9))
    _, res = _record(110, 1500, cpml, (0, 0), (0, 41))
    peak = float(res.energy.max())
    assert float(res.energy[-1]) <= 1e-10 * peak


def test_zero_thickness_is_plain_yee():
    """thickness=0 must produce zero psi coupling (c = 0, kappa = 1)."""
    coeffs = axis_coefficients(64, DX, 1e-12, CPMLSpec(thickness=0), half=False)
    assert float(jnp.abs(coeffs.c).max()) == 0.0
    assert float(jnp.abs(coeffs.inv_kappa - 1.0).max()) == 0.0


def test_degenerate_matched_loss_layer():
    """kappa_max=1, alpha_max=0 reduces s_w to 1 + sigma/(j w eps0):
    b = exp(-sigma dt/eps0) and c = b - 1 exactly."""
    spec = CPMLSpec(thickness=8, kappa_max=1.0, alpha_max=0.0)
    co = axis_coefficients(40, DX, 3e-12, spec, half=False)
    inside = co.c != 0.0
    assert bool(jnp.all(jnp.isclose(co.c[inside], co.b[inside] - 1.0)))
    assert float(jnp.abs(co.inv_kappa - 1.0).max()) == 0.0


def test_coefficients_have_no_nan():
    """sigma = alpha = 0 cells must not produce NaN (guarded division)."""
    for spec in (CPMLSpec(), CPMLSpec(alpha_max=0.05), CPMLSpec(thickness=0)):
        for half in (False, True):
            co = axis_coefficients(50, DX, 3e-12, spec, half=half)
            assert bool(jnp.all(jnp.isfinite(co.b)))
            assert bool(jnp.all(jnp.isfinite(co.c)))
            assert bool(jnp.all(jnp.isfinite(co.inv_kappa)))
