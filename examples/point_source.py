"""Radiate a pulse from a line current and (a) plot the field snapshot,
(b) differentiate the probe energy w.r.t. a permittivity patch.

Run:  uv run python examples/point_source.py
Writes point_source.png next to this script.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from gradenna import CPMLSpec, Grid2D, alpha_max_for_fmin, gaussian_derivative, simulate_tm

grid = Grid2D(nx=160, ny=160, dx=2e-3, dy=2e-3)
cpml = CPMLSpec(thickness=10, alpha_max=alpha_max_for_fmin(1e9))
tau = 1.0 / (2 * jnp.pi * 2.5e9)
t = (jnp.arange(420) + 0.5) * grid.dt
current = gaussian_derivative(t, t0=6 * tau, tau=tau)

# Snapshot while the wavefront is still inside the domain (~100 cells radius).
res = simulate_tm(grid, source_ij=(80, 80), source_current=current[:150], cpml=cpml)


# Differentiate probe energy w.r.t. an 16x16 permittivity patch — the seed of
# topology optimization (Phase 3).
def loss(eps_patch):
    eps_r = jnp.ones(grid.shape).at[100:116, 72:88].set(eps_patch)
    out = simulate_tm(
        grid,
        source_ij=(80, 80),
        source_current=current,
        probe_ij=((140, 80),),
        eps_r=eps_r,
        cpml=cpml,
    )
    return jnp.sum(out.probe_ez**2)


g = jax.grad(loss)(2.0 * jnp.ones((16, 16)))

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
extent = [0, grid.nx * grid.dx * 1e3, 0, grid.ny * grid.dy * 1e3]
vmax = float(jnp.abs(res.ez).max())
axes[0].imshow(res.ez.T, origin="lower", cmap="RdBu", vmin=-vmax, vmax=vmax, extent=extent)
axes[0].set_title("Ez snapshot (CPML-absorbed pulse)")
axes[0].set_xlabel("x [mm]")
axes[0].set_ylabel("y [mm]")
patch_mm = g.shape[0] * grid.dx * 1e3
im = axes[1].imshow(g.T, origin="lower", cmap="PiYG", extent=[0, patch_mm, 0, patch_mm])
axes[1].set_title(r"$\partial$(probe energy)/$\partial\varepsilon_r$ patch")
axes[1].set_xlabel("x [mm]")
axes[1].set_ylabel("y [mm]")
fig.colorbar(im, ax=axes[1])
out_path = Path(__file__).with_name("point_source.png")
fig.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"saved {out_path}")
print(f"grad norm: {float(jnp.linalg.norm(g)):.3e}")
