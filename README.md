# gradenna

**grad**ient + ant**enna** — Differentiable FDTD antenna inverse design in JAX. Grow antennas by gradient descent.

JAXで書いた微分可能FDTD（時間領域差分法）の上で、RFアンテナのトポロジーを勾配降下で設計するプロジェクト。

## Status

**Phase 1: 微分可能2D FDTDコア（完了）**

- [x] 2D TMモード FDTD（`jax.lax.scan`、jit可能、float32/float64切替）
- [x] 解析解（線電流の円筒波 `H0^(2)(kρ)`）による場の検証 — 正規化プロファイル誤差 <3%、2次グリッド収束
- [x] `jax.grad` の勾配 vs 有限差分の自動テスト — 方向微分の相対誤差 ≤1e-4（float64）
- [x] CPML吸収境界（CFS、Roden & Gedney 2000）— 拡大領域リファレンス比較で反射 ≤ −60 dB

## Quick start

```bash
uv sync
uv run pytest -m "not slow"          # fast verification suite
uv run python examples/point_source.py
```

```python
import jax, jax.numpy as jnp
from gradenna import Grid2D, CPMLSpec, simulate_tm, gaussian_derivative

grid = Grid2D(nx=160, ny=160, dx=2e-3, dy=2e-3)
t = (jnp.arange(400) + 0.5) * grid.dt
current = gaussian_derivative(t, t0=0.4e-9, tau=64e-12)

def loss(eps_patch):
    eps_r = jnp.ones(grid.shape).at[100:116, 72:88].set(eps_patch)
    res = simulate_tm(grid, source_ij=(80, 80), source_current=current,
                      probe_ij=((140, 80),), eps_r=eps_r, cpml=CPMLSpec(thickness=10))
    return jnp.sum(res.probe_ez**2)

grad = jax.grad(loss)(2.0 * jnp.ones((16, 16)))  # ∂loss/∂εr — 逆設計の種
```

## Roadmap

1. **Phase 1**: 微分可能2D FDTDコア ← いまここ
2. **Phase 2**: 集中ポート・S11・周波数応答（openEMSクロスチェック）
3. **Phase 3**: 2Dトポロジー最適化デモ（密度法＋フィルタ＋β継続）
4. **Phase 4**: 2.5D/3D・Gerber出力・NanoVNA実測
5. **Phase 5**: 指向性・マルチバンド・アレイ最適化

## License

MIT
