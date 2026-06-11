# gradenna

**grad**ient + ant**enna** — Differentiable FDTD antenna inverse design in JAX. Grow antennas by gradient descent.

JAXで書いた微分可能FDTD（時間領域差分法）の上で、RFアンテナのトポロジーを勾配降下で設計するプロジェクト。

## Status

**Phase 1: 微分可能2D FDTDコア（開発中）**

- [ ] 2D TMモード FDTD（`jax.lax.scan`、jit、float32/float64切替）
- [ ] 解析解（線電流の円筒波）による場の検証
- [ ] `jax.grad` の勾配 vs 有限差分の自動テスト
- [ ] CPML吸収境界と反射率の定量検証

## Roadmap

1. **Phase 1**: 微分可能2D FDTDコア ← いまここ
2. **Phase 2**: 集中ポート・S11・周波数応答（openEMSクロスチェック）
3. **Phase 3**: 2Dトポロジー最適化デモ（密度法＋フィルタ＋β継続）
4. **Phase 4**: 2.5D/3D・Gerber出力・NanoVNA実測
5. **Phase 5**: 指向性・マルチバンド・アレイ最適化

## License

MIT
