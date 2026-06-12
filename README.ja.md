# gradenna

[![CI](https://github.com/s-hosomi/gradenna/actions/workflows/ci.yml/badge.svg)](https://github.com/s-hosomi/gradenna/actions/workflows/ci.yml)

**grad**ient + ant**enna** — JAX で書いた微分可能 FDTD によるアンテナ逆設計。RF アンテナを勾配降下で「育てる」。

[English README](README.md)

gradenna は RF/マイクロ波アンテナ設計のための、完全微分可能な電磁界（FDTD）ソルバー＋トポロジー最適化ツールキットです。Yee 更新・CPML 吸収境界・50Ω 集中ポート・ランニング DFT による S パラメータ・近傍場→遠方場変換（NTFF）までのシミュレーション全体が単一の JAX 計算グラフになっており、`jax.grad` 一発で任意の目的関数（S11・放射電力・指向性・利得）の**全設計ピクセル同時の**厳密な随伴勾配が得られます。

## 特徴

- **微分可能な 2D TM / フル 3D FDTD コア**（`jax.lax.scan`、jit 可、float32/float64）、CPML/CFS 吸収境界、複数ソース、√N 勾配チェックポイント
- **集中 RVS ポートと S パラメータ**: 半陰的な抵抗付き電圧源ポート、厳密位相ランニング DFT、パワー波 S11、離散ギャップサセプタンスのデエンベッディング
- **微分可能 NTFF**（2D/3D）: 放射電力・指向性・利得を最適化目的に
- **トポロジー最適化ツールキット**: conic 密度フィルタ、tanh 射影＋β 継続、対数導電率の金属補間、連結性・最小線幅チェック
- **製造パイプライン**: 密度マップ → ポリゴン → RS-274X Gerber（JLCPCB デザインルールチェック付き、`pip install gradenna[fab]`）
- **実測ループ**: Touchstone 入出力、sim vs 実測 S11 比較、NanoVNA 取得スクリプト（`pip install gradenna[measure]`）

## 検証

全物理コンポーネントを解析解・教科書値に対して CI でテストしています（94 テスト）:

| ベンチマーク | 結果 |
|---|---|
| 線電流の円筒波 vs `H0^(2)(kρ)`（Harrington） | プロファイル誤差 <2.5%、2次グリッド収束 |
| CPML 反射（拡大領域リファレンス比較） | −92 dB（基準 −60 dB） |
| 2D 線電流の放射抵抗 vs ωμ0/4 | 1.3–2.4% |
| 微小ダイポール放射抵抗 vs 80π²(l/λ)² | デエンベッド後 0.34% |
| NTFF 経由のダイポール指向性 vs D₀=1.5 | 0.14% |
| 2.45 GHz FR-4 パッチ共振 vs Balanis 設計式 | −2.5% |
| 2.45 GHz パッチ vs openEMS（コミット済み参照データ） | 共振 0.83%、\|S11\| RMS 0.51 dB、パターン相関 ≥ 0.999 |
| 細線 MoM ダイポール共振 vs 教科書値（0.47–0.48 λ、~72 Ω） | L_res = 0.476 λ、Re Zin = 71.7 Ω |
| `jax.grad` vs 有限差分（全パラメータ種別） | 相対誤差 ≤1e-4 |
| チェックポイント随伴 vs 素朴随伴 | ビット一致 |

## デモ

| スクリプト | 内容 |
|---|---|
| `examples/optimize_2d_antenna.py` | 一様グレーからアンテナが生える: 2.45 GHz 放射エネルギー最大化、空箱比 4 倍、完全二値の最終設計 |
| `examples/optimize_directivity.py` | 遠方界変換を通したビーム形成: D(0°) 0.31 → 4.47、F/B 比 16.8 dB |
| `examples/optimize_multiband.py` | 2.0 + 3.0 GHz 同時の最悪帯域（softmin）放射電力最大化 |
| `examples/optimize_3d_patch.py` | **3Dトポロジー最適化**: 実FR-4スタックアップ上の銅密度をチェックポイント随伴で最適化。`--preset cpu-demo`（放射39倍、約2.5分）/ `--preset gpu-24gb` |
| `examples/optimize_beamsteering.py` | **ビームステアリング**: 4素子 λ/2 アレイの複素給電重みを遠方界変換ごしに最適化。±30° 目標に角度分解能内で到達 |
| `examples/patch_to_gerber.py` | Balanis パッチ設計 → 密度マップ → DRC → Gerber |

## クイックスタート

```bash
git clone https://github.com/s-hosomi/gradenna && cd gradenna
uv sync                                  # または pip install -e ".[fab,measure]"
uv run pytest -m "not slow" -q           # 高速検証スイート（CPU で 1〜2 分）
uv run python examples/optimize_2d_antenna.py
```

CPU でそのまま動きます（全デモ数分で完走）。JAX の GPU/TPU バックエンドも無変更で利用可能です。

## GPU と Apple Silicon

- **メモリ制約下の3D随伴**: CPML補助変数を PML スラブ形状で格納（3Dでψメモリ−74%）し、√Nチェックポイントと組み合わせることで、フル解像度の3Dパッチ最適化のピークメモリは **7.7 GB（float64）/ 約3.9 GB（float32）** — 24GB コンシューマGPUに余裕で収まります。`gradenna.fdtd3d_memory_estimate` で起動前に予算を見積もれます。
- **float32 エンドツーエンド**: トポロジー最適化が素の float32（complex64 DFT）で動作 — コンシューマGPUのネイティブ精度です。強い減衰がある場合のみ `dft_dtype=jnp.complex128` で DFT 累積器だけ高精度化できます。
- **周波数領域随伴（2D・3D）**: 目的関数が周波数領域量（S11・流束・遠方界）のみに依存する場合、`simulate_tm_freq` / `simulate_3d_freq` は**順方向シミュレーション2回**で勾配を計算 — タイムテープ不要、residual は O(設計セル×周波数)。完全AD を正解とした検証で両次元ともコサイン ≥0.9999997（2D方向微分誤差 1.7×10⁻⁵）。**3DのNTFF指向性目的**まで対応 — 実3Dアンテナの微分可能な利得最適化が O(設計セル) メモリで可能。磁気コタンジェント結合定数は閉形式 −ε₀/μ₀（Yeeシンプレクティック計量比）として導出済み。
- **設計領域限定 DFT モニタ（3D）**: `simulate_3d(dft_regions=...)` はランニング DFT を成分ごとの静的スラブ上だけで蓄積し、`freq_adjoint_gradient_3d(objective_kind="port" | "ntff_box" | "field")` がそのスラブを自動導出する — 勾配縮約に使う設計領域の E 成分と、目的関数が読むセルだけ。勾配は全格子パスと一致（コサイン ≥ 1−1e−12）したまま、DFT キャリーは全格子6成分からスラブへ縮小。ネイティブカーネルもスラブを直接蓄積する（64³・1周波数で自身の全格子 DFT パス比 **2.35倍**、M1 実測）。
- **融合 Rust カーネル（2D・3D、オプション）**: タイムループをネイティブカーネルにコンパイル（初回 cargo build、Rust 無し環境は自動フォールバック）。M1 Pro 実測: 2D 1024² float32 **5,040 Mcell-steps/s（XLA比8.4倍）**、3D 96³ **796 Mcell-steps/s（2.4倍、S11掃引経路）**。周波数領域随伴の forward/adjoint も `backend="native"` でカーネル実行（勾配パリティ検証済み）。

## ロードマップ

- [x] Phase 1 — 微分可能 2D TM FDTD コア、CPML、解析解・勾配検証
- [x] Phase 2 — 集中ポート・S11・ランニング DFT モニタ
- [x] Phase 3 — 2D トポロジー最適化（密度法・β 継続）
- [x] Phase 4 — 3D コア、パッチベンチマーク、Gerber 出力、測定ツール
- [x] Phase 5 — 遠方界の指向性・マルチバンド目的
- [x] GPU メモリ最適化（ψ の PML スラブ格納、√N チェックポイント、float32 目的関数）、3D トポロジー最適化
- [x] 周波数領域随伴（勾配＝順方向2回、タイムテープ不要）と融合 Rust CPU カーネル — 2D・3D 両対応
- [x] 設計領域限定 DFT モニタ（DFT 主体の 3D 勾配計算でのカーネル高速化を解禁）
- [x] openEMS クロスチェック参照データ（CSV コミット済み、CI で比較）
- [x] Phase 5 拡張 — アレイ・ビームステアリングデモ、微分可能細線 MoM バックエンド（自由空間）
- [ ] PCB 製造 + NanoVNA 実測キャンペーン — 発注可能な Gerber/ドリル一式と手順書を `fab_campaign/` に用意（プローブ給電のベンチマークパッチ、JLCPCB DRC クリア）。物理的な発注・実測が残り

## ライセンス

MIT
