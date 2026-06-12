# gradenna

[![CI](https://github.com/s-hosomi/gradenna/actions/workflows/ci.yml/badge.svg)](https://github.com/s-hosomi/gradenna/actions/workflows/ci.yml)

**grad**ient + ant**enna** — JAX で書いた微分可能 FDTD によるアンテナ逆設計。RF アンテナを勾配降下で「育てる」ツールキットです。

[English README](README.md)

<p align="center">
  <img src="assets/optimization.gif" width="460" alt="2.45 GHz アンテナが一様グレーから勾配降下で育っていく様子"/>
</p>
<p align="center">
  <em>一様グレーの設計領域からアンテナが生えてくる。104×104 ピクセル（1 mm セル）の設計領域を、
  微分可能な Maxwell ソルバーを通して Adam で 200 イテレーション最適化し、2.45 GHz の放射電力を
  最大化したもの。目的関数の値は初期状態の <strong>58,000 倍</strong>に達します。
  形状は一切、人間が描いていません。</em>
</p>

gradenna は、RF・マイクロ波アンテナ設計のための完全微分可能な電磁界（FDTD）ソルバーとトポロジー最適化ツールキットです。Yee 更新、CPML 吸収境界、50 Ω 集中ポート、ランニング DFT による S パラメータ、近傍場—遠方場変換（NTFF）まで、シミュレーション全体がひとつの JAX 計算グラフとして書かれています。そのため `jax.grad` を呼ぶだけで、任意の目的関数（S11、放射電力、指向性、利得）について、**設計領域の全ピクセルに対する**厳密な随伴勾配が一度に得られます:

```python
import jax, jax.numpy as jnp
from gradenna import (Grid2D, CPMLSpec, Port, simulate_tm,
                      gaussian_pulse_for_band, sigma_from_density,
                      poynting_flux_box_2d)

grid = Grid2D(nx=140, ny=140, dx=2e-3, dy=2e-3)
pulse = gaussian_pulse_for_band(2.0e9, 3.0e9)
t = (jnp.arange(3500) + 0.5) * grid.dt

def neg_radiated_power(rho):                      # rho: 0 = 空気, 1 = 銅
    sigma = jnp.zeros(grid.shape).at[44:96, 60:112].set(sigma_from_density(rho))
    res = simulate_tm(grid, sigma=sigma, dft_freqs=(2.45e9,),
                      ports=(Port(ij=(70, 55), resistance=50.0, voltage=pulse(t)),),
                      cpml=CPMLSpec(thickness=10))
    return -poynting_flux_box_2d(res.dft_ez, res.dft_hx, res.dft_hy, grid,
                                 box=(20, 120, 20, 120))[0]

grad = jax.grad(neg_radiated_power)(0.5 * jnp.ones((52, 52)))  # 逆方向パス 1 回
```

## インタラクティブ・ビジュアライザ

**[ライブデモを開く →](https://s-hosomi.github.io/gradenna/)** — ローカルで動かす場合は
`cd web/app && npm install && npm run dev`。上記の内容をすべてブラウザで体験できる
three.js 製のビューアで、**26 kB の Rust→wasm カーネルがブラウザの中で 2D FDTD を
リアルタイムに解く**デモも含まれています（表示用データは生成済みのものをリポジトリに同梱）。

<table>
  <tr>
    <td width="50%">
      <img src="assets/viewer_optimization.png" alt="トポロジー最適化の再生"/>
      <p align="center"><sub><b>Optimization</b> — 冒頭の成長アニメーションを再生・スクラブ
      できるビュー。密度マップは GPU 上でバイキュービック補間され、収束カーブ（対数スケール）
      と連動します</sub></p>
    </td>
    <td width="50%">
      <img src="assets/viewer_live_fdtd.png" alt="ブラウザ内ライブ FDTD"/>
      <p align="center"><sub><b>Live FDTD</b> — 最適化で得たアンテナを wasm カーネルが
      その場で駆動。時間平均強度の表示に切り替えると、指向性ビームがはっきり浮かび上がります。
      銅を描き込んだり、ソースを動かしたり、ダブルスリットやミラーのシーンも</sub></p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <img src="assets/viewer_antenna3d.png" alt="Antenna 3D: 形状・近傍場・遠方界"/>
      <p align="center"><sub><b>Antenna 3D</b> — 実寸のパッチアンテナの上に、3D 計算で得た
      |E| 近傍場が半透明のスライス面として光ります（放射端のフリンジング場が見えます）。
      その上空には遠方界ローブ。形状から近傍場、遠方界までを、回転できるひとつの
      3D シーンにまとめました</sub></p>
    </td>
    <td width="50%">
      <img src="assets/viewer_s11.png" alt="S11 の openEMS との重ね描き"/>
      <p align="center"><sub><b>S11</b> — 同じアンテナを gradenna と独立ソルバー openEMS で
      解いた比較。共振周波数の差 0.83%、カーブの RMS 差 0.51 dB</sub></p>
    </td>
  </tr>
</table>

表示用データは `scripts/export_viz.py` で再生成できます。wasm のビルド方法と
GitHub Pages へのデプロイ手順は `web/app/README.md` を参照してください。

## 特徴

- **微分可能な 2D TM / フル 3D の FDTD コア**（`jax.lax.scan` ベース、jit 対応、float32/float64）。CPML/CFS 吸収境界、マルチソース、√N 勾配チェックポイントによるメモリ制御
- **集中 RVS ポートと S パラメータ**: 半陰解法の抵抗付き電圧源ポート、位相を厳密に扱うランニング DFT、パワー波定義の S11、離散ギャップのサセプタンスを除去するデエンベディング
- **微分可能な NTFF**（2D/3D）: 放射電力・指向性・利得をそのまま最適化の目的関数にできます
- **トポロジー最適化の道具一式**: conic 密度フィルタ、tanh 射影と β 継続、対数スケールの導電率補間、連結性・最小線幅のチェック
- **製造パイプライン**: 密度マップ → ポリゴン → RS-274X Gerber（JLCPCB のデザインルールチェック付き、`pip install gradenna[fab]`）。ベンチマークパッチはそのまま発注できるパッケージを `fab_campaign/` に用意
- **実測との突き合わせ**: Touchstone ファイルの入出力、シミュレーションと実測の S11 比較、NanoVNA の測定スクリプト（`pip install gradenna[measure]`）
- **微分可能な細線 MoM バックエンド**（`gradenna.mom`）: 自由空間 PEC ワイヤを区分正弦基底の Galerkin EFIE で解く高速なサロゲート。長さや半径といった形状パラメータを `jax.grad` で微分できます。誘電体基板への対応（層状媒質グリーン関数）は今後の課題

## 手元のマシンで速く動く — GPU と Apple Silicon

微分可能な RF FDTD の先行研究は、データセンター級の GPU（中解像度の 3D アンテナで
GPU あたり約 90 GB）を前提にしていました。gradenna は、同じ問題が**手元にある普通の
ハードウェアで回る**ことを設計目標にしています:

| | |
|---|---|
| 3D パッチのトポロジー最適化（フル解像度） | ピークメモリ **7.7 GB**（float64）/ **約 3.9 GB**（float32）— 24 GB のコンシューマ GPU に収まります |
| 3D の CPML 補助変数（ψ）のメモリ | PML スラブ格納により **−74%** |
| 随伴勾配のメモリ | 残差は **O(設計セル数 × 周波数点数)** — 時間方向のテープを一切持ちません（周波数領域随伴） |
| 2D Rust カーネル（Apple M1 Pro） | 1024²・float32 で **5,040 Mcell-steps/s** — XLA CPU 比 **8.4 倍** |
| 3D Rust カーネル（Apple M1 Pro） | 96³ で **796 Mcell-steps/s**（2.4 倍）。DFT が重い勾配計算は、領域限定 DFT でさらに **2.35 倍** |

- **メモリを抑えた 3D 随伴**: ψ の PML スラブ格納と √N 勾配チェックポイントの組み合わせ。`gradenna.fdtd3d_memory_estimate` が実行前に必要メモリを見積もり、`examples/optimize_3d_patch.py` の `gpu-24gb` プリセットはその値を表示して assert までします。
- **float32 で一気通貫**: トポロジー最適化は float32 のまま（DFT は complex64）動きます — コンシューマ GPU の素の精度です。ソースとモニタの間の減衰が極端な場合だけ、`dft_dtype=jnp.complex128` で DFT の蓄積部分のみ倍精度に上げられます。
- **周波数領域随伴（2D・3D）**: 目的関数が周波数領域の量（S11、放射束、遠方界）だけで決まる場合、`simulate_tm_freq` / `simulate_3d_freq` は**順方向シミュレーション 2 回分**のコストで勾配を計算します。時間方向のテープは不要です。完全な自動微分を正解とした検証で、2D・3D ともコサイン類似度 0.9999997 以上。**3D の NTFF 指向性を目的関数にした場合**まで検証済みで、実際の 3D アンテナの利得最適化が O(設計セル数) のメモリでできます。磁気コタンジェントの結合定数は閉形式 −ε₀/μ₀（Yee 格子のシンプレクティック計量比）として導出してあります。
- **設計領域限定の DFT モニタ（3D）**: `simulate_3d(dft_regions=...)` はランニング DFT を成分ごとの小さなスラブ上だけで蓄積します。`freq_adjoint_gradient_3d(objective_kind="port" | "ntff_box" | "field")` なら、必要なスラブ — 勾配の縮約に使う設計領域の E 成分と、目的関数が実際に読むセル — を自動で割り出します。勾配は全格子で計算した場合と一致（コサイン類似度 1−1e−12 以上）したまま、DFT が保持するデータは全格子 6 成分からスラブだけに減ります。
- **Rust 製ネイティブカーネル（オプション、ARM 向けに調整済み）**: 2D/3D の時間発展ループを、キャッシュ効率のよいマルチスレッドのネイティブカーネルとして実行できます（初回利用時に cargo でビルド。Rust が無い環境では自動で JAX 実装にフォールバック）。スレッド数やタイリングは Apple Silicon で実測しながら調整しました。周波数領域随伴は forward・adjoint の両方をカーネル上で実行でき（`backend="native"`）、勾配の一致もテストしています。数値は `scripts/benchmark.py --backend native` で再現できます。

## 検証

物理に関わるすべてのコンポーネントを、解析解・教科書値・独立ソルバーと突き合わせて CI でテストしています（175 本以上）:

| ベンチマーク | 結果 |
|---|---|
| 線電流の円筒波 vs `H0^(2)(kρ)`（Harrington） | プロファイル誤差 2.5% 未満、2 次のグリッド収束 |
| CPML の反射（領域を拡大した参照解との比較） | −92 dB（要求仕様は −60 dB） |
| 2D 線電流の放射抵抗 vs ωμ₀/4 | 1.3–2.4% |
| 微小ダイポールの放射抵抗 vs 80π²(l/λ)² | デエンベディング後 0.34% |
| NTFF 経由のダイポール指向性 vs D₀ = 1.5 | 0.14% |
| 2.45 GHz FR-4 パッチの共振 vs Balanis の設計式 | −2.5% |
| 2.45 GHz パッチ vs openEMS（参照データを同梱） | 共振 0.83%、\|S11\| の RMS 差 0.51 dB、パターン相関 0.999 以上 |
| 細線 MoM のダイポール共振 vs 教科書値（0.47–0.48 λ、約 72 Ω） | L_res = 0.476 λ、Re Zin = 71.7 Ω |
| ビームステアリングの主ローブ vs アレイファクタ理論 | −30°/0°/+30° で誤差 ±5° 以内 |
| `jax.grad` vs 有限差分(すべてのパラメータ種別) | 相対誤差 1e-4 以下 |
| チェックポイントあり随伴 vs 通常の随伴 | ビット単位で一致 |

## デモ

| スクリプト | 内容 |
|---|---|
| `examples/optimize_2d_antenna.py` | 成長デモ。2.45 GHz の放射エネルギーを最大化し、最終設計は完全に二値化されます（冒頭の GIF は同じ問題を 1 mm 解像度で解いたもの — `scripts/export_viz.py`） |
| `examples/optimize_directivity.py` | 遠方界変換を通したビーム整形: D(0°) が 0.31 → 4.47、F/B 比 16.8 dB |
| `examples/optimize_multiband.py` | 2.0 GHz と 3.0 GHz を同時に狙う、最悪帯域（softmin）の放射電力最適化 |
| `examples/optimize_beamsteering.py` | **ビームステアリング**: 4 素子 λ/2 アレイの複素給電重みを、遠方界変換を通して最適化 |
| `examples/optimize_3d_patch.py` | **3D トポロジー最適化**: 実際の FR-4 積層構成の上で銅の密度を最適化（チェックポイント随伴）。`--preset cpu-demo`（約 2.5 分で放射電力 39 倍）と `--preset gpu-24gb` |
| `examples/patch_to_gerber.py` | Balanis 式でのパッチ設計 → 密度マップ → DRC チェック → Gerber 出力 |

## クイックスタート

```bash
git clone https://github.com/s-hosomi/gradenna && cd gradenna
uv sync                                  # または: pip install -e ".[fab,measure]"
uv run pytest -m "not slow" -q           # 高速の検証スイート（CPU で 1〜2 分）
uv run python examples/optimize_2d_antenna.py
```

CPU だけでそのまま動きます（どのデモも数分で終わります）。JAX の GPU/TPU バックエンドもコード変更なしで使えます。

## なぜ微分可能 FDTD なのか

50×50 の設計領域には 2,500 個の自由度があります。勾配を使わない手法（GA やピクセル反転）では 1 世代ごとに数千回のシミュレーションが必要になりますが、随伴法なら — そして leapfrog 型の Maxwell ソルバーでは reverse-mode 自動微分が随伴法を自動かつ厳密に実行してくれます — パラメータがいくつあろうと、**およそシミュレーション 2 回分のコストで全勾配**が手に入ります。gradenna は、フォトニクスの逆設計で実績のあるこの仕組みを RF 帯に持ち込みます。RF では導体損失・集中給電・製造制約によって、問題の性質がフォトニクスとは変わってくるのです。

## ロードマップ

- [x] Phase 1 — 微分可能な 2D TM FDTD コア、CPML、解析解と勾配の検証
- [x] Phase 2 — 集中ポート、S11、ランニング DFT モニタ
- [x] Phase 3 — 2D トポロジー最適化（密度法、β 継続）
- [x] Phase 4 — 3D コア、パッチベンチマーク、Gerber 出力、測定ツール
- [x] Phase 5 — 遠方界の指向性・マルチバンドの目的関数
- [x] GPU メモリ最適化（ψ の PML スラブ格納、√N チェックポイント、float32 の目的関数）と 3D トポロジー最適化
- [x] 周波数領域随伴(勾配 = 順方向 2 回、時間テープ不要)と Rust 製 CPU カーネル — 2D・3D の両方
- [x] 設計領域限定の DFT モニタ（DFT が重い 3D 勾配計算でカーネルの速さを活かせるように）
- [x] openEMS とのクロスチェック用参照データ（CSV を同梱し CI で比較）
- [x] Phase 5 拡張 — アレイのビームステアリングデモ、微分可能な細線 MoM バックエンド（自由空間）
- [x] Web ビジュアライザ — three.js ビューアと Rust→wasm のブラウザ内 FDTD カーネル
- [ ] PCB 製造と NanoVNA 実測のキャンペーン — 発注できる状態の Gerber/ドリル一式と手順書は `fab_campaign/` に用意済み（プローブ給電のベンチマークパッチ、JLCPCB の DRC クリア）。残るのは実際の発注と測定のみ

## ライセンス

MIT
