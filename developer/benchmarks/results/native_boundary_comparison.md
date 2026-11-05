# Python依存分離の実測比較

> 追調査：同一プロセスの交互測定では計算の2〜4%遅延は再現せず、モデル生成の約0.5 µs増加は残った。詳細は[遅延の切り分け](native_boundary_diagnosis.md)。以下は当初の測定記録を維持している。

モデル構築と状態容器を通常のRust型へ移し、Python公開用クラスがそれらを直接保持する形に変更した。
辞書の抽出・NumPy入出力・Python例外はバインディングに残す。型付きモデルへの変換には一時的なVecの確保が追加され、モデル構築時に木構造の検証も行う。計算カーネルの数式は変更していない。

## 測定条件

- 変更前：コミット ca662cba0117e50778d46afdf56304fd8713e094 のRustソースを /tmp へ取り出してreleaseビルド。
- 変更後：今回の未コミットのRustソースをmaturinでreleaseビルド。
- 両方とも Cargo.toml の release 設定（codegen-units=1、LTO=thin）。同じPython環境から同じAPIを呼び出す。
- 最終測定は変更前・変更後の順で逐次実行。テスト・ビルドとの同時実行なし。測定後は変更後の拡張へ復元。
- 環境：macOS-15.7.4-arm64-arm-64bit-Mach-O / Python 3.13.1 / NumPy 2.4.6。
- モデル：branched_fixed.urdf、乱数seed=812、float64、非ゼロ重力 [0.2, -0.3, -9.81]。
- モデル生成：10 warmups、50 samples × 50 calls。URDF読込・プロセス起動・import時間は対象外。
- 状態・微分：motion order=4、5 warmups、30 samples × 5 calls。world空間運動・姿勢・動力学の混在要求。
- 状態計算はmotion取込み込み。微分は計算済み状態・ウォーム済み微分キャッシュを使用。Python APIの呼出し・配列変換は含む。
- 初回呼出しと全サンプル、拡張のSHA256、計算結果は対応するJSONへ保存。表は中央値、単位はµs。
- JAXは変更対象外のため今回未測定。数値差分は正確性の検証に使用。

## 前後比較

時間増減は (変更後 / 変更前 - 1) × 100。正の値が遅くなった方向。

| 処理 | 変更前 µs | 変更後 µs | 時間増減 |
| --- | ---: | ---: | ---: |
| モデル生成（辞書を準備済み） | 11.193 | 11.627 | +3.88% |
| Pythonモデルの辞書化＋モデル生成 | 58.914 | 60.738 | +3.10% |
| rust/single/import_and_dynamics | 16.179 | 16.358 | +1.11% |
| rust/single/dense | 277.225 | 288.275 | +3.99% |
| rust/single/jvp | 36.533 | 37.354 | +2.25% |
| rust/single/vjp | 37.271 | 38.275 | +2.69% |
| rust/batch2/import_and_dynamics | 22.596 | 22.104 | -2.18% |
| rust/batch2/dense | 523.225 | 530.379 | +1.37% |
| rust/batch2/jvp | 60.908 | 61.008 | +0.16% |
| rust/batch2/vjp | 60.454 | 61.379 | +1.53% |

モデル生成は約0.43 µs増加。単体微分は約2〜4%の増加、バッチ計算は約−2〜+2%だった。
この測定では高速化は確認していない。変更前・変更後それぞれ一回の測定セッションなので、数%の差を恒常的な性能差とは断定しない。

## 正確性と検証

- 前後比較のRNEA、dense、JVP、VJP：最大絶対差0、相対Frobenius誤差0。
- 空間微分ベンチマークはNumPy・Rust・中心差分（刻み1e-8）を比較し、全assert成功。各誤差は python_boundary_after.json / .md に記録。
- Python回帰：405 passed、slowの2件はdeselected。既存mathroboからDeprecationWarningが75件。
- Pythonなし：cargo test --offline --manifest-path robokots/_rust/Cargo.toml --no-default-features で7 passed。
- cargo tree --no-default-features はこのcrateのみで、PyO3・NumPy依存なし。
- git diff --check 成功。コミットは行っていない。

## 範囲

モデル・状態容器の型と生成処理にはPython依存がない。Pythonクラスはpy_api.rsのラッパーとして維持する。
独立Rustライブラリとしての全計算API公開は今回の対象外。crate-privateの計算メソッドと、バインディング内の高水準の状態更新・出力処理は残る。

## 再現用ファイル

- [モデル測定スクリプト](../native_model_boundary.py)
- [状態・微分測定スクリプト](../spatial_selected_outputs.py)
- [モデル・変更前](native_model_before.json) / [変更後](native_model_after.json)
- [状態と微分・変更前](python_boundary_before.json) / [変更後](python_boundary_after.json)
- [NumPyと数値差分との一致度](python_boundary_after.md)
