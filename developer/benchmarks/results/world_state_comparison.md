# world空間運動の値取得とRust公開API

## 変更

- world_link_vec/world_joint_vecをnativeの単体・バッチ状態容器へ追加。PythonのRust adapterと公開get_valueがこれらを使う。
- world変換は、既存selected微分と同じmotion/wrench双対変換とCMTM係数の規約を使用。移動する座標系の時間微分を含め、通常の時間微分値を返す。
- 関節の相対運動は子リンクの座標系でworldへ変換し、子リンクの絶対運動に置き換えない。
- 次数3のキャッシュ・minimal dynamics・固定関節・バッチでも同じ規約を適用。未計算状態・範囲外の番号や次数を拒否する。
- 公開Rust selected APIは整数タプルからStateOutputとStateOwner/StateQuantity/ReferenceFrameへ変更。Python bindingの内部タプルは維持。JVP/VJPの密行列生成はない。
- inverse_dynamics/forward_dynamicsと各batchメソッドをRustCompiledRobotに追加。Python RNEA/ABAも同じnative入口を使う。
- jointのjerkにlocal/worldを明示した場合、空間運動として状態値・dense・JVP/VJPの選択規約を統一。従来のフレーム未指定の関節座標選択は維持。

公開Rust APIの使い方は[実行可能な例](../../../robokots/_rust/src/lib.rs)と[外部利用テスト](../../../robokots/_rust/tests/native_api.rs)を参照。
このRust APIは実験的なもの。従来の外部Rustコードで整数タプルを渡していた場合はStateOutputへの移行が必要。

## 検証

- Python回帰：491 passed、2 deselected（slow）、75 warnings（既存mathrobo等）。
- PythonなしRust：単体7件、外部利用integration7件、ドキュメント内の実行例1件が成功。
- NumPyとの比較、時間方向の中央差分、world状態値のmotion方向中央差分（刻み1e-6）で高次まで検証。
- Python world変換・CMTM view取得を禁止し、Rust状態値が取得できることを確認。
- 固定関節、ゼロ姿勢、非ゼロ重力、次数2/3/6、多次元バッチ、minimal dynamics、空バッチ、更新後の値・返却配列の独立性を検証。
- native RNEAを単関節の独立した式と照合し、ABAの逆算、バッチ一致、無効入力の拒否を検証。
- git diff --check成功。

## 性能

変更前のPython world変換経路と、変更後のnative getter経路を各1セッションで測定。
同一モデルbranched_fixed.urdf、seed=944、float64、motion order=3/6、単体/バッチ(2,3)、重力[0.2,-0.3,-9.81]。
リンクa_tip、関節a_elbow、固定関節b_payload_fixedのworld空間運動を要求。
変更前のjoint jerkには座標との曖昧さがあったため、速度比較ではjoint jerkのみを除外（正しさのテストでは検証済み）。

状態計算済みの値取得と、primalキャッシュ済みdense/JVP/VJPを個別に測定。
10 warmup、20標本×10呼出しの中央値。Python/PyO3/NumPy変換・公開APIの配列コピーを含む。
初回費用と全標本、環境、拡張のSHA256はJSONに保存。JITなし、初期化・ファイル読込・RNEA/ABAの単独速度はこの比較に含めない。

単位：µs/呼出し。負の変化率が時間短縮。

| 操作 | 変更前 | 変更後 | 変化 |
| --- | ---: | ---: | ---: |
| order3/batch()/world_values | 28.71 | 12.16 | -57.6% |
| order3/batch()/dense_cached | 67.19 | 65.15 | -3.0% |
| order3/batch()/jvp_cached | 14.49 | 14.08 | -2.9% |
| order3/batch()/vjp_cached | 15.76 | 15.48 | -1.8% |
| order3/batch(2, 3)/world_values | 33.03 | 22.00 | -33.4% |
| order3/batch(2, 3)/dense_cached | 299.09 | 295.51 | -1.2% |
| order3/batch(2, 3)/jvp_cached | 42.85 | 42.54 | -0.7% |
| order3/batch(2, 3)/vjp_cached | 42.40 | 43.08 | +1.6% |
| order6/batch()/world_values | 66.53 | 27.69 | -58.4% |
| order6/batch()/dense_cached | 303.28 | 307.00 | +1.2% |
| order6/batch()/jvp_cached | 30.33 | 30.74 | +1.3% |
| order6/batch()/vjp_cached | 32.25 | 32.87 | +1.9% |
| order6/batch(2, 3)/world_values | 86.16 | 53.90 | -37.4% |
| order6/batch(2, 3)/dense_cached | 1663.54 | 1667.39 | +0.2% |
| order6/batch(2, 3)/jvp_cached | 109.38 | 108.88 | -0.5% |
| order6/batch(2, 3)/vjp_cached | 105.26 | 104.88 | -0.4% |

world状態値の取得時間は約33〜58%短縮。dense/JVP/VJPは概ね±3%以内だった。
単発セッション間の比較なので、小さい差を高速化・退行の確定値とはしない。
状態値の最大絶対差は5.55e-17、相対Frobenius誤差は最大1.39e-16。dense/JVP/VJPは完全一致した。

## 再現と記録

```bash
.venv/bin/python -m developer.benchmarks.world_state_values --output developer/benchmarks/results/world_state_after.json
.venv/bin/python -m developer.benchmarks.world_state_values --reference --output /tmp/world_state_python_reference.json
```

[スクリプト](../world_state_values.py)、[変更前の記録](world_state_before.json)、[変更後の記録](world_state_after.json)。
--referenceは旧Python変換を使う比較用入口。現在の公開dispatch上で動くため、変更前スナップショットの呼出し費用を厳密に復元するものではない。
環境はmacOS 15.7.4 arm64、Python 3.13.1、NumPy 2.4.6。
URDF reader、Python互換CMTM view、モデル交換時の同期、未対応関節の拡張は今回の対象外。
