# Native状態API・軽量モデル情報の変更比較

対象：基準 `1e31b3e` と今回の未コミット変更。Rust releaseビルド。

## 変更と検証

- 名前・自由度・motion配置・親子関係を保持するModelInfoをnativeに追加。PythonのRust状態adapterと微分要求の解決を軽量情報へ切り替えた。
- outward/batch compute・主要getter、selected JVP/VJP、ABA prepare/solveをPythonなしで呼べるAPIへ移動。計算カーネルは共通。
- Python回帰：410 passed、2 deselected（slow）、75 warnings（既存mathrobo等の警告）。
- PythonなしCargo検証：7 unit tests + 3 integration tests成功。外部crate相当のテストで中央差分（刻み1e-6）、直接JVP/VJP、ABA、バッチ、キャッシュ、無効入力を確認。
- RobotStructへの参照を禁止したPythonテストで、Rust状態の読み取り・辞書出力・selected積・状態容器の再生成を確認。
- git diff --check成功。

## 性能

branched_fixed.urdf、order=4、float64、seed=812、重力[0.2,-0.3,-9.81]。
各操作20 warmup、30標本×50呼出し。新しいプロセスを変更前→後→後→前の順に2巡し、各側4セッションの中央値をさらに集計。
変更前はPythonコードも基準コミットを使用。rawにもPyO3/NumPy変換を含む。JITなし。
状態更新はmotion importとdynamicsを含む。cached微分はprimal計算済みで、dense/JVP/VJPを別々に比較する。
この測定にはバッチgetterの性能、ファイル読込、初回import・ビルド費用、メモリ使用量は含めない。

単位：µs/呼出し。変化率は正が時間増加。

| 操作 | 変更前 | 変更後 | 変化 |
| --- | ---: | ---: | ---: |
| control/serialize_only | 47.388 | 47.538 | +0.3% |
| model/compile_dict | 11.586 | 14.776 | +27.5% |
| model/serialize_and_compile | 60.741 | 65.129 | +7.2% |
| raw/model_dof_getter | 0.052 | 0.051 | -0.8% |
| raw/rnea | 1.827 | 1.799 | -1.5% |
| raw/compute_dynamics | 2.689 | 2.631 | -2.1% |
| public/import_and_dynamics | 15.737 | 15.695 | -0.3% |
| raw/dense_cached | 249.443 | 247.272 | -0.9% |
| raw/jvp_cached | 21.951 | 21.854 | -0.4% |
| raw/vjp_cached | 19.894 | 19.773 | -0.6% |
| public/dense_cached | 286.900 | 285.796 | -0.4% |
| public/jvp_cached | 37.227 | 37.548 | +0.9% |
| public/vjp_cached | 38.955 | 38.577 | -1.0% |

通常計算は概ね±2%程度で、大きな退行は今回の条件では見られない。微小差を高速化の保証とはしない。
モデル辞書からの構築には約3.2µs（27.5%）の増加がある。追加された名前抽出・検証・メタデータ生成が要因と考えられるが、内訳ごとの計時はしていない。
Python辞書化も含めたモデル構築は約4.4µs（7.2%）増。ファイルからPythonモデルを生成する費用はこの値に含まない。
比較したRNEA・dense・JVP・VJPの出力は変更前後で完全一致し、最大絶対差・相対Frobenius誤差はともに0。

初回の非交互測定では公開JVP/VJPが12〜14%増加したため、モデル情報取得を出力ループ外へ移し、キャッシュ利用時のimportを省いた。
その後の単発測定では変更していない処理も大幅に遅くなったため、単発値から効果を断定せず、上記の交互測定を最終結果とした。
単発のbefore/after/after_initial JSONも調査記録として残す。最適化単独の寄与は未分離。

## 再現

[測定スクリプト](../native_boundary_diagnosis.py)に `--before-root` を追加し、基準Pythonコードと基準拡張を併用できるようにした。
基準チェックアウト（developer/__init__.py、developer/benchmarks、robokots、tests/test_modelを含む）と、そのrelease拡張を用意して実行する。

```bash
.venv/bin/python -m developer.benchmarks.native_boundary_diagnosis \
  --before-root /tmp/robokots-native-api-baseline \
  --before /tmp/robokots-native-api-baseline/robokots/_rust_core.cpython-313-darwin.so \
  --after robokots/_rust_core.cpython-313-darwin.so \
  --output developer/benchmarks/results/native_api_sessions.json
```

拡張名は環境に合わせる。集計・全セッション・拡張ハッシュ・配列出力は[native_api_sessions.json](native_api_sessions.json)と参照先JSONに保存。
環境はmacOS 15.7.4 arm64、Python 3.13.1、NumPy 2.4.6。スレッド数指定環境変数は未設定。

## 未対応

Kots初期化・ファイル読込時のRobotStruct構築、全入力検証の移行、モデル交換に伴うキャッシュ無効化、world空間運動の状態値変換は引き続き課題。
Rust CMTMの対応範囲は固定・回転関節。直動関節のRNEA/ABA対応は維持し、未対応のCMTM容器生成は明示的に拒否する。
