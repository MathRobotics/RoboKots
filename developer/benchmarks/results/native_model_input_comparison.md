# Rust優先のモデル入力：経路と性能

## 実装

既存の入口にキーワード引数 `backend="rust"` を追加した。

- JSONファイル：Pythonでテキストを読み、Rustのserde_jsonで解析・canonical入力を検証してモデルを生成。
- Python辞書：独立したコピーを保持し、辞書からPyO3を経てRust入力値へ直接変換。JSON文字列への変換は行わない。
- URDF：既存Python readerによるXML解析・慣性変換・トポロジー整列の後、得た辞書をRustへ直接渡す。

いずれもRobotStruct/LinkStruct/JointStruct生成とRobotStruct.to_dictを通らない。
名前・自由度・親子関係はRustから取得した軽量情報を使う。通常のRust状態計算・selected dense/JVP/VJP・RNEA/ABA・次数変更では完全なPythonモデルを展開しない。
NumPy/JAXやrobot_参照など、そのモデルが必要になった場合に一度だけ展開する。ファイルは再読込しない。
Rust crate単体にもJSON文字列・ファイルからの入口を追加した。URDF readerのRust移行は未対応。

## 検証

- 既存APIと入力変更を含むPython回帰：471 passed、2 deselected（slow）、75 warnings。
- 入力の循環参照拒否を加えた最終releaseビルドで、入力経路の26テストが成功（上記と重複する25件を含む）。
- Rust単体：7 unit tests + 5 integration tests成功。PythonなしでJSON読込・モデル生成・計算が可能。
- JSON・辞書・URDF、単体・多次元バッチ、逆順の入力配列、元入力変更、ファイル削除、set_order、NumPyへの切替、直動関節のRNEA/ABA、無効入力を検証。
- RobotStruct.from_dict/to_dictを禁止してRustの主要経路を実行し、旧経路との数値一致を確認。
- git diff --check成功。

## 比較方法

同一の変更後コード・release拡張内で「既存のPythonモデル経由」と「backend='rust'の直接入力」を比較する。異なるコミットのバイナリ比較ではない。
branched_fixed.urdf、3自由度、order=4、float64、seed=346、重力[0.2,-0.3,-9.81]、単体入力。
各操作20 warmup、30標本×20呼出し、AB/BAの交互測定。プロセスを変えて2回実行。
初回費用（import/buildを除く）、全標本、中央値、環境をJSONに記録。
ファイル読込はOSキャッシュが温まった条件。Python/PyO3/NumPyの変換を含む。JITなし。

model_readyはKots・Rust計算用モデル・軽量モデル情報の準備まで。
construct_and_dynamicsはそれにmotion importと初回dynamicsを含む。
残りの計算は同じ生成済みモデルを使い、cached微分はprimal計算済み。
バッチの速度、遅延Pythonモデル展開、メモリ使用量、NumPy/JAXカーネルとの速度比較は対象外。

## 結果

各実行の中央値の範囲。単位はµs/呼出し。変化率も2実行それぞれの比率の範囲で、正が時間増加。

| 操作 | 既存経路 | 新経路 | 時間変化 |
| --- | ---: | ---: | ---: |
| dict/model_ready | 341.41–345.99 | 141.19–143.82 | -58.6〜-58.4% |
| dict/construct_and_dynamics | 392.35–923.02 | 175.40–431.91 | -55.3〜-53.2% |
| json/model_ready | 421.52–426.90 | 92.20–97.43 | -78.1〜-77.2% |
| json/construct_and_dynamics | 459.38–465.15 | 126.48–126.81 | -72.8〜-72.4% |
| urdf/model_ready | 590.92–599.61 | 373.19–378.54 | -36.9〜-36.8% |
| urdf/construct_and_dynamics | 637.30–638.28 | 421.84–423.91 | -33.8〜-33.6% |
| state_values | 18.09–18.41 | 17.98–18.38 | -0.6〜-0.2% |
| dense_cached | 273.71–275.88 | 274.89–276.51 | +0.2〜+0.4% |
| jvp_cached | 31.84–32.47 | 31.57–31.74 | -2.8〜-0.3% |
| vjp_cached | 31.17–31.29 | 31.08–31.48 | -0.3〜+0.6% |
| import_and_dynamics | 15.86–15.87 | 15.87–15.94 | +0.1〜+0.5% |

モデル準備は辞書で約58%、JSONファイルで約77〜78%、URDFで約37%短縮した。
RobotStructの展開と再辞書化を省いた効果であり、モデルのサイズや形式で変わる。
通常計算は概ね±3%以内で、大きな退行は見られない。細かな差を高速化の保証とはしない。
第1実行のdict/construct_and_dynamicsは両経路とも他の測定より大きな値だったため再測定した。
比率は両実行とも短縮しているが、絶対時間には環境の揺れがある。両方の結果を保存し上表に含めた。

状態値・dense・JVP・VJPで最大絶対差は最大2.67e-15、相対Frobenius誤差は最大1.43e-15。
Pythonモデルで座標変換を展開してから再び辞書化する経路を省いたため、完全なビット一致ではなく許容誤差内の一致として確認した。

## 再現

```bash
.venv/bin/python -m developer.benchmarks.native_model_input_compare \
  --output developer/benchmarks/results/native_model_input.json
```

[スクリプト](../native_model_input_compare.py)、[第1実行](native_model_input.json)、[第2実行](native_model_input_repeat.json)。
環境：macOS 15.7.4 arm64、Python 3.13.1、NumPy 2.4.6。スレッド数指定の環境変数は未設定。

## 制約

入力指定を省略した場合は既存経路を維持する。新経路はdim=3、lib='numpy'で使用する。
対応モデルは剛体リンクと固定・回転・直動関節。直動関節のCMTM状態計算は未対応で明示的に拒否する。
モデルは入力のスナップショットとして扱う。変更するときはKotsを作り直す。遅延生成したPythonモデルへの直接編集はRustへ自動同期しない。
一部の非対応微分フォールバックやNumPy/JAX等はPythonモデルを展開する。world空間運動の状態値変換にもPython計算が残る。
