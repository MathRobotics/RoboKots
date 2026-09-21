# RoboKots — エージェント向け作業指示

このファイルは、GPT-6 Astraを含むCodexがRoboKotsを変更・検証するときのプロジェクト指針です。
実行環境の上位指示とユーザーの明示的な依頼を優先してください。
実装状況はコードで確認し、このファイルを未検証の機能保証や作業履歴として扱わないでください。

## 作業の進め方

- 日本語で、結論・変更点・検証結果を簡潔に伝える。数式、配列形状、比較表は判断に役立つ場合に使う。
- 実装の依頼は、調査・実装・必要な検証・結果報告まで進める。計画の提示だけで終了しない。
- 依頼範囲内の可逆的な実装判断は、既存コードと会話から合理的に決める。重大な仕様の曖昧さや権限不足がある場合は、その点だけ確認する。
- 最初に `git status --short` と関連コードを確認する。既存の未コミット変更を保持し、無関係な整形・リファクタリングを混ぜない。
- 検索には `rg` / `rg --files` を使い、関係するモジュールとテストから読む。
- 長い作業では、判明したことと次に確かめることを短く報告する。
- 完了報告では、実装済み・検証済み・未対応を区別する。過去の測定値やテスト成功数を今回の実測として報告しない。

## プロジェクトと実装場所

RoboKotsは、CMTMによる高次運動学・動力学、ヤコビ行列とその積を扱うPythonライブラリです。
NumPyによる解析計算、JAXによる自動微分、PyO3経由のRust実装があります。

| 場所 | 責務 |
|---|---|
| `robokots/kots.py` | 公開ファサード `Kots`、モデル・運動データの操作 |
| `robokots/api/` | 状態、微分、inward/outward、バックエンドの振り分け |
| `robokots/core/` | ロボット構造、運動データ、軸・状態の仕様と共通容器 |
| `robokots/outward/kernels/` | 関節・リンク・全身の計算部品、行列・直接積・微分 |
| `robokots/outward/state.py` | NumPyの運動学・動力学状態計算 |
| `robokots/outward/diff/` | 解析微分、数値微分、JAXの計算 |
| `robokots/outward/rust/` | Rustモデル・状態へのアダプター |
| `robokots/inward/` | 順動力学と関連キャッシュ |
| `robokots/_rust/src/` | RustカーネルとPythonバインディング |
| `tests/` | 単体・API・バックエンド間の回帰テスト |
| `developer/benchmarks/` | 再現可能な比較・性能測定スクリプト |

公開APIの追加では、既存の `api/` の責務分割を使う。アルゴリズム本体を `Kots` に集中させない。
`StateCache` は意味上の状態を保持し、Rustやinwardの計算用ワークスペースとは分離する。
詳細は [README](README.md)、[開発ガイド](developer/README.md)、[モデルJSON仕様](docs/model_json.md)を参照する。

## 数学・APIの整合性

- 配列の末尾軸とバッチ軸を維持する。公開レイアウトは `(..., dof, order)`、平坦形式はowner-major。変換には既存のmotion/axisヘルパーを使う。
- ヤコビ行列は `(..., state_dim, motion_dim)`。状態の指定順、`total_joint` の展開順、`list_output` を一致させる。
- 時間微分と、状態に対するヤコビ行列を区別する。運動量のN階時間微分にはmotion order N+2、力・トルクにはN+3が必要。
- 通常の時間微分列と階乗で正規化したCMTM係数を区別する。変換時に階乗係数を落とさない。
- local/worldの変換では、座標変換自身の微分も含める。リンク・関節を混在指定した場合も同じ規約を使う。
- 重力はworld座標系。`dynamics()` の既定はゼロ重力、`inverse_dynamics()` の既定は `[0, 0, -9.81]`。比較時は明示的に揃える。
- `jacobian()` の解析経路を数値差分に黙って置き換えない。数値差分は `numerical=True` など明示的な参照・検証経路として扱う。
- `Jv` / `Jᵀv` だけが必要な処理では、既存の直接計算経路を優先する。密行列を生成してから掛ける実装への退行を避ける。

## JAXを変更するとき

- 動力学の自動微分本体は `robokots/outward/diff/dynamics_jax.py`、公開入口は `Kots.jacobian_autodiff()`。
- 現行の動力学ADは剛体リンクと固定・回転・直動関節に対応する。柔軟リンク、球面・浮遊関節は明示的な未対応。拡張する場合は対応範囲とテストを一緒に更新する。
- 現行の `jacobian_autodiff()` はJITなしでNumPy配列を返す。JIT測定には純粋なJAX関数を使う。この違いを説明とベンチマークで明記する。
- トレース対象の値を `np.asarray`、Pythonの `float`、データ依存のPython分岐に通さない。静的なモデル定数と動的な入力を分ける。
- JITではmotionを実行時引数として渡す。測定対象のmotionをクロージャ内の定数にしない。
- 微分比較はfloat64で行い、ゼロ姿勢でも有限値になることを確認する。異なる入力でもコンパイル済み関数を検証する。

## 検証と実行環境

既存の `.venv` を優先し、依存関係の再インストールを毎回行わない。初期セットアップが必要なら `uv sync --extra test` を使う。
以下はリポジトリルートから実行するコマンド例。変更に合うものを選び、毎回すべてを実行する必要はない。

```bash
# 動力学ADの変更
.venv/bin/python -m pytest tests/outward/test_dynamics_jax.py -q

# outward/API/微分の変更
.venv/bin/python -m pytest tests/outward tests/test_kots.py tests/test_dynamics_jacobians.py -q

# 高コストの数値差分テストが必要な場合
.venv/bin/python -m pytest tests/test_dynamics_jacobians.py -m slow -q

git diff --check
```

- pytestの既定設定は `slow` を除外する。除外・skip・未実行を成功件数に含めない。
- 数学的な変更では、独立した数値差分や別バックエンドを参照にする。実装をそのまま写した期待値だけで正しさを判断しない。
- 対象に応じて、非ゼロ重力、ゼロ姿勢、分岐、固定・直動関節、高次微分、local/world、バッチを検証する。
- 文書だけの変更では、リンク・記述・差分を確認する。コードのテストや長時間ベンチマークは不要。
- 関連チェックが通ったら、新しい変更・失敗・未解決の疑問がない限りテストを広げ続けない。
- Rustメソッドがソースには存在するのに実行時に見つからない場合、ロード中の拡張とビルドの鮮度を調べる。古いバイナリに合わせてソースを削らない。

Rust拡張の更新が必要な場合は、対象の仮想環境を明示して再ビルドし、新しいPythonプロセスで検証する。

```bash
VIRTUAL_ENV="$PWD/.venv" uvx maturin develop --release --manifest-path robokots/_rust/Cargo.toml
```

## 計算時間・一致度の比較

- 比較対象はPython解析、Rust解析、数値微分、自動微分。JAXはJITなし・初回JIT込み・JIT済みを分ける。
- 同じモデル、motion、重力、出力、微分次数、dtypeを使う。密ヤコビ行列とJVP/VJPを同じ計算として比較しない。
- 状態計算込みと状態計算済みの時間を区別する。Rust指定でもPythonへのフォールバックがある場合は明記する。
- JAXは `block_until_ready` などで同期し、入出力変換を計測に含めたかも記録する。
- 単位を揃え、初回費用、warmup数、測定回数、中央値、環境、乱数seedを記録する。1回測定はそう明記する。
- 一致度は最大絶対差と相対Frobenius誤差を示す。数値差分の方式・刻み幅も記録し、比較対象を厳密解とは呼ばない。
- 実測結果は再現用スクリプトとレポートに保存する。環境依存の時間や一時的な成功件数をこのファイルへ固定しない。

```bash
.venv/bin/python -u -m developer.benchmarks.dynamics_autodiff_compare
```

詳細な実行方法と出力先は [ベンチマークガイド](developer/benchmarks/README.md)を参照する。

## この指示ファイルの保守

長期的に必要な規約と、間違えやすい計算上の前提を残す。作業ログ・TODO・性能表はそれぞれの文書に置く。
上位指示を大量に複製せず、矛盾する常時確認や常時全テスト実行のルールを追加しない。
Astra向けの整理方針は [OpenAI公式モデルガイド](https://developers.openai.com/api/docs/guides/latest-model#prompting-best-practices)、
ファイルの配置は [AGENTS.md公式ガイド](https://learn.chatgpt.com/docs/agent-configuration/agents-md)を参考にしている。
