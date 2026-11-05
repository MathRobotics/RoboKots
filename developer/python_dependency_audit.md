# Python依存の棚卸し

棚卸し対象：`1e31b3e` の実装。本文は調査時点の記録で、後続の変更は「棚卸し後の実装状況」に記す。
目的は、Rustでモデル構築・状態計算・微分を完結させ、Rustのみの利用ではPythonにモデル全体を展開しない構成への移行点を明らかにすること。

## 棚卸し後の実装状況

以下の棚卸し本文は基準コミット時点の記録として維持する。
その後、nativeに名前・自由度・親子関係のModelInfoを追加し、PythonのRust状態adapterとRust微分選択を軽量情報へ切り替えた。
outward/batchのcompute・主要getter、selected apply/cache_info、ABA prepare/solveをnative APIとして公開した。
外部crateのintegration testと、RobotStruct参照を禁止したPythonテストを追加している。
D04〜D09の主要経路が前進したが、D01のKots生成、D02の入力形式・検証の移行、D11のモデル変更通知、world空間運動の値変換は未完了。

### 入力入口の追加変更

`backend="rust"`付きのJSON/辞書/URDF入力は、RobotStruct生成・再辞書化を省略する。
JSON解析とcanonical入力の検証はnativeへ移し、URDFは既存Python readerの出力を直接渡す。
完全なPythonモデルはNumPy/JAX等の利用時に一度だけ展開する。Rust-only crateからのJSON読込も追加した。
URDF reader自体のRust移行、モデル編集・交換時の同期、残るworld状態値変換は未対応。
詳細と速度比較は[開発ガイド](README.md#native-first-model-input)を参照。

### world状態値・Rust公開APIの追加変更

world空間運動の値取得をnativeへ移し、公開Python経路もそのgetterを使うようにした。
Rustの出力指定をStateOutput/StateOwner/StateQuantity/ReferenceFrameに整理し、RNEA/ABAの単体・バッチ入口を公開した。
詳細と検証は[開発ガイド](README.md#native-world-values-and-typed-calculation-api)を参照。
過去の節に残るworld状態値変換の未対応記述は、この変更前の記録である。

## 結論

計算カーネルとモデル・状態容器はすでにPythonから分離されている。一方、公開API全体がPythonから独立したわけではない。
残る依存は次の三種類に分かれる。

1. **Pythonの完全なモデル構造への依存**：読込、名前解決、motion配置、状態アダプター、対応可否判定。
2. **Python側に残る計算・制御への依存**：world状態値の変換、出力要求の解決、状態更新・キャッシュ操作、フォールバック。
3. **Pythonを利用窓口にするための依存**：PyO3、NumPy入出力、Python例外。Python APIでは残してよく、native側へ漏らさない。

最初に取り組むべきものは、名前・番号・自由度・接続情報を提供する軽量なモデル情報と、native側の状態操作の入口。
入力をJSONに変更するだけでは1と2は解消しない。

## 確認方法と範囲

- Kotsの生成からRustモデル生成、状態更新・取得、dense/JVP/VJPの主要経路を静的に追跡。
- inward、whole-body、NumPy/JAX、可視化・出力は境界と代表的な呼出し箇所を確認。全内部アルゴリズムを網羅した監査ではない。
- RustソースでPyO3/NumPy依存を検索した結果、該当したのは `py_api.rs` と `lib.rs` のみ。
- `cargo tree --offline --manifest-path robokots/_rust/Cargo.toml --no-default-features` を今回実行し、このcrate以外の依存がないことを確認。
- 性能再測定・コード変更・全経路の動的トレースは行っていない。モデル直接編集時の挙動も未再現で、下記はコードからの指摘。

## 現在の生成・計算経路

```text
JSON / URDF / Python辞書
  → Pythonで読込・検証・並べ替え
  → RobotStruct / LinkStruct / JointStruct / SE3 / NumPy配列
  → RobotStruct.to_dict() と配列のリスト化
  → PyDictからRustのRobotModelへ抽出
  → RustCompiledRobot
  → native計算 + PyO3状態操作
  → Pythonで名前・出力・配列形状を解決
```

[Kots](../robokots/kots.py) の `from_json_file/from_json_data/from_urdf_file` はすべて `RobotStruct.from_dict` を呼ぶ。
コンストラクターも `robot.dof` と `robot.motion_owners()` を必要とするため、現在はファイル読込だけRustへ移してもKotsをそのまま初期化できない。
[RobotStruct](../robokots/core/robot.py) はリンク・関節ごとの配列、選択行列、SE3、名前辞書、接続リスト、自由度番号を生成する。
[outward/rust/model.py](../robokots/outward/rust/model.py) はそのモデルを再度辞書化してRustへ渡す。

## 箇所別の棚卸し

P0：Pythonモデルを省く前に整える。P1：通常のRust専用経路を成立させる。P2：必要な機能から後続対応。

| ID | 箇所・入口 | 現在の依存 | 移行方針 | 優先 |
| --- | --- | --- | --- | --- |
| D01 | `kots.py`: 各factory、`__init__` | 完全なRobotStructを必須とする | nativeモデルと軽量情報から初期化する経路を追加。既存Python経路は当面維持 | P0 |
| D02 | `core/robot.py`: `validate_model_data/from_dict`、`robot_io.py`、`urdf_io.py` | schema・名前・ID・数値検証、ソート、URDF変換 | 入力アダプターと共通native検証へ移す。仕様差を先に整理 | P0 |
| D03 | `outward/rust/model.py`、`api/rust_backend.py`: compile・対応判定 | 辞書化、rigid判定、関節型判定、Pythonのリンク・関節一覧 | nativeモデル構築を正本にし、演算ごとの対応可否を問い合わせる | P0 |
| D04 | `kots.py`: `motion_owners/_active_joint_names/_joint_motion_state_info_list` | dof、dof_index、owner順、関節名 | 軽量情報からmotion配置・total_joint展開を作る | P0 |
| D05 | `core/state/spec.py`: `state_output`、`api/rust_derivatives.py` | `robot.link/joint()` が返すPythonオブジェクトのid・dof、配列順・オブジェクト同一性 | 名前→内部番号、自由度、出力仕様で解決。全JointStructを代理実装しない | P0 |
| D06 | `outward/rust/state.py`、`outward/rust/data.py`: state生成 | 名前一覧、joint_dofs、robot参照 | 軽量情報とraw_dataだけで状態を生成。計算用モデルの複製をなくす | P1 |
| D07 | `outward/rust/data.py`: `state_value`、`outward/state.py`: `get_value` | world空間運動のPython変換、子リンク参照、SE3/CMVector/CMTMビュー | 数値状態取得をnativeへ。mathrobo形式はPythonの互換ビューとして残す | P1 |
| D08 | `_rust/src/py_api.rs`: outward/batchのcompute・getter | 次数分岐、状態フラグ、遅延確保、値の取得がpymethods内 | nativeメソッドへ移し、bindingは配列・例外変換に限定 | P1 |
| D09 | 同ファイル：`RustSelectedWorkspace.apply` | 要求検証、primal再利用、バッチ・RHS確保、nativeカーネル呼出しをbinding内で制御 | nativeのapply/into等へ移す。JVP/VJPの密行列非生成を維持 | P1 |
| D10 | `api/derivatives.py`、`api/fast_derivatives.py` | Rust不適用時のPython解析、座標出力混在の合成、数値差分・JAX | Rust専用対応範囲を明示。残すPython機能では必要時のモデル生成を設計 | P1/P2 |
| D11 | `api/state.py`、`api/state_cache.py`、`api/rust_backend.py`、`inward/cache.py` | motion revision、重力、次数、状態公開、Pythonの結果キャッシュ | 意味上のStateCacheはPython側でもよい。モデル交換とnative workspace寿命を統一 | P0/P1 |
| D12 | `api/inward.py`、`_rust/src/rust_data.rs` | RNEA/ABA入口の形状・重力処理はPython。ABA prepare/solve本体はnativeに移行済みだがcrate-private | 小さいnative公開APIの最初の対象に向く。reference経路は別機能として維持 | P1 |
| D13 | `api/whole_body.py`、`api/outward.py` | 質量・重心・形状情報を使うPython計算、NumPy状態生成 | Rust専用の最初の範囲から区別。必要なら計算をnativeへ移す | P2 |
| D14 | `state_io/dictionary.py`、Kotsの可視化・軌跡・数値微分 | 名前・dofだけで足りる処理と、完全モデルが必要な処理が混在 | exportや骨格表示は軽量情報で対応可能。計算モデル生成を一律に要求しない | P2 |
| D15 | `_rust/src/lib.rs/py_api.rs`、Python import群 | PyO3登録、NumPy変換、Python例外、Kots経由の関連モジュールimport | Python API用として残す。native buildと分離し、import軽量化は別測定で判断 | 維持/後続 |

表の主要な根拠：
[backend](../robokots/api/rust_backend.py)、[微分dispatch](../robokots/api/derivatives.py)、
[Rust微分adapter](../robokots/api/rust_derivatives.py)、[状態仕様](../robokots/core/state/spec.py)、
[Rust状態adapter](../robokots/outward/rust/data.py)、[Rust配列状態生成](../robokots/outward/rust/state.py)、
[PyO3操作](../robokots/_rust/src/py_api.rs)、[inward](../robokots/api/inward.py)、
[whole-body](../robokots/api/whole_body.py)、[export](../robokots/state_io/dictionary.py)。

## Python側に必要な最小モデル情報

Rust専用の主要経路では、SE3・慣性配列・関節選択行列をPythonへ複製する必要はない。
ただし、次の情報を一貫した順序で供給する必要がある。これは提案するデータ契約であり、実装済みの新APIではない。

| 情報 | 利用先 | 現在のnative側 |
| --- | --- | --- |
| link/jointの名前と内部番号の対応 | StateType、状態辞書、ターゲット | RobotModel・RustCompiledRobotに名前の保持なし。追加が必要 |
| dof、link数、joint数 | 入出力形状、workspace作成 | 内部に保持。Python getterあり、native外部向けgetterは未整備 |
| 各関節のdofとmotion開始位置、active joint順 | 座標・トルク・total_joint、motion配置 | q_index・関節順から現在のsubsetでは導出可能。明示的な公開が必要 |
| 親・子リンク番号 | 関節world状態、骨格表示 | parent_link/child_linkに保持、crate-private |
| 演算ごとの対応可否 | CMTM/RNEA/ABA等のdispatch | Python側の型・dof判定が残る。モデルを生成できることと全演算対応は別 |
| モデル識別・世代 | 状態、微分、inwardの再利用判定 | 共通のモデルrevision契約は確認できない。モデル交換時の無効化設計が必要 |

番号は明確に区別する。現行Pythonモデルはidでソートし、Rustは入力配列順の番号を使う。
固定関節が入ると関節番号とmotionの開始番号は一致しない。名前解決でPythonオブジェクトの同一性に頼る処理も置き換える。
柔軟リンク・多自由度関節まで拡張する際は、jointだけでなく一般のmotion owner順を再確認する。

最初は既存のcore/state仕様に渡せる小さな不変の情報集合でよい。完全なRobotStruct互換クラスや多層のtrait/protocolは先に作らない。
名前と接続を含むモデル情報をRustから一括取得すれば、各行のための細かな言語境界呼出しも避けられる。

## モデル構築をRustへ移す前に必要な検証

現行nativeの `from_model` は木構造・インデックス・関節対応等を検証するが、Pythonの公開モデル入力検証と同等ではない。
[Python検証](../robokots/core/robot.py) はschema version、名前重複、ID集合、有限値、軸等を確認する。
[型付きRustモデル](../robokots/_rust/src/model.rs) には名前・schema versionがなく、全数値の同じ検証も移されていない。

したがって、Pythonの検証を単純に削除してnative constructorをファイル読込の入口にすることはできない。
外部定義の検証とnative数値モデルの不変条件を分担し、未対応要素を誤って無視しないようにする。

URDFではさらに、親子のトポロジカル順序、root/worldリンク、rpy→quaternion、慣性座標の回転をPythonが処理している。
慣性指定がないURDFリンクはゼロ慣性を明示する一方、汎用モデルの既定値には単位慣性がある。この違いも保持する。
参照：[URDF読込](../robokots/urdf_io.py)。

## 状態取得と微分は別々に移す

`RustOutwardState.state_value` はworldの空間運動を `world_spatial_value` で計算し、関節についてはPythonモデルから子リンクを探す。
frame値は単体入力でmathroboのSE3として返す。CMTM/CMVector互換ビューにもPythonでの変換・演算が残る。

一方、selected微分カーネルはworld/姿勢混在をnative側で計算できる。
「微分がRustで計算される」ことは「状態値取得もPythonの数値処理なし」を意味しない。
値の配列取得をnativeで提供し、既存Python戻り値形式への変換を別に残すのが移行方針。
位置・姿勢の値と接空間ヤコビ行列の表現差、ordinary derivativeとCMTM係数の違いを崩さない。

また、`RustSelectedWorkspace.apply` はRustで書かれているがPython binding内にあり、Pythonなしビルドではその操作入口が存在しない。
ここは「Pythonを呼び戻している」問題ではなく、「nativeライブラリから同じ操作を呼べない」問題である。
主要カーネル内からPythonコールバックを呼ぶ箇所は今回の検索・読解では確認していない。

## 更新・キャッシュの扱い

- RustCompiledRobotはPython側モデルから作った数値情報を保持し、各状態容器もモデルのcloneを持つ。
- PythonのRustモデルキャッシュは、未生成なら生成する方式。outwardキャッシュは次数・バッチ形状とmotion revision・重力等を使う。
- selected workspaceはモデルオブジェクトと次数に紐づき、motion・重力等でprimalを再利用する。
- `set_order` ではRustモデル・outwardキャッシュ等も初期化する。モデル変更を検出する共通の世代番号や編集APIは、調査対象には見つからなかった。
- `InwardCache` は作成時のKotsとABA dataを保持し、q/v/gravity/tauを比較する。モデル交換通知の仕組みは確認できなかった。

Pythonの公開属性を書き換えた際にすべてのコピーが自動更新されるとは保証できない。これは今回実行して再現した不具合報告ではない。
独立化ではまずモデルを不変とし、変更時に新モデルへ交換して関連workspace・状態・メタデータを無効化する案を優先する。
motion revisionとモデルrevisionは別に扱う。参照共有への変更は、コピー量の実測と寿命設計をした後の別課題とする。

## 全APIを一度にRust専用へしない

NumPy/JAXの解析・AD、数値差分、whole-body等には実際のPythonモデルを使う計算が残る。
例として `center_of_mass()` はPythonで運動学状態を生成し、リンク質量・重心を集計する。
`forward_dynamics(backend='reference'/'numpy')` はPythonで質量行列を構成するが、内部のRNEAはRustを使う。backend名だけで依存を判断しない。

対照的に、状態辞書出力は主に名前・dof・状態providerを必要とし、骨格表示は主に親子接続と位置があればよい。
これらをすべて「完全なPythonモデルを生成しないと使えない」とする必要はない。
モデル定義の再出力・編集に必要な元情報の保持と、計算用構造のPython展開は別に設計する。

## 次の実装単位

1. **モデル情報の契約を確定**：名前・内部番号・motion配置・親子関係・対応機能。代表的なStateType解決とRust状態adapterをこの情報で動かす。
2. **native状態操作を公開**：outward compute/getter、selected apply、ABA操作。外部Rust利用側のintegration testで使用可能性を検証する。
3. **検証の分担と更新契約を確定**：外部入力の検証漏れを埋め、モデル交換時のキャッシュ無効化をまとめる。
4. **読み込みとKots生成を接続**：選んだ形式を直接Rustへ読み込み、Rust経路でRobotStruct構築・to_dictを呼ばないことを検査する。
5. **残りのPython機能を必要時に接続**：NumPy/JAX・編集等でのみ変換する。対応していない機能での暗黙の展開を避ける。

受け入れ試験には、名前・固定関節・出力順・joint/link混在・local/world・高次・バッチ・モデル交換・未対応入力を含める。
Rust専用経路ではRobotStruct生成と辞書化を禁止する検査を入れる。native単体利用とPythonの完全モデル省略は別々に検証する。
今回は新しい速度測定をしておらず、この棚卸しから高速化率は見積もらない。初期化時間・メモリ・言語境界変換・計算時間を次の変更単位で分けて測る。
