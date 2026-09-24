# モデル化誤差の生成

`apply_perturbation` は、公称モデルからパラメータの異なる新しい `Kots`
を生成します。乱数は生成時に一度だけサンプルされ、軌道の評価中は固定です。
観測ノイズや毎時刻の外乱とは別の機能です。

```python
from robokots.kots import Kots
from robokots import PerturbationSpec, apply_perturbation

nominal = Kots.from_urdf_file("robot.urdf", backend="numpy")
perturbed, report = apply_perturbation(
    nominal,
    PerturbationSpec(
        seed=42,
        mass_relative_std=0.05,
        link_length_relative_std=0.01,
    ),
    return_report=True,
)
```

`*_relative_std` は対数空間の標準偏差です。倍率は `exp(N(0, std²))` なので
中央値が1、平均値は `exp(std²/2)` です。厳密な「相対標準偏差」や平均1では
ありません。質量0のリンクは簡易指定では質量0のままです。

## TOMLファイルからの指定

Pythonでのコンストラクタ指定に加えて、設定をTOMLに保存できます。
追加依存は不要です（Python 3.11以降の標準 `tomllib` を使用）。

```toml
# perturbation.toml
seed = 42
mass_relative_std = 0.05
link_length_relative_std = 0.01
scale_inertia_with_mass = true
```

```python
spec = PerturbationSpec.from_toml_file("perturbation.toml")
perturbed, report = apply_perturbation(nominal, spec, return_report=True)
```

詳細指定は `[[rules]]` と `[rules.noise]` で表現します。
次は上の簡易設定とは独立した、別の設定ファイルの例です。

```toml
seed = 42

[[rules]]
parameter = "mass"
[rules.noise]
distribution = "uniform"
mode = "scale"
low = 0.8
high = 1.2

[[rules]]
parameter = "link_length"
names = ["left_thigh", "right_thigh"]
groups = [["left_thigh", "right_thigh"]]
[rules.noise]
distribution = "normal"
mode = "additive"
std = 0.0003  # m
```

`names` は実モデルに合わせて置き換えてください。`names` と `groups` は
`[rules.noise]` **より前**に記述します。後に置くとnoise内の未知キーとして
エラーになります。単純指定と明示ルールを両方書くと累積適用されます。

設定はルート直下に記述します。未知キーは無視せず例外にし、
`spec.rules[0].noise` などの位置を表示します。省略フィールドにはPython APIと
同じ既定値を使います。`seed` や `names` の `None` 相当はキーを省略します。
空の `names = []` は対象なしです。

TOML文字列は `PerturbationSpec.from_toml(text)` で読み込めます。
既存のTOMLに `[perturbation]` セクションとして格納する場合は、
`PerturbationSpec.from_dict(tomllib.loads(text)["perturbation"])` を使います。
ファイル読込エラーとTOML構文エラーは、そのまま呼び出し側に通知します。

設定例と比較スクリプトは [examples/perturbation_example](../examples/perturbation_example/README.md) にあります。

## Pythonによる詳細指定

```python
from robokots import NoiseSpec, ParameterPerturbation, PerturbationSpec

spec = PerturbationSpec(seed=42, rules=(
    ParameterPerturbation(
        parameter="mass",
        noise=NoiseSpec(distribution="uniform", mode="scale", low=0.8, high=1.2),
    ),
    ParameterPerturbation(
        parameter="link_length",
        names=("left_thigh", "right_thigh"),  # 実際のモデルの名前を指定
        groups=(("left_thigh", "right_thigh"),),
        noise=NoiseSpec(distribution="normal", mode="additive", std=0.0003),
    ),
    ParameterPerturbation(
        parameter="joint_rotation",
        names=("hip_joint",),
        noise=NoiseSpec(std=0.001),
    ),
))
```

`names` は、リンクパラメータではリンク名、関節パラメータでは関節名です。
`groups` の各組は同一サンプルを共有し、その他は独立にサンプルします。
これは完全な共分散行列の指定ではありません。

| parameter | 操作と単位 | mode |
|---|---|---|
| `mass` | リンク質量。加算はkg | `additive` / `scale` |
| `link_length` | 剛体の出側関節までの距離、または柔軟リンクの長さ。加算はm | `additive` / `scale` |
| `cog_translation` | リンク座標系での重心位置への3軸独立加算、m | `additive` |
| `joint_translation` | 親リンク座標系での関節原点への3軸独立加算、m | `additive` |
| `joint_rotation` | 関節原点姿勢への右側からの回転合成。回転ベクトル各成分、rad | `additive` |
| `joint_offset` | 回転関節はrad、直動関節はmの零点偏差 | `additive` |
| `inertia` | 重心まわりの質量二次モーメントの固有値を3方向独立に変更 | `scale` |

`joint_offset` は `T_new(q) = T_nominal(q + offset)` の符号規約です。
回転・直動の1自由度関節に対応し、固定・球面・浮遊関節を明示指定するとエラーです。
関節原点に偏差を組み込むため、計算バックエンドに零点補正の処理は不要です。

| distribution | 指定 | サンプル |
|---|---|---|
| `normal` | `std`、任意の `mean` | 正規分布。既定meanは加算0、倍率1 |
| `uniform` | `low`, `high` | 指定区間の一様分布 |
| `lognormal` | `std`、任意の `mean` | 指定された対数平均・標準偏差から指数化。既定meanは0 |
| `loguniform` | `low`, `high` | 指定した正の倍率区間で対数一様分布 |

対数分布は `scale` 専用です。範囲外・非有限値・非正の質量や長さを生成した
場合は例外になります。クリップや再抽選で分布を変更しません。

`names=None` は適用可能な対象を選びます。world、およびworldを直接親とする
関節は明示ルールの既定対象から除外します。質量・慣性は正の質量を持つリンク、
長さは非ゼロの距離を持つリンク、零点は回転・直動関節が対象です。
明示的な名前の誤りや、適用できない長さ・零点指定は例外になります。
worldへの固定取付を変える場合は関節名を明示します。

簡易指定は先に、`rules` は記述順に実行します。同じパラメータへの複数指定は
累積します。`link_names` は簡易指定だけに適用されます。
同じモデル・設定・seed・実装環境で再現でき、NumPyのグローバル乱数には影響しません。
ルール順や対象順が変わると乱数の割当も変わります。

## 剛体のリンク長の意味

URDFには剛体の単一の「リンク長」はないため、リンク `L` の長さ摂動は、
**Lを親とする関節の `origin.position`** に適用します。親リンク座標系で
ベクトルの方向を保ち、倍率指定なら `p' = s p`、加算指定なら
`p' = (||p|| + delta) p / ||p||` とします。

分岐では同じリンクのすべての非ゼロ接続距離に同じ倍率／長さ増分を適用します。
ゼロ距離は保ちます。終端リンクなど出側距離がない対象を明示した場合は例外に
なります。任意の取付誤差や個別の枝を変える場合は `joint_translation` を使います。
柔軟リンクではカーネルで使う `length` を変更します。

この操作は運動学的な寸法誤差です。重心、質量、慣性、メッシュの寸法を連動して
変形する操作ではありません。CAD形状全体の伸縮には密度・形状の前提が必要です。
旧APIの `link_translation_std` は従来通り、選択リンクに**入る**関節原点の
3軸並進誤差であり、長さ摂動とは対象と意味が異なります。

## 慣性の物理的整合性

質量変更は既定で重心まわりの慣性テンソルも同じ倍率にします。
`scale_inertia_with_mass=False` は質量だけを変更します。

独立の `inertia` ルールでは、重心まわりの慣性テンソル `I` から
`S = trace(I)/2 * E - I` を作り、その非負固有値を正の倍率で変更して、
`I' = trace(S') * E - S'` に戻します。これにより慣性の半正定値性と
主慣性モーメントの三角不等式を維持します。元の不整合な慣性は例外にします
（浮動小数点丸めの微小な負固有値のみ0に補正）。
形状の境界内に質量分布が収まることまでは検証しません。

## 出力と再現

元モデルのmotion・targets・計算済み状態はコピーしません。order・dim・lib・
入力backendを引き継いだ新しいKotsを返すので、比較するmotionを両モデルへ
明示的にインポートしてください。Rustは変更済みモデルから再コンパイルします。

`report` の従来フィールドは簡易指定の変更を記録します。明示ルールの実際の
サンプルは `rule_changes`、累積適用後のモデルは `model_data` に記録します。

```python
import json

record = json.dumps(report.to_dict())
replayed = Kots.from_json_data(json.loads(record)["model_data"], backend="numpy")
```

再入力時のorder等は必要に応じて指定してください。
レポート内の辞書は出力モデルから独立したスナップショットです。

## 文献と対応範囲

- [Wang et al., 2019, Table 2](https://journals.sagepub.com/doi/pdf/10.1177/1687814018816894?download=true)：リンク寸法の正規分布。
- [Exarchos et al., ICRA 2021, §V-A](https://arxiv.org/html/2011.01891v1)：リンク長倍率の一様分布、左右脚の共有。
- [Peng et al., ICRA 2018, Table I](https://xbpeng.github.io/projects/SimToReal/SimToReal_2018.pdf)：質量などの対数空間サンプリング。
- [Tan et al., RSS 2018, Table I](https://roboticsproceedings.org/rss14/p10.pdf)：質量・慣性などの一様分布。
- [Chebotar et al., SimOpt, 2019](https://arxiv.org/html/1810.05687v4)：相関を含む分布。現APIはグループ共有のみで、共分散学習は未対応。
- [Wensing et al., 2017](https://arxiv.org/pdf/1701.04395)：質量二次モーメントを通じた慣性の物理的整合性。

上記を参考にしたAPIであり、論文の実験を完全再現するプリセットではありません。
摩擦・減衰・駆動ゲイン・遅延・センサノイズ、独立した関節軸の摂動は未対応です。
関節原点の回転誤差は軸の取付姿勢にも影響します。
