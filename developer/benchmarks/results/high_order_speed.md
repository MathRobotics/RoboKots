# High-order dynamics speed experiment

This report records the pre-integration experiment. The combined optimization is now implemented in production; see [the production API comparison](high_order_production.md) for measurements after integration. Timings below remain the original prototype measurements.

Environment: macOS-15.7.4-arm64-arm-64bit-Mach-O; Python 3.13.1, NumPy 2.4.6, rustc 1.91.1 (ed61e7d7e 2025-11-07). float64, seed 917.
Warmup 5, randomized interleaved samples 30, 5 evaluations per timing. Median µs per batch. First-call kernel timings (one evaluation), quartiles and raw timings in JSON.

Isolated prototype extension; production sources and installed extension unchanged. Timing excludes Python conversion, allocation of the persistent workspace, model construction and output extraction. Includes a temporary 3x3 rotation-series buffer in the spatial variant. The same complete state is computed in every dynamics variant; no outputs or derivative orders are omitted. Kinematics is timed separately, not subtracted from dynamics as if it were an exact profile.

Variants: current = CMTM matrix-series composition/recovery; spatial = propagate ordinary body-velocity derivatives using the relative inverse-rotation series (fixed/revolute only); transport = reuse momentum transport blocks for gravity instead of rebuilding the same prefix; both = combine these changes.

|DOF|Batch|Order|Gravity|Current dynamics|Spatial|Transport|Both|Speedup|Current kinematics|Spatial kinematics|
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
|16|1|4|zero|5.59|3.55|5.65|3.67|1.52x|3.20|1.12|
|16|1|4|nonzero|6.47|4.38|6.42|4.21|1.54x|3.15|1.12|
|16|1|5|zero|8.29|5.15|8.34|5.16|1.61x|4.47|1.53|
|16|1|5|nonzero|9.84|6.66|9.35|6.32|1.56x|4.45|1.52|
|16|1|6|zero|11.28|7.11|11.32|7.21|1.56x|5.94|1.99|
|16|1|6|nonzero|13.46|9.55|12.75|8.88|1.52x|6.20|2.03|
|16|1|8|zero|18.69|12.13|18.72|12.10|1.54x|9.80|3.14|
|16|1|8|nonzero|23.30|16.54|21.68|14.88|1.57x|9.92|3.15|
|16|8|4|zero|46.12|28.90|46.39|29.02|1.59x|25.87|8.91|
|16|8|4|nonzero|52.85|35.62|51.29|34.15|1.55x|25.91|9.10|
|16|8|5|zero|65.93|41.51|65.66|41.42|1.59x|36.70|12.27|
|16|8|5|nonzero|77.37|53.43|75.46|49.76|1.55x|36.48|12.10|
|16|8|6|zero|90.28|57.29|89.92|57.26|1.58x|48.61|15.62|
|16|8|6|nonzero|107.47|74.69|102.47|68.89|1.56x|48.05|15.56|
|16|8|8|zero|152.38|97.21|151.46|96.15|1.58x|80.41|25.18|
|16|8|8|nonzero|186.57|132.56|173.99|118.69|1.57x|79.18|25.23|
|64|1|4|zero|22.72|14.57|22.50|14.45|1.57x|12.34|4.31|
|64|1|4|nonzero|25.77|17.60|24.97|16.76|1.54x|12.33|4.30|
|64|1|5|zero|32.49|20.75|32.43|20.66|1.57x|17.40|5.83|
|64|1|5|nonzero|38.07|26.60|36.54|24.96|1.53x|17.47|5.83|
|64|1|6|zero|44.23|28.73|44.45|28.72|1.54x|23.25|7.56|
|64|1|6|nonzero|53.08|37.72|50.40|34.49|1.54x|23.40|7.71|
|64|1|8|zero|74.20|48.63|73.95|48.57|1.53x|37.80|12.65|
|64|1|8|nonzero|90.38|65.24|84.11|58.82|1.54x|38.02|12.47|
|64|8|4|zero|180.97|115.34|181.38|115.62|1.57x|99.08|34.55|
|64|8|4|nonzero|207.39|141.57|201.20|134.68|1.54x|98.87|34.72|
|64|8|5|zero|262.92|166.55|261.88|167.40|1.57x|140.64|46.08|
|64|8|5|nonzero|308.66|214.35|294.26|199.87|1.54x|140.73|46.67|
|64|8|6|zero|357.94|230.82|358.20|231.60|1.55x|186.39|60.29|
|64|8|6|nonzero|427.69|300.30|405.50|277.97|1.54x|186.27|60.03|
|64|8|8|zero|595.82|392.34|598.60|392.30|1.52x|307.58|99.14|
|64|8|8|nonzero|732.59|524.40|682.64|478.20|1.53x|305.36|99.50|

## Correctness and limits

Against existing Rust state: max absolute difference 6.40284e-10, max relative Frobenius error 1.10512e-14. Compared all semantic buffers including poses, velocity series, link/joint momentum and force, torque and gravity intermediates. Each batch sample, zero motion and changed motion were checked independently.
Independent NumPy comparison on oblique branched/fixed-joint model, orders 4/5/6/8, random and zero motion: max absolute difference 1.42109e-14, max relative Frobenius error 2.08275e-16 for all link/joint momentum/force series.
The new recurrence is analytically equivalent for fixed/revolute joints, and retains factorial scaling at series boundaries. It does not extend the current Rust model support to prismatic/spherical/floating joints. The prototype does not replace JVP/VJP kernels and their runtime was not measured here. Public Python API speedups can be smaller because conversion/dispatch cost is excluded. No derivative-result caching, dense Jacobian caching, parallel execution or additional per-link persistent cache is introduced.

## 検討結果と推奨順序

第一候補は order 4 以降の運動学計算の置換。現行はCMTMの4×4行列系列を合成し、逆変換して速度系列へ戻す。試作は相対回転の逆行列系列を使い、親の空間速度系列から子の系列を直接計算する。次数に対する計算量のオーダーは同じでも、行列サイズと中間演算が減る。
相対並進 p が関節座標に依存しない固定・回転関節に対して、ω_child = R_rel^T ω_parent + axis qdot、v_child = R_rel^T (v_parent − p × ω_parent) を高次の積の微分で評価する。回転系列は階乗正規化し、公開の速度系列では通常の時間微分へ戻す。
第二候補は重力ありの場合の変換系列の共有。子の関節運動量の変換で構築したブロックの先頭部分を、同じ関節の重力系列の変換にも使う。新しい永続キャッシュではなく、一回の計算内の既存scratchの再利用。重力ゼロの場合にはこの重複自体がない。
測定した組み合わせ案は約1.52～1.61倍の速度（約34～38%の時間短縮）。order 3 の専用高速経路には手を入れない。追加scratchは1サンプルの関節巡回で共用する3×3行列の系列のみで、order 4で216 bytes、order 8で504 bytes。リンク数やバッチ数に比例する永続状態を増やしていない。
本体へ適用する場合は、最初に運動学の置換、次に重力変換共有を独立変更として検証する。固定関節・ゼロ姿勢・斜め軸・分岐・各次数・重力・バッチの状態値に加え、既存JVP/VJPカーネルとの整合性を確認する。今回の速度測定は状態計算のみで、微分APIの高速化倍率は未測定。
前回の必要時補完キャッシュ案とは併用可能だが、削減する仕事に重なりがあるため、速度改善倍率を単純に掛け合わせることはできない。公開APIに組み込んだ状態で dynamics 単体と dynamics＋微分を再測定して判断する。
今回は検討用の隔離拡張であり、本体および通常利用するインストール済み拡張への適用はしていない。
