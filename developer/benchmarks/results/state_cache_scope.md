# Rust state-cache scope experiment

This is an isolated prototype; production dynamics and installed extension are unchanged.

Environment: macOS-15.7.4-arm64-arm-64bit-Mach-O, Python 3.13.1, NumPy 2.4.6. Seed 872, float64, RHS columns 2.
Warmup 5, interleaved randomized measurements 30; 4 JVP/VJP pairs after each dynamics calculation.
All times are medians in microseconds. Rust kernel timers exclude Python conversion/dispatch, model compilation, and initial persistent-buffer allocation; derivative buffer allocation is included. First-call timings and raw samples are in the JSON.

Policies: current = existing lean/full dynamics followed by independent derivative recomputation; eager = full state computed unconditionally; lazy_fill = keep existing dynamics and fill omitted fields on first derivative only; lazy_recompute = keep dynamics but recompute full state once on first derivative. Prepared derivatives skip repeated dynamics and kinematics evaluations.

| DOF | Batch | Order | Gravity | Dynamics current | Dynamics eager | Dynamics lazy | Promotion fill | Promotion recompute | Total current | Total eager | Total lazy |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
|16|1|3|zero|1.38|2.08|1.38|0.71|1.96|360.96|334.46|329.58|
|16|1|3|nonzero|3.50|3.38|3.50|0.02|0.00|482.77|427.50|428.08|
|16|1|5|zero|8.58|8.29|8.29|0.00|0.00|730.23|585.96|581.15|
|16|1|5|nonzero|9.85|9.75|9.75|0.04|0.00|914.90|724.00|736.31|
|16|8|3|zero|12.62|17.52|11.94|5.65|16.56|2901.08|2650.83|2637.23|
|16|8|3|nonzero|29.67|29.08|28.94|0.04|0.00|3787.40|3347.31|3511.21|
|16|8|5|zero|70.98|69.73|70.85|0.00|0.04|5772.21|4670.15|4661.08|
|16|8|5|nonzero|83.75|80.15|81.83|0.04|0.04|7052.08|5776.62|5806.42|
|64|1|3|zero|5.38|8.58|5.17|2.58|8.12|1352.14|1222.75|1228.90|
|64|1|3|nonzero|13.67|13.33|13.31|0.00|0.00|1874.17|1689.69|1658.48|
|64|1|5|zero|32.75|32.56|32.52|0.04|0.00|2882.25|2240.35|2197.37|
|64|1|5|nonzero|39.96|37.94|39.67|0.00|0.04|3533.56|2881.79|2846.04|
|64|8|3|zero|44.21|67.23|42.58|18.73|67.10|10817.50|9788.21|9912.75|
|64|8|3|nonzero|112.04|110.92|113.71|0.04|0.04|14675.54|13312.92|13079.92|
|64|8|5|zero|271.02|266.56|271.10|0.04|0.04|21892.04|17471.27|17445.46|
|64|8|5|nonzero|315.94|310.75|313.38|0.04|0.00|27801.58|22727.33|22641.48|

## Memory

Numeric Vec capacities only (KiB): excludes model copies, Vec/struct headers, allocator overhead, Python views and output arrays. Semantic state is a subset of dynamics buffers, not an additional copy. Tangent buffer is one sample with two RHS columns; it can be reused across batch samples. Removing scratch/separate kinematics is a byte-count estimate, not a measured alternative implementation.

|DOF|Batch|Order|Dynamics buffers|Separate kinematics|Semantic subset|One tangent buffer|
|---:|---:|---:|---:|---:|---:|---:|
|16|1|3|20.09|12.74|14.33|27.02|
|16|1|5|35.00|17.23|27.14|50.42|
|16|8|3|160.75|101.94|114.62|27.02|
|16|8|5|280.00|137.81|217.12|50.42|
|64|1|3|68.47|42.74|53.70|101.27|
|64|1|5|118.62|56.23|101.77|189.17|
|64|8|3|547.75|341.94|429.62|101.27|
|64|8|5|949.00|449.81|814.12|189.17|

## Numerical agreement and limits

Against the existing selected Rust kernels: maximum absolute difference 0, maximum relative Frobenius error 0. Also checked distinct motion and all-zero motion for each case. This is a reference implementation, not an exact solution.
The experiment isolates cache reuse; it does not measure JAX or numerical differentiation speed. Production regression tests independently compare selected derivatives with NumPy, central differences and JAX. No production cache invalidation or promotion policy has been changed by this experiment.

## 判断と保持範囲

常時全状態を計算する eager 案は採用しない。order 3・重力ゼロでは現行 dynamics が省略している計算を復活させ、今回の測定では dynamics 単体が約1.39～1.60倍になった。
推奨は lazy_fill。通常の dynamics は同じ計算経路を維持し、微分が最初に要求されたときだけ、既存のリンク運動・関節運動量から不足する関節情報、リンク運動量・力、関節力を補う。order 5や非ゼロ重力では元々計算済みの状態をそのまま利用する。
補完先の数値領域は現在の DynamicsCmtmWorkspace に既に確保されており、今回の試作では状態保存用の追加配列を作っていない。保持するのは motion/重力/次数に依存する状態。JVP/VJPの右辺に依存する配列は計算用バッファとして分離し、密ヤコビ行列・world変換後の全系列を無条件に状態キャッシュへ追加しない。
同一条件の新旧状態を重複保持せず、motion更新時は既存領域を更新する。モデル・motion revision・次数・重力・batch shape を有効性条件とし、異なる形状を無制限には蓄積しない。最新の利用中状態と必要な作業バッファを優先する。キャッシュ上限や形状変更時の割り当て性能は今回の測定対象外。
64自由度・batch 8・order 5では、動力学バッファ949 KiBに対し、別の運動学バッファが449.81 KiB。後者は遅延確保・共用化の候補だが、今回は削除した場合の速度を測定していないため、削減効果の予測と実測を区別する。状態とscratchの分離によるコピーを増やす変更も採用前に測定する。

### dynamics + 最初のJVP/VJP各1回

以下も同じ測定の先頭4フェーズの和の中央値。単位はµs。

|DOF|Batch|Order|Gravity|Current|Lazy fill|
|---:|---:|---:|:---:|---:|---:|
|16|1|3|zero|94.92|83.31|
|16|1|3|nonzero|128.83|110.88|
|16|1|5|zero|196.67|148.21|
|16|1|5|nonzero|235.08|188.40|
|16|8|3|zero|775.48|671.75|
|16|8|3|nonzero|953.54|876.02|
|16|8|5|zero|1509.38|1206.17|
|16|8|5|nonzero|1814.40|1512.87|
|64|1|3|zero|351.77|310.85|
|64|1|3|nonzero|472.19|422.23|
|64|1|5|zero|733.06|580.73|
|64|1|5|nonzero|926.96|738.13|
|64|8|3|zero|2784.54|2472.90|
|64|8|3|nonzero|3822.56|3385.04|
|64|8|5|zero|5658.04|4595.25|
|64|8|5|nonzero|7208.08|5932.33|

本体への適用は未実施。この結果はキャッシュ範囲の判断用の試作測定であり、公開API全体の速度向上を保証する値ではない。
