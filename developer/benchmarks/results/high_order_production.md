# High-order production dynamics comparison

Environment: macOS-15.7.4-arm64-arm-64bit-Mach-O; Python 3.13.1, NumPy 2.4.6, rustc 1.91.1 (ed61e7d7e 2025-11-07). Seed 917, float64.
Fresh processes, alternating binary order across 3 rounds; 5 warmups, 30 samples per round, 5 evaluations per sample. Median microseconds per batch. First-call times, raw samples, binary and source hashes are in JSON.

Raw = persistent Rust batch workspace compute_dynamics, including Python binding cost. Public = import_motions + dynamics(backend="rust"), including validation, conversion, cache invalidation and state update. Every timed call recomputes dynamics; no cache-hit timings. Model/workspace construction and state extraction are outside timing. This measures state computation, not Jacobian/JVP/VJP evaluation.

|DOF|Batch|Order|Gravity|Raw before|Raw after|Public before|Public after|Public speedup|
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
|16|1|3|zero|2.68|2.66|16.93|16.89|1.00x|
|16|1|3|nonzero|6.43|6.25|23.33|23.18|1.01x|
|16|1|4|zero|11.06|7.15|25.98|21.93|1.18x|
|16|1|4|nonzero|12.60|8.35|29.38|25.33|1.16x|
|16|1|5|zero|15.67|9.94|30.86|24.74|1.25x|
|16|1|5|nonzero|18.35|11.78|35.89|29.03|1.24x|
|16|1|6|zero|21.47|13.65|36.87|28.98|1.27x|
|16|1|6|nonzero|25.77|16.31|43.41|34.04|1.28x|
|16|1|8|zero|35.83|22.98|52.62|39.31|1.34x|
|16|1|8|nonzero|43.51|27.92|62.92|47.13|1.34x|
|16|8|3|zero|18.77|18.78|34.85|34.62|1.01x|
|16|8|3|nonzero|50.39|48.64|69.59|67.83|1.03x|
|16|8|4|zero|90.44|55.61|110.55|72.26|1.53x|
|16|8|4|nonzero|102.04|65.19|122.35|84.81|1.44x|
|16|8|5|zero|128.91|81.05|146.50|98.60|1.49x|
|16|8|5|nonzero|152.10|96.50|172.60|116.08|1.49x|
|16|8|6|zero|176.57|112.48|195.63|129.87|1.51x|
|16|8|6|nonzero|210.40|136.20|233.41|157.79|1.48x|
|16|8|8|zero|295.50|189.17|318.07|209.64|1.52x|
|16|8|8|nonzero|366.36|229.22|399.79|252.52|1.58x|
|64|1|3|zero|9.20|9.17|24.86|24.69|1.01x|
|64|1|3|nonzero|24.00|23.11|43.68|42.87|1.02x|
|64|1|4|zero|43.14|27.23|60.38|43.84|1.38x|
|64|1|4|nonzero|48.92|31.67|67.82|50.48|1.34x|
|64|1|5|zero|61.60|38.93|79.06|56.22|1.41x|
|64|1|5|nonzero|72.19|46.63|92.34|65.66|1.41x|
|64|1|6|zero|84.19|53.86|100.23|71.84|1.40x|
|64|1|6|nonzero|99.69|64.43|118.68|86.24|1.38x|
|64|1|8|zero|138.83|90.91|155.71|107.61|1.45x|
|64|1|8|nonzero|171.48|110.66|191.75|129.76|1.48x|
|64|8|3|zero|70.73|71.82|88.71|88.55|1.00x|
|64|8|3|nonzero|195.42|193.58|216.27|213.63|1.01x|
|64|8|4|zero|347.98|222.64|376.46|246.88|1.52x|
|64|8|4|nonzero|401.27|256.29|433.40|284.80|1.52x|
|64|8|5|zero|503.96|317.81|551.36|357.46|1.54x|
|64|8|5|nonzero|579.59|378.04|610.33|415.44|1.47x|
|64|8|6|zero|670.25|433.45|714.66|469.20|1.52x|
|64|8|6|nonzero|807.90|530.25|864.30|574.43|1.50x|
|64|8|8|zero|1131.25|729.95|1179.76|783.41|1.51x|
|64|8|8|nonzero|1389.14|894.20|1441.82|952.29|1.51x|

All local link/joint momentum and force derivatives agree with the previous binary: maximum absolute difference 2.00816e-09; maximum relative Frobenius error 1.61516e-14.

Production changes: order >=4 fixed/revolute kinematics directly propagates spatial velocity derivatives; gravity transport reuses blocks already built for momentum. Order-3 zero-gravity specialized dispatch is preserved. No persistent state fields are added. Temporary rotation scratch is 72*(order-1) bytes per evaluation, reused across joints (216 bytes at order 4, 504 bytes at order 8). Results are environment-dependent.
