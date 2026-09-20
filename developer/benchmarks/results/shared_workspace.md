# Shared kinematics and lazy dynamics workspace

Environment: macOS-15.7.4-arm64-arm-64bit-Mach-O, Python 3.13.1, NumPy 2.4.6, rustc 1.91.1 (ed61e7d7e 2025-11-07). float64, seed 921.
3 rounds in fresh processes; each process loads both binaries and randomly interleaves baseline/optimized per phase and sample. Per case: 5 warmups, 30 samples per round, 5 calls per hot sample. Medians in microseconds per batch. Creation and first dynamics are one call per sample. Full samples, first hot-phase calls and hashes are in JSON.

create: workspace allocation including model cloning; first dynamics: after kinematics on a fresh workspace, including lazy allocation. Hot raw calls include Python binding cost. Pair is kinematics + dynamics on the same workspace. Public dynamics includes import_motions and cache invalidation each time. Model compilation and output extraction are excluded. No derivative timings are claimed.

|DOF|Batch|Order|Gravity|Create before/after|First dynamics before/after|Hot kinematics before/after|Hot dynamics before/after|Pair before/after|Public dynamics before/after|
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
|16|1|3|zero|6.29 / 4.83|2.96 / 3.65|2.27 / 2.27|2.73 / 2.73|4.86 / 4.86|17.57 / 17.78|
|16|1|3|nonzero|6.67 / 5.19|6.79 / 7.42|2.29 / 2.27|6.36 / 6.34|8.57 / 8.53|23.89 / 24.04|
|16|1|5|zero|6.85 / 5.33|10.71 / 11.38|3.29 / 3.16|10.10 / 10.09|13.27 / 13.10|25.90 / 26.12|
|16|1|5|nonzero|7.00 / 5.08|12.67 / 13.46|3.24 / 3.17|11.96 / 11.96|15.18 / 14.97|30.16 / 30.11|
|16|1|8|zero|8.75 / 5.65|25.35 / 25.35|6.24 / 6.22|23.22 / 23.39|29.73 / 29.26|40.06 / 40.10|
|16|1|8|nonzero|9.10 / 6.27|29.50 / 30.92|6.23 / 6.22|28.54 / 28.31|34.73 / 34.20|47.75 / 47.72|
|16|8|3|zero|26.27 / 12.73|19.54 / 25.25|16.77 / 16.59|19.08 / 19.00|36.26 / 35.51|36.14 / 36.84|
|16|8|3|nonzero|25.85 / 12.12|50.85 / 55.38|16.68 / 16.46|49.09 / 48.69|66.34 / 65.20|69.78 / 68.79|
|16|8|5|zero|33.48 / 16.27|81.81 / 88.96|24.86 / 24.39|81.93 / 82.08|107.42 / 106.99|102.28 / 101.59|
|16|8|5|nonzero|38.17 / 18.75|97.79 / 105.23|25.30 / 24.21|98.24 / 96.83|123.66 / 121.00|122.68 / 120.21|
|16|8|8|zero|50.88 / 22.38|189.35 / 207.25|50.20 / 49.75|189.52 / 189.63|242.78 / 239.53|215.25 / 215.14|
|16|8|8|nonzero|70.12 / 25.25|233.52 / 254.50|52.88 / 51.32|233.68 / 233.10|287.16 / 282.26|263.15 / 264.67|
|64|1|3|zero|20.92 / 17.85|10.08 / 11.52|7.97 / 7.93|9.26 / 9.29|17.79 / 17.27|26.05 / 26.13|
|64|1|3|nonzero|21.56 / 17.48|25.00 / 26.27|8.02 / 7.91|23.49 / 23.81|32.64 / 31.33|43.61 / 43.85|
|64|1|5|zero|25.73 / 21.50|39.75 / 42.15|11.35 / 11.32|39.44 / 39.29|52.40 / 50.49|58.46 / 58.37|
|64|1|5|nonzero|24.10 / 18.62|46.35 / 49.21|11.36 / 11.25|46.54 / 48.09|58.45 / 59.21|65.97 / 66.06|
|64|1|8|zero|31.21 / 23.79|90.62 / 95.81|24.25 / 24.33|91.01 / 91.04|114.65 / 114.96|107.65 / 109.90|
|64|1|8|nonzero|47.04 / 31.67|110.52 / 116.33|23.77 / 23.67|111.28 / 110.78|135.09 / 134.76|133.22 / 134.07|
|64|8|3|zero|80.15 / 45.06|73.83 / 88.44|68.80 / 67.20|72.90 / 72.43|140.07 / 138.70|95.88 / 99.40|
|64|8|3|nonzero|90.83 / 64.21|202.79 / 218.06|71.13 / 68.61|201.39 / 198.52|269.90 / 264.88|247.28 / 232.69|
|64|8|5|zero|148.42 / 54.08|319.71 / 347.42|97.45 / 95.23|324.40 / 316.37|416.57 / 411.20|362.88 / 354.00|
|64|8|5|nonzero|141.02 / 72.96|384.31 / 422.42|100.45 / 96.25|383.16 / 380.85|483.02 / 473.33|434.01 / 430.00|
|64|8|8|zero|180.19 / 81.42|734.31 / 776.04|194.61 / 192.51|726.71 / 731.62|926.74 / 915.86|773.66 / 770.23|
|64|8|8|nonzero|227.42 / 71.42|896.02 / 941.71|195.35 / 192.13|883.71 / 880.05|1086.04 / 1073.43|924.17 / 933.19|

All local link/joint momentum and force derivative outputs: max absolute difference 0, max relative Frobenius error 0.

## Numeric buffer capacity

Diagnostic values count Vec capacity times element size; exclude object/allocator overhead, robot storage, Python outputs and temporary per-call allocations. Baseline binary has no diagnostic; its capacities below are from the previous recorded cache-scope experiment, not a new RSS measurement.

64 DOF, 66 links, 65 joints, batch 8, order 5:

|Stage|Baseline numeric buffers KiB|Optimized measured KiB|
|---|---:|---:|
|created|1398.81|449.81|
|kinematics|1398.81|449.81|
|dynamics|1398.81|949.00|
|kinematics_after_dynamics|1398.81|949.00|

The shared CMTM allocation is retained through all transitions; dynamics-only arrays have zero capacity before first dynamics and remain reusable afterwards. Batch-shape cache limits and reduction of unused CMTM scratch arrays are outside this change.
