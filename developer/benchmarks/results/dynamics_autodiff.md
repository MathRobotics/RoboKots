# Dynamics Jacobian timing and accuracy

Measured: 2026-09-13T08:03:25.521299+00:00

Environment: macOS-15.7.4-arm64-arm-64bit-Mach-O; arm; Python 3.13.1; JAX 0.10.1; NumPy 2.4.6; [CpuDevice(id=0)]; float64.

Dense Jacobians, single sample, gravity [0.3, -0.4, -9.81]. Momentum/force select the last link; torque selects all active joints. Generated serial models use alternating x/y/z joint axes.

## Median execution time (ms)

Full analytic timings include motion import, dynamics state computation (without dictionary materialization), and Jacobian computation. Cached timings exclude state computation. JAX JIT timings include input conversion, synchronized execution and NumPy output conversion. JIT compilation is reported separately; a compiled function must be reused. The public jacobian_autodiff() is the eager column, not the JIT column.

NumPy/Rust labels indicate the requested dynamics-state backend. With the nonzero gravity used here, link momentum/force dense Jacobians fall back to the Python analytic path even after Rust state computation. The torque cases use Rust RNEA/CMTM derivative kernels.

| Model | Output | Shape | NumPy full | Rust full | NumPy cached | Rust cached | Numerical | JAX eager | JAX JIT warm | JIT first (ms) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sample4 | momentum | 6×8 | 2.2328 | 0.9175 | 0.1841 | 0.1306 | 17.1211 | 46.8465 | 0.0674 | 200.6072 |
| sample4 | force | 6×12 | 11.4579 | 6.7624 | 3.4890 | 2.0461 | 75.6236 | 198.2693 | 0.1232 | 550.2782 |
| sample4 | torque | 4×12 | 16.4431 | 0.0393 | 5.8439 | 0.0205 | 302.9640 | 194.6675 | 0.1243 | 737.0389 |
| sample4 | torque_diff1 | 4×16 | 25.5196 | 0.4412 | 7.6005 | 0.4244 | 594.3375 | 535.2135 | 0.2345 | 1465.3604 |
| sample4 | torque_diff2 | 4×20 | 33.0470 | 0.6395 | 9.3400 | 0.6222 | 977.4048 | 903.1801 | 0.2937 | 2412.0022 |
| branched3 | torque | 3×9 | 14.3883 | 0.0348 | 5.4494 | 0.0168 | 162.5199 | 196.1259 | 0.1192 | 582.5026 |
| branched3 | torque_diff2 | 3×15 | 29.0229 | 0.4765 | 9.1602 | 0.4517 | 539.5300 | 885.9669 | 0.2677 | 2258.9398 |
| serial8 | torque | 8×24 | 36.3201 | 0.0612 | 11.0923 | 0.0429 | 2153.4311 | 359.3650 | 0.2251 | 1140.7419 |
| serial8 | torque_diff2 | 8×40 | 76.4309 | 2.0883 | 18.8971 | 2.4721 | 7153.0939 | 1590.2544 | 0.6627 | 4909.5128 |
| serial16 | torque | 16×48 | 94.7709 | 0.1200 | 25.7982 | 0.0992 | 16654.8050 | 664.9386 | 0.4490 | 2336.8368 |
| serial16 | torque_diff2 | 16×80 | 210.5606 | 8.3398 | 49.0107 | 8.2099 | 56521.7348 | 3005.0541 | 2.1810 | 11281.0863 |

## Agreement with JAX JIT

Errors are maximum absolute difference and relative Frobenius norm (||A−J||F / ||J||F). JAX is a comparison reference, not an exact oracle. NumPy and Rust use analytic derivatives; numerical=True uses central differences with the library default eps=1e-8.

| Model | Output | NumPy max abs | Rust max abs | Numerical max abs | Numerical relative | NumPy relative |
|---|---|---:|---:|---:|---:|---:|
| sample4 | momentum | 2.665e-15 | 6.661e-15 | 2.976e-08 | 1.644e-09 | 1.968e-16 |
| sample4 | force | 4.996e-15 | 1.599e-14 | 6.861e-08 | 5.585e-09 | 3.666e-16 |
| sample4 | torque | 1.421e-14 | 2.842e-14 | 2.078e-06 | 1.900e-08 | 1.730e-16 |
| sample4 | torque_diff1 | 4.619e-14 | 5.684e-14 | 2.366e-06 | 1.959e-08 | 6.160e-16 |
| sample4 | torque_diff2 | 1.634e-13 | 7.105e-14 | 3.230e-06 | 2.899e-08 | 1.904e-15 |
| branched3 | torque | 1.332e-15 | 1.554e-15 | 7.413e-08 | 4.291e-08 | 7.192e-16 |
| branched3 | torque_diff2 | 2.880e-15 | 1.776e-15 | 9.889e-08 | 2.247e-08 | 6.793e-16 |
| serial8 | torque | 1.137e-13 | 5.684e-14 | 2.420e-05 | 1.304e-07 | 8.246e-16 |
| serial8 | torque_diff2 | 3.695e-13 | 3.411e-13 | 9.886e-06 | 5.371e-08 | 1.837e-15 |
| serial16 | torque | 6.537e-13 | 5.116e-13 | 1.739e-04 | 1.146e-07 | 8.221e-16 |
| serial16 | torque_diff2 | 9.948e-12 | 4.547e-12 | 4.438e-04 | 1.167e-07 | 3.398e-15 |

## Measurement details

Analytic: 20 timed calls after two warmups. JAX JIT: at least 100 calls after two warmups. Eager: three calls after one warmup. Numerical: three calls, or one for 16 DOF, no warmup. JSON contains min/p90 and repeat counts. These are local machine measurements, not universal performance guarantees.

Each compiled function was additionally checked at two different random inputs against NumPy analytic derivatives. Model creation and initial backend initialization are excluded from warm execution timings.

Reproduce: `python -u -m developer.benchmarks.dynamics_autodiff_compare`
