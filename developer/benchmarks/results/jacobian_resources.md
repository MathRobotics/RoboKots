# Jacobian CPU time and memory

Measured: 2026-09-16T05:11:08.318247+00:00; Linux-7.0.0-31-generic-x86_64-with-glibc2.39; CPU: x86_64.

7 DOF, float64, seed 20260916, gravity [0.3, -0.4, -9.81]. Model, inputs, environment, revision and raw timings are in JSON. A single random motion is measured repeatedly, not independent trials.

Every method/case uses a fresh sequential subprocess with JAX_PLATFORMS=cpu and OMP/OPENBLAS/MKL_NUM_THREADS=1. These settings do not guarantee single-threaded XLA. Process startup/import/model construction and common CPU runtime initialization are excluded. Cached analytic cases additionally exclude initial state preparation; full cases include motion import and state calculation without dictionary export. Numerical uses central differences, eps=1e-8. AD includes runtime input conversion, synchronized execution, and NumPy output conversion. Unprefixed forward/reverse AD computes analytic time derivatives directly on ordinary derivative coefficient series; eager may compute unused dynamics work. Methods prefixed time_ instead obtain velocity from FK directional JVPs, momentum rate from another JVP, and higher output derivatives from nested total-time JVPs before outer jacfwd/jacrev. Methods prefixed id_time_ start from ordinary ID(q,qdot,qddot), always motion order 3, and use total-time JVPs only for the higher torque derivatives. Methods prefixed cmtm_ instead construct explicit spatial lower block-Toeplitz CMTMs and use factorial-normalized coefficient vectors; only the outer motion Jacobian uses AD. The coefficient-series and explicit-CMTM paths are implementation representations of the same high-order algebra, not separate mathematical differentiation methods. Both use recurrences to construct transform coefficients. No trajectory is constructed: D_t f(x)=JVP(f,x,shift(x)) with x=(q,qdot,...); repeated AD differentiates the state-dependent direction too.

First call includes lazy initialization (and tracing/compilation for JIT). Warm medians exclude the first call and configured warmups. No JIT function closes over motion. A second seeded input is checked against NumPy analytic results outside measurement.

Peak RSS is the OS process-lifetime high-water mark, not live array size or device memory. It includes Python, libraries, compiler and allocator caches. Growth subtracts the baseline high-water mark after setup; it is NOT exact per-call allocation or a separately measured warm-only peak. Each memory measurement is one process run. Validation occurs after the memory snapshot.

Rust labels specify the requested state backend, not a guarantee of Rust derivative kernels. The selections retain explicit local frames from the accuracy table; current Rust fast paths require frame_name=None. Nonzero gravity also restricts spatial derivative kernels. These Rust-labelled measurements therefore include Python derivative fallback, including torque.

Completed: 72 / 72; failures: 0.

## Forward vs reverse AD

Speed ratio = forward median / reverse median; above 1 means reverse was faster. Peak RSS includes first call/compilation, even for JIT warm timings.

| Quantity | k | Mode | Forward ms | Reverse ms | Speed ratio | Forward peak MiB | Reverse peak MiB |
|---|---:|---|---:|---:|---:|---:|---:|
| link velocity / local | 0 | eager | 62.406 | 101.387 | 0.62 | 409.1 | 419.7 |
| link velocity / local | 0 | jit | 0.157 | 0.226 | 0.70 | 296.8 | 299.0 |
| link velocity / local | 4 | eager | 1983.375 | 4868.992 | 0.41 | 513.6 | 546.7 |
| link velocity / local | 4 | jit | 0.335 | 0.374 | 0.90 | 481.4 | 486.6 |
| link momentum / local | 0 | eager | 63.644 | 101.579 | 0.63 | 409.6 | 421.0 |
| link momentum / local | 0 | jit | 0.259 | 0.219 | 1.18 | 299.8 | 300.0 |
| link momentum / local | 4 | eager | 1964.707 | 4837.213 | 0.41 | 512.1 | 544.0 |
| link momentum / local | 4 | jit | 0.356 | 0.474 | 0.75 | 484.1 | 485.6 |
| link force / local | 0 | eager | 276.409 | 613.535 | 0.45 | 490.7 | 492.8 |
| link force / local | 0 | jit | 0.294 | 0.368 | 0.80 | 339.8 | 336.7 |
| link force / local | 4 | eager | 2882.378 | 7371.499 | 0.39 | 516.4 | 577.2 |
| link force / local | 4 | jit | 0.818 | 0.683 | 1.20 | 573.7 | 602.9 |
| joint torque | 0 | eager | 272.912 | 660.445 | 0.41 | 491.4 | 510.7 |
| joint torque | 0 | jit | 0.383 | 0.167 | 2.29 | 362.3 | 355.1 |
| joint torque | 4 | eager | 2868.046 | 8535.002 | 0.34 | 521.0 | 626.4 |
| joint torque | 4 | jit | 1.241 | 0.484 | 2.56 | 709.1 | 875.3 |

## All methods

| Quantity | k | Method | First ms | Warm median ms | n | Baseline MiB | Peak MiB | Growth MiB |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| link velocity / local | 0 | numpy_full | 6.690 | 1.785 | 5 | 184.1 | 185.6 | 1.5 |
| link velocity / local | 0 | numpy_cached | 1.771 | 0.202 | 5 | 184.9 | 185.2 | 0.3 |
| link velocity / local | 0 | rust_full | 7.054 | 1.046 | 5 | 183.8 | 186.3 | 2.5 |
| link velocity / local | 0 | rust_cached | 4.237 | 0.134 | 5 | 185.2 | 186.3 | 1.1 |
| link velocity / local | 0 | numerical | 18.299 | 13.416 | 1 | 183.9 | 185.0 | 1.1 |
| link velocity / local | 0 | forward_eager | 1922.991 | 62.406 | 5 | 184.0 | 409.1 | 225.1 |
| link velocity / local | 0 | reverse_eager | 1999.604 | 101.387 | 5 | 183.9 | 419.7 | 235.8 |
| link velocity / local | 0 | forward_jit | 265.194 | 0.157 | 5 | 183.8 | 296.8 | 112.9 |
| link velocity / local | 0 | reverse_jit | 289.791 | 0.226 | 5 | 184.1 | 299.0 | 114.9 |
| link momentum / local | 0 | numpy_full | 7.719 | 2.603 | 5 | 184.2 | 185.6 | 1.4 |
| link momentum / local | 0 | numpy_cached | 1.590 | 0.234 | 5 | 185.1 | 185.5 | 0.4 |
| link momentum / local | 0 | rust_full | 7.248 | 1.111 | 5 | 183.9 | 186.6 | 2.7 |
| link momentum / local | 0 | rust_cached | 4.310 | 0.164 | 5 | 185.4 | 186.4 | 1.0 |
| link momentum / local | 0 | numerical | 39.003 | 33.833 | 1 | 183.8 | 185.3 | 1.4 |
| link momentum / local | 0 | forward_eager | 1912.434 | 63.644 | 5 | 183.9 | 409.6 | 225.7 |
| link momentum / local | 0 | reverse_eager | 2006.606 | 101.579 | 5 | 183.7 | 421.0 | 237.3 |
| link momentum / local | 0 | forward_jit | 266.627 | 0.259 | 5 | 183.8 | 299.8 | 116.0 |
| link momentum / local | 0 | reverse_jit | 291.734 | 0.219 | 5 | 183.8 | 300.0 | 116.2 |
| link force / local | 0 | numpy_full | 22.458 | 16.805 | 5 | 183.8 | 186.7 | 2.9 |
| link force / local | 0 | numpy_cached | 13.295 | 4.634 | 5 | 185.5 | 186.9 | 1.4 |
| link force / local | 0 | rust_full | 16.510 | 9.960 | 5 | 183.8 | 187.8 | 4.1 |
| link force / local | 0 | rust_cached | 13.615 | 2.659 | 5 | 185.6 | 187.8 | 2.2 |
| link force / local | 0 | numerical | 165.994 | 160.948 | 1 | 184.0 | 185.3 | 1.3 |
| link force / local | 0 | forward_eager | 3212.804 | 276.409 | 5 | 183.8 | 490.7 | 306.8 |
| link force / local | 0 | reverse_eager | 3264.828 | 613.535 | 5 | 184.1 | 492.8 | 308.7 |
| link force / local | 0 | forward_jit | 663.095 | 0.294 | 5 | 183.6 | 339.8 | 156.2 |
| link force / local | 0 | reverse_jit | 855.267 | 0.368 | 5 | 184.3 | 336.7 | 152.4 |
| joint torque | 0 | numpy_full | 29.621 | 23.705 | 5 | 183.7 | 187.5 | 3.7 |
| joint torque | 0 | numpy_cached | 21.066 | 7.845 | 5 | 185.2 | 187.3 | 2.0 |
| joint torque | 0 | rust_full | 22.395 | 15.779 | 5 | 184.0 | 188.5 | 4.6 |
| joint torque | 0 | rust_cached | 19.279 | 4.819 | 5 | 185.3 | 188.1 | 2.7 |
| joint torque | 0 | numerical | 167.359 | 162.278 | 1 | 183.9 | 185.3 | 1.3 |
| joint torque | 0 | forward_eager | 3193.295 | 272.912 | 5 | 184.0 | 491.4 | 307.4 |
| joint torque | 0 | reverse_eager | 3489.683 | 660.445 | 5 | 184.0 | 510.7 | 326.7 |
| joint torque | 0 | forward_jit | 797.533 | 0.383 | 5 | 183.6 | 362.3 | 178.7 |
| joint torque | 0 | reverse_jit | 1049.232 | 0.167 | 5 | 183.7 | 355.1 | 171.4 |
| link velocity / local | 4 | numpy_full | 11.611 | 6.387 | 5 | 183.9 | 186.1 | 2.2 |
| link velocity / local | 4 | numpy_cached | 4.897 | 0.266 | 5 | 185.3 | 186.3 | 1.0 |
| link velocity / local | 4 | rust_full | 13.159 | 6.968 | 5 | 184.0 | 186.9 | 2.9 |
| link velocity / local | 4 | rust_cached | 10.170 | 0.183 | 5 | 185.4 | 186.9 | 1.5 |
| link velocity / local | 4 | numerical | 180.446 | 175.237 | 1 | 183.9 | 185.0 | 1.1 |
| link velocity / local | 4 | forward_eager | 5619.973 | 1983.375 | 5 | 184.2 | 513.6 | 329.5 |
| link velocity / local | 4 | reverse_eager | 7733.330 | 4868.992 | 5 | 184.0 | 546.7 | 362.7 |
| link velocity / local | 4 | forward_jit | 3095.899 | 0.335 | 5 | 184.0 | 481.4 | 297.4 |
| link velocity / local | 4 | reverse_jit | 5188.706 | 0.374 | 5 | 183.7 | 486.6 | 302.9 |
| link momentum / local | 4 | numpy_full | 19.298 | 13.929 | 5 | 183.9 | 186.3 | 2.4 |
| link momentum / local | 4 | numpy_cached | 4.768 | 0.286 | 5 | 185.8 | 186.6 | 0.7 |
| link momentum / local | 4 | rust_full | 12.951 | 6.886 | 5 | 184.2 | 187.0 | 2.8 |
| link momentum / local | 4 | rust_cached | 10.320 | 0.213 | 5 | 185.3 | 186.7 | 1.5 |
| link momentum / local | 4 | numerical | 798.324 | 789.902 | 1 | 183.3 | 185.3 | 1.9 |
| link momentum / local | 4 | forward_eager | 5158.136 | 1964.707 | 5 | 184.0 | 512.1 | 328.1 |
| link momentum / local | 4 | reverse_eager | 7726.186 | 4837.213 | 5 | 183.7 | 544.0 | 360.3 |
| link momentum / local | 4 | forward_jit | 3136.093 | 0.356 | 5 | 183.9 | 484.1 | 300.2 |
| link momentum / local | 4 | reverse_jit | 5189.943 | 0.474 | 5 | 183.6 | 485.6 | 302.1 |
| link force / local | 4 | numpy_full | 68.893 | 61.706 | 5 | 183.8 | 193.9 | 10.1 |
| link force / local | 4 | numpy_cached | 51.051 | 17.873 | 5 | 185.8 | 194.0 | 8.2 |
| link force / local | 4 | rust_full | 63.260 | 55.316 | 5 | 183.8 | 194.3 | 10.4 |
| link force / local | 4 | rust_cached | 61.179 | 11.356 | 5 | 185.4 | 194.4 | 8.9 |
| link force / local | 4 | numerical | 1138.540 | 1132.930 | 1 | 183.8 | 185.7 | 1.9 |
| link force / local | 4 | forward_eager | 6301.810 | 2882.378 | 5 | 184.0 | 516.4 | 332.3 |
| link force / local | 4 | reverse_eager | 10309.871 | 7371.499 | 5 | 184.0 | 577.2 | 393.3 |
| link force / local | 4 | forward_jit | 4987.823 | 0.818 | 5 | 184.1 | 573.7 | 389.6 |
| link force / local | 4 | reverse_jit | 8677.417 | 0.683 | 5 | 184.0 | 602.9 | 418.9 |
| joint torque | 4 | numpy_full | 98.962 | 93.160 | 5 | 183.9 | 198.0 | 14.2 |
| joint torque | 4 | numpy_cached | 81.657 | 34.224 | 5 | 185.4 | 198.0 | 12.6 |
| joint torque | 4 | rust_full | 98.221 | 91.805 | 5 | 184.1 | 198.8 | 14.7 |
| joint torque | 4 | rust_cached | 94.649 | 26.596 | 5 | 185.6 | 198.7 | 13.1 |
| joint torque | 4 | numerical | 1129.497 | 1127.607 | 1 | 183.8 | 185.7 | 1.9 |
| joint torque | 4 | forward_eager | 6240.024 | 2868.046 | 5 | 184.1 | 521.0 | 337.0 |
| joint torque | 4 | reverse_eager | 11535.403 | 8535.002 | 5 | 183.9 | 626.4 | 442.6 |
| joint torque | 4 | forward_jit | 7422.910 | 1.241 | 5 | 183.8 | 709.1 | 525.3 |
| joint torque | 4 | reverse_jit | 14145.863 | 0.484 | 5 | 184.0 | 875.3 | 691.3 |

## Agreement against NumPy analytic

Errors are maximum absolute and relative Frobenius differences over the two tested inputs. NumPy analytic is a comparison reference, not an exact oracle.

| Quantity | k | Method | Max abs | Relative Frobenius |
|---|---:|---|---:|---:|
| link velocity / local | 0 | numpy_full | 0.000e+00 | 0.000e+00 |
| link velocity / local | 0 | numpy_cached | 0.000e+00 | 0.000e+00 |
| link velocity / local | 0 | rust_full | 1.998e-15 | 8.359e-16 |
| link velocity / local | 0 | rust_cached | 1.998e-15 | 8.359e-16 |
| link velocity / local | 0 | numerical | 1.991e-08 | 1.023e-08 |
| link velocity / local | 0 | forward_eager | 9.992e-16 | 4.487e-16 |
| link velocity / local | 0 | reverse_eager | 1.110e-15 | 4.764e-16 |
| link velocity / local | 0 | forward_jit | 9.992e-16 | 4.486e-16 |
| link velocity / local | 0 | reverse_jit | 1.110e-15 | 4.708e-16 |
| link momentum / local | 0 | numpy_full | 0.000e+00 | 0.000e+00 |
| link momentum / local | 0 | numpy_cached | 0.000e+00 | 0.000e+00 |
| link momentum / local | 0 | rust_full | 9.770e-15 | 1.015e-15 |
| link momentum / local | 0 | rust_cached | 9.770e-15 | 1.015e-15 |
| link momentum / local | 0 | numerical | 6.196e-08 | 1.022e-08 |
| link momentum / local | 0 | forward_eager | 4.885e-15 | 5.544e-16 |
| link momentum / local | 0 | reverse_eager | 5.329e-15 | 5.762e-16 |
| link momentum / local | 0 | forward_jit | 4.885e-15 | 5.543e-16 |
| link momentum / local | 0 | reverse_jit | 5.329e-15 | 5.856e-16 |
| link force / local | 0 | numpy_full | 0.000e+00 | 0.000e+00 |
| link force / local | 0 | numpy_cached | 0.000e+00 | 0.000e+00 |
| link force / local | 0 | rust_full | 2.132e-14 | 3.172e-16 |
| link force / local | 0 | rust_cached | 2.132e-14 | 3.172e-16 |
| link force / local | 0 | numerical | 8.904e-07 | 1.740e-08 |
| link force / local | 0 | forward_eager | 1.421e-14 | 2.736e-16 |
| link force / local | 0 | reverse_eager | 1.421e-14 | 3.262e-16 |
| link force / local | 0 | forward_jit | 1.421e-14 | 2.778e-16 |
| link force / local | 0 | reverse_jit | 1.421e-14 | 3.408e-16 |
| joint torque | 0 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 0 | numpy_cached | 0.000e+00 | 0.000e+00 |
| joint torque | 0 | rust_full | 2.842e-14 | 2.797e-16 |
| joint torque | 0 | rust_cached | 2.842e-14 | 2.797e-16 |
| joint torque | 0 | numerical | 8.261e-06 | 5.832e-08 |
| joint torque | 0 | forward_eager | 5.684e-14 | 3.904e-16 |
| joint torque | 0 | reverse_eager | 4.263e-14 | 4.375e-16 |
| joint torque | 0 | forward_jit | 8.527e-14 | 5.529e-16 |
| joint torque | 0 | reverse_jit | 5.684e-14 | 5.121e-16 |
| link velocity / local | 4 | numpy_full | 0.000e+00 | 0.000e+00 |
| link velocity / local | 4 | numpy_cached | 0.000e+00 | 0.000e+00 |
| link velocity / local | 4 | rust_full | 7.061e-14 | 3.806e-15 |
| link velocity / local | 4 | rust_cached | 7.061e-14 | 3.806e-15 |
| link velocity / local | 4 | numerical | 8.771e-07 | 7.781e-08 |
| link velocity / local | 4 | forward_eager | 3.464e-14 | 3.104e-15 |
| link velocity / local | 4 | reverse_eager | 3.508e-14 | 3.106e-15 |
| link velocity / local | 4 | forward_jit | 3.553e-14 | 3.112e-15 |
| link velocity / local | 4 | reverse_jit | 3.508e-14 | 3.119e-15 |
| link momentum / local | 4 | numpy_full | 0.000e+00 | 0.000e+00 |
| link momentum / local | 4 | numpy_cached | 0.000e+00 | 0.000e+00 |
| link momentum / local | 4 | rust_full | 3.517e-13 | 4.903e-15 |
| link momentum / local | 4 | rust_cached | 3.517e-13 | 4.903e-15 |
| link momentum / local | 4 | numerical | 4.352e-06 | 8.921e-08 |
| link momentum / local | 4 | forward_eager | 1.741e-13 | 3.573e-15 |
| link momentum / local | 4 | reverse_eager | 1.741e-13 | 3.566e-15 |
| link momentum / local | 4 | forward_jit | 1.776e-13 | 3.591e-15 |
| link momentum / local | 4 | reverse_jit | 1.741e-13 | 3.579e-15 |
| link force / local | 4 | numpy_full | 0.000e+00 | 0.000e+00 |
| link force / local | 4 | numpy_cached | 0.000e+00 | 0.000e+00 |
| link force / local | 4 | rust_full | 1.478e-12 | 1.779e-15 |
| link force / local | 4 | rust_cached | 1.478e-12 | 1.779e-15 |
| link force / local | 4 | numerical | 2.146e-05 | 3.686e-08 |
| link force / local | 4 | forward_eager | 1.648e-12 | 2.786e-15 |
| link force / local | 4 | reverse_eager | 1.648e-12 | 2.828e-15 |
| link force / local | 4 | forward_jit | 1.648e-12 | 2.825e-15 |
| link force / local | 4 | reverse_jit | 1.648e-12 | 2.842e-15 |
| joint torque | 4 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 4 | numpy_cached | 0.000e+00 | 0.000e+00 |
| joint torque | 4 | rust_full | 1.137e-12 | 1.894e-15 |
| joint torque | 4 | rust_cached | 1.137e-12 | 1.894e-15 |
| joint torque | 4 | numerical | 3.700e-05 | 8.269e-08 |
| joint torque | 4 | forward_eager | 1.734e-12 | 2.552e-15 |
| joint torque | 4 | reverse_eager | 1.961e-12 | 2.766e-15 |
| joint torque | 4 | forward_jit | 1.876e-12 | 2.526e-15 |
| joint torque | 4 | reverse_jit | 1.904e-12 | 2.757e-15 |
