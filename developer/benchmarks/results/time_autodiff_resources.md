# Jacobian CPU time and memory

Measured: 2026-09-16T06:37:17.841824+00:00; Linux-7.0.0-31-generic-x86_64-with-glibc2.39; CPU: x86_64.

7 DOF, float64, seed 20260916, gravity [0.3, -0.4, -9.81]. Model, inputs, environment, revision and raw timings are in JSON. A single random motion is measured repeatedly, not independent trials.

Every method/case uses a fresh sequential subprocess with JAX_PLATFORMS=cpu and OMP/OPENBLAS/MKL_NUM_THREADS=1. These settings do not guarantee single-threaded XLA. Process startup/import/model construction and common CPU runtime initialization are excluded. Cached analytic cases additionally exclude initial state preparation; full cases include motion import and state calculation without dictionary export. Numerical uses central differences, eps=1e-8. AD includes runtime input conversion, synchronized execution, and NumPy output conversion. Unprefixed forward/reverse AD computes analytic time derivatives directly on ordinary derivative coefficient series; eager may compute unused dynamics work. Methods prefixed time_ instead obtain velocity from FK directional JVPs, momentum rate from another JVP, and higher output derivatives from nested total-time JVPs before outer jacfwd/jacrev. Methods prefixed id_time_ start from ordinary ID(q,qdot,qddot), always motion order 3, and use total-time JVPs only for the higher torque derivatives. Methods prefixed cmtm_ instead construct explicit spatial lower block-Toeplitz CMTMs and use factorial-normalized coefficient vectors; only the outer motion Jacobian uses AD. The coefficient-series and explicit-CMTM paths are implementation representations of the same high-order algebra, not separate mathematical differentiation methods. Both use recurrences to construct transform coefficients. No trajectory is constructed: D_t f(x)=JVP(f,x,shift(x)) with x=(q,qdot,...); repeated AD differentiates the state-dependent direction too.

First call includes lazy initialization (and tracing/compilation for JIT). Warm medians exclude the first call and configured warmups. No JIT function closes over motion. A second seeded input is checked against NumPy analytic results outside measurement.

Peak RSS is the OS process-lifetime high-water mark, not live array size or device memory. It includes Python, libraries, compiler and allocator caches. Growth subtracts the baseline high-water mark after setup; it is NOT exact per-call allocation or a separately measured warm-only peak. Each memory measurement is one process run. Validation occurs after the memory snapshot.

Rust labels specify the requested state backend, not a guarantee of Rust derivative kernels. The selections retain explicit local frames from the accuracy table; current Rust fast paths require frame_name=None. Nonzero gravity also restricts spatial derivative kernels. These Rust-labelled measurements therefore include Python derivative fallback, including torque.

Completed: 40 / 40; failures: 0.

## Forward vs reverse AD

Speed ratio = forward median / reverse median; above 1 means reverse was faster. Peak RSS includes first call/compilation, even for JIT warm timings.

| Quantity | k | Mode | Forward ms | Reverse ms | Speed ratio | Forward peak MiB | Reverse peak MiB |
|---|---:|---|---:|---:|---:|---:|---:|
| link velocity / local | 0 | jit | 0.171 | 0.200 | 0.85 | 298.7 | 297.2 |
| link velocity / local | 0 | time_jit | 0.321 | 0.363 | 0.89 | 311.3 | 312.4 |
| link velocity / local | 1 | jit | 0.310 | 0.284 | 1.09 | 336.5 | 333.3 |
| link velocity / local | 1 | time_jit | 0.425 | 0.396 | 1.07 | 332.1 | 336.5 |
| link momentum / local | 0 | jit | 0.236 | 0.157 | 1.51 | 299.0 | 299.2 |
| link momentum / local | 0 | time_jit | 0.213 | 0.372 | 0.57 | 313.7 | 313.1 |
| link momentum / local | 1 | jit | 0.223 | 0.336 | 0.66 | 335.7 | 327.3 |
| link momentum / local | 1 | time_jit | 0.339 | 0.244 | 1.39 | 334.0 | 338.8 |
| link force / local | 0 | jit | 0.316 | 0.331 | 0.96 | 343.3 | 335.6 |
| link force / local | 0 | time_jit | 0.445 | 0.374 | 1.19 | 350.3 | 365.3 |
| link force / local | 1 | jit | 0.474 | 0.344 | 1.38 | 384.3 | 382.2 |
| link force / local | 1 | time_jit | 0.398 | 0.469 | 0.85 | 400.5 | 501.2 |
| joint torque | 0 | jit | 0.376 | 0.236 | 1.59 | 360.5 | 355.7 |
| joint torque | 0 | time_jit | 0.369 | 0.237 | 1.55 | 384.9 | 410.2 |
| joint torque | 1 | jit | 0.497 | 0.309 | 1.61 | 426.5 | 427.2 |
| joint torque | 1 | time_jit | 0.457 | 0.364 | 1.26 | 465.3 | 578.9 |

## All methods

| Quantity | k | Method | First ms | Warm median ms | n | Baseline MiB | Peak MiB | Growth MiB |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| link velocity / local | 0 | numpy_full | 6.706 | 1.781 | 5 | 183.9 | 185.4 | 1.5 |
| link velocity / local | 0 | forward_jit | 259.099 | 0.171 | 5 | 183.8 | 298.7 | 114.9 |
| link velocity / local | 0 | reverse_jit | 293.957 | 0.200 | 5 | 183.8 | 297.2 | 113.3 |
| link velocity / local | 0 | time_forward_jit | 245.394 | 0.321 | 5 | 232.4 | 311.3 | 78.8 |
| link velocity / local | 0 | time_reverse_jit | 331.594 | 0.363 | 5 | 232.1 | 312.4 | 80.2 |
| link momentum / local | 0 | numpy_full | 8.096 | 2.524 | 5 | 183.8 | 185.5 | 1.6 |
| link momentum / local | 0 | forward_jit | 262.410 | 0.236 | 5 | 183.9 | 299.0 | 115.1 |
| link momentum / local | 0 | reverse_jit | 292.630 | 0.157 | 5 | 183.9 | 299.2 | 115.3 |
| link momentum / local | 0 | time_forward_jit | 280.313 | 0.213 | 5 | 233.8 | 313.7 | 79.9 |
| link momentum / local | 0 | time_reverse_jit | 358.223 | 0.372 | 5 | 233.8 | 313.1 | 79.3 |
| link force / local | 0 | numpy_full | 21.897 | 16.236 | 5 | 183.9 | 186.7 | 2.8 |
| link force / local | 0 | forward_jit | 686.303 | 0.316 | 5 | 183.6 | 343.3 | 159.7 |
| link force / local | 0 | reverse_jit | 843.815 | 0.331 | 5 | 184.0 | 335.6 | 151.6 |
| link force / local | 0 | time_forward_jit | 781.125 | 0.445 | 5 | 233.6 | 350.3 | 116.7 |
| link force / local | 0 | time_reverse_jit | 1432.695 | 0.374 | 5 | 233.7 | 365.3 | 131.7 |
| joint torque | 0 | numpy_full | 29.976 | 24.073 | 5 | 183.9 | 187.7 | 3.8 |
| joint torque | 0 | forward_jit | 797.949 | 0.376 | 5 | 184.2 | 360.5 | 176.3 |
| joint torque | 0 | reverse_jit | 1056.971 | 0.236 | 5 | 184.1 | 355.7 | 171.6 |
| joint torque | 0 | time_forward_jit | 1201.949 | 0.369 | 5 | 234.1 | 384.9 | 150.7 |
| joint torque | 0 | time_reverse_jit | 1896.937 | 0.237 | 5 | 234.0 | 410.2 | 176.1 |
| link velocity / local | 1 | numpy_full | 7.496 | 2.456 | 5 | 183.9 | 185.1 | 1.2 |
| link velocity / local | 1 | forward_jit | 627.306 | 0.310 | 5 | 183.9 | 336.5 | 152.6 |
| link velocity / local | 1 | reverse_jit | 826.141 | 0.284 | 5 | 184.2 | 333.3 | 149.1 |
| link velocity / local | 1 | time_forward_jit | 455.721 | 0.425 | 5 | 234.1 | 332.1 | 98.0 |
| link velocity / local | 1 | time_reverse_jit | 728.546 | 0.396 | 5 | 234.1 | 336.5 | 102.4 |
| link momentum / local | 1 | numpy_full | 10.968 | 5.719 | 5 | 183.9 | 185.4 | 1.6 |
| link momentum / local | 1 | forward_jit | 635.624 | 0.223 | 5 | 184.2 | 335.7 | 151.5 |
| link momentum / local | 1 | reverse_jit | 816.555 | 0.336 | 5 | 184.0 | 327.3 | 143.3 |
| link momentum / local | 1 | time_forward_jit | 527.616 | 0.339 | 5 | 234.2 | 334.0 | 99.8 |
| link momentum / local | 1 | time_reverse_jit | 776.174 | 0.244 | 5 | 233.8 | 338.8 | 105.1 |
| link force / local | 1 | numpy_full | 32.307 | 25.941 | 5 | 183.5 | 187.5 | 4.0 |
| link force / local | 1 | forward_jit | 1232.201 | 0.474 | 5 | 184.2 | 384.3 | 200.2 |
| link force / local | 1 | reverse_jit | 1822.179 | 0.344 | 5 | 184.1 | 382.2 | 198.1 |
| link force / local | 1 | time_forward_jit | 1541.385 | 0.398 | 5 | 233.7 | 400.5 | 166.9 |
| link force / local | 1 | time_reverse_jit | 6086.234 | 0.469 | 5 | 233.6 | 501.2 | 267.6 |
| joint torque | 1 | numpy_full | 42.797 | 36.261 | 5 | 183.9 | 189.6 | 5.7 |
| joint torque | 1 | forward_jit | 1616.147 | 0.497 | 5 | 183.7 | 426.5 | 242.8 |
| joint torque | 1 | reverse_jit | 2468.274 | 0.309 | 5 | 184.0 | 427.2 | 243.2 |
| joint torque | 1 | time_forward_jit | 2522.465 | 0.457 | 5 | 233.7 | 465.3 | 231.5 |
| joint torque | 1 | time_reverse_jit | 6971.564 | 0.364 | 5 | 233.7 | 578.9 | 345.3 |

## Agreement against NumPy analytic

Errors are maximum absolute and relative Frobenius differences over the two tested inputs. NumPy analytic is a comparison reference, not an exact oracle.

| Quantity | k | Method | Max abs | Relative Frobenius |
|---|---:|---|---:|---:|
| link velocity / local | 0 | numpy_full | 0.000e+00 | 0.000e+00 |
| link velocity / local | 0 | forward_jit | 9.992e-16 | 4.486e-16 |
| link velocity / local | 0 | reverse_jit | 1.110e-15 | 4.708e-16 |
| link velocity / local | 0 | time_forward_jit | 1.443e-15 | 6.608e-16 |
| link velocity / local | 0 | time_reverse_jit | 1.221e-15 | 6.121e-16 |
| link momentum / local | 0 | numpy_full | 0.000e+00 | 0.000e+00 |
| link momentum / local | 0 | forward_jit | 4.885e-15 | 5.543e-16 |
| link momentum / local | 0 | reverse_jit | 5.329e-15 | 5.856e-16 |
| link momentum / local | 0 | time_forward_jit | 7.105e-15 | 7.399e-16 |
| link momentum / local | 0 | time_reverse_jit | 6.661e-15 | 6.739e-16 |
| link force / local | 0 | numpy_full | 0.000e+00 | 0.000e+00 |
| link force / local | 0 | forward_jit | 1.421e-14 | 2.778e-16 |
| link force / local | 0 | reverse_jit | 1.421e-14 | 3.408e-16 |
| link force / local | 0 | time_forward_jit | 1.421e-14 | 3.181e-16 |
| link force / local | 0 | time_reverse_jit | 1.421e-14 | 3.007e-16 |
| joint torque | 0 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 0 | forward_jit | 8.527e-14 | 5.529e-16 |
| joint torque | 0 | reverse_jit | 5.684e-14 | 5.121e-16 |
| joint torque | 0 | time_forward_jit | 4.974e-14 | 3.748e-16 |
| joint torque | 0 | time_reverse_jit | 8.527e-14 | 6.199e-16 |
| link velocity / local | 1 | numpy_full | 0.000e+00 | 0.000e+00 |
| link velocity / local | 1 | forward_jit | 1.665e-15 | 1.016e-15 |
| link velocity / local | 1 | reverse_jit | 1.665e-15 | 1.017e-15 |
| link velocity / local | 1 | time_forward_jit | 1.665e-15 | 1.236e-15 |
| link velocity / local | 1 | time_reverse_jit | 1.499e-15 | 1.224e-15 |
| link momentum / local | 1 | numpy_full | 0.000e+00 | 0.000e+00 |
| link momentum / local | 1 | forward_jit | 8.438e-15 | 1.239e-15 |
| link momentum / local | 1 | reverse_jit | 7.994e-15 | 1.252e-15 |
| link momentum / local | 1 | time_forward_jit | 8.438e-15 | 1.416e-15 |
| link momentum / local | 1 | time_reverse_jit | 7.994e-15 | 1.316e-15 |
| link force / local | 1 | numpy_full | 0.000e+00 | 0.000e+00 |
| link force / local | 1 | forward_jit | 2.132e-14 | 3.832e-16 |
| link force / local | 1 | reverse_jit | 2.487e-14 | 4.299e-16 |
| link force / local | 1 | time_forward_jit | 2.487e-14 | 4.155e-16 |
| link force / local | 1 | time_reverse_jit | 2.842e-14 | 4.523e-16 |
| joint torque | 1 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 1 | forward_jit | 9.948e-14 | 6.286e-16 |
| joint torque | 1 | reverse_jit | 9.237e-14 | 7.149e-16 |
| joint torque | 1 | time_forward_jit | 8.527e-14 | 6.023e-16 |
| joint torque | 1 | time_reverse_jit | 9.237e-14 | 6.664e-16 |
