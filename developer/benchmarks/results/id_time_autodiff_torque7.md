# Jacobian CPU time and memory

Measured: 2026-09-16T09:30:08.908480+00:00; Linux-7.0.0-31-generic-x86_64-with-glibc2.39; CPU: x86_64.

7 DOF, float64, seed 20260916, gravity [0.3, -0.4, -9.81]. Model, inputs, environment, revision and raw timings are in JSON. A single random motion is measured repeatedly, not independent trials.

Every method/case uses a fresh sequential subprocess with JAX_PLATFORMS=cpu and OMP/OPENBLAS/MKL_NUM_THREADS=1. These settings do not guarantee single-threaded XLA. Process startup/import/model construction and common CPU runtime initialization are excluded. Cached analytic cases additionally exclude initial state preparation; full cases include motion import and state calculation without dictionary export. Numerical uses central differences, eps=1e-8. AD includes runtime input conversion, synchronized execution, and NumPy output conversion. Unprefixed forward/reverse AD computes analytic time derivatives directly on ordinary derivative coefficient series; eager may compute unused dynamics work. Methods prefixed time_ instead obtain velocity from FK directional JVPs, momentum rate from another JVP, and higher output derivatives from nested total-time JVPs before outer jacfwd/jacrev. Methods prefixed id_time_ start from ordinary ID(q,qdot,qddot), always motion order 3, and use total-time JVPs only for the higher torque derivatives. Methods prefixed cmtm_ instead construct explicit spatial lower block-Toeplitz CMTMs and use factorial-normalized coefficient vectors; only the outer motion Jacobian uses AD. The coefficient-series and explicit-CMTM paths are implementation representations of the same high-order algebra, not separate mathematical differentiation methods. Both use recurrences to construct transform coefficients. No trajectory is constructed: D_t f(x)=JVP(f,x,shift(x)) with x=(q,qdot,...); repeated AD differentiates the state-dependent direction too.

First call includes lazy initialization (and tracing/compilation for JIT). Warm medians exclude the first call and configured warmups. No JIT function closes over motion. A second seeded input is checked against NumPy analytic results outside measurement.

Peak RSS is the OS process-lifetime high-water mark, not live array size or device memory. It includes Python, libraries, compiler and allocator caches. Growth subtracts the baseline high-water mark after setup; it is NOT exact per-call allocation or a separately measured warm-only peak. Each memory measurement is one process run. Validation occurs after the memory snapshot.

Rust labels specify the requested state backend, not a guarantee of Rust derivative kernels. The selections retain explicit local frames from the accuracy table; current Rust fast paths require frame_name=None. Nonzero gravity also restricts spatial derivative kernels. These Rust-labelled measurements therefore include Python derivative fallback, including torque.

Completed: 25 / 25; failures: 0.

## Forward vs reverse AD

Speed ratio = forward median / reverse median; above 1 means reverse was faster. Peak RSS includes first call/compilation, even for JIT warm timings.

| Quantity | k | Mode | Forward ms | Reverse ms | Speed ratio | Forward peak MiB | Reverse peak MiB |
|---|---:|---|---:|---:|---:|---:|---:|
| joint torque | 0 | jit | 0.332 | 0.177 | 1.88 | 356.3 | 355.1 |
| joint torque | 0 | id_time_jit | 0.263 | 0.168 | 1.56 | 361.3 | 357.6 |
| joint torque | 1 | jit | 0.371 | 0.215 | 1.73 | 424.8 | 433.6 |
| joint torque | 1 | id_time_jit | 0.440 | 0.227 | 1.94 | 420.4 | 421.1 |
| joint torque | 2 | jit | 0.546 | 0.272 | 2.01 | 506.2 | 536.5 |
| joint torque | 2 | id_time_jit | 0.606 | 0.274 | 2.21 | 499.5 | 517.2 |
| joint torque | 3 | jit | 0.820 | 0.413 | 1.98 | 606.8 | 680.7 |
| joint torque | 3 | id_time_jit | 0.830 | 0.361 | 2.30 | 638.7 | 694.2 |
| joint torque | 4 | jit | 1.259 | 0.478 | 2.64 | 709.1 | 872.9 |
| joint torque | 4 | id_time_jit | 1.241 | 0.600 | 2.07 | 946.6 | 1022.8 |

## All methods

| Quantity | k | Method | First ms | Warm median ms | n | Baseline MiB | Peak MiB | Growth MiB |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| joint torque | 0 | numpy_full | 29.353 | 23.270 | 5 | 184.1 | 187.7 | 3.6 |
| joint torque | 0 | forward_jit | 795.484 | 0.332 | 5 | 184.2 | 356.3 | 172.2 |
| joint torque | 0 | reverse_jit | 1041.824 | 0.177 | 5 | 184.2 | 355.1 | 170.9 |
| joint torque | 0 | id_time_forward_jit | 773.519 | 0.263 | 5 | 209.7 | 361.3 | 151.5 |
| joint torque | 0 | id_time_reverse_jit | 1036.429 | 0.168 | 5 | 210.1 | 357.6 | 147.5 |
| joint torque | 1 | numpy_full | 42.999 | 36.554 | 5 | 183.9 | 189.4 | 5.5 |
| joint torque | 1 | forward_jit | 1634.441 | 0.371 | 5 | 184.0 | 424.8 | 240.8 |
| joint torque | 1 | reverse_jit | 2436.071 | 0.215 | 5 | 183.7 | 433.6 | 250.0 |
| joint torque | 1 | id_time_forward_jit | 1495.586 | 0.440 | 5 | 209.8 | 420.4 | 210.6 |
| joint torque | 1 | id_time_reverse_jit | 2027.932 | 0.227 | 5 | 209.9 | 421.1 | 211.1 |
| joint torque | 2 | numpy_full | 59.138 | 52.681 | 5 | 183.7 | 191.8 | 8.1 |
| joint torque | 2 | forward_jit | 3010.662 | 0.546 | 5 | 183.7 | 506.2 | 322.5 |
| joint torque | 2 | reverse_jit | 5083.966 | 0.272 | 5 | 184.1 | 536.5 | 352.4 |
| joint torque | 2 | id_time_forward_jit | 3120.913 | 0.606 | 5 | 210.4 | 499.5 | 289.1 |
| joint torque | 2 | id_time_reverse_jit | 4216.587 | 0.274 | 5 | 210.2 | 517.2 | 307.0 |
| joint torque | 3 | numpy_full | 77.879 | 70.775 | 5 | 184.2 | 194.9 | 10.7 |
| joint torque | 3 | forward_jit | 4913.473 | 0.820 | 5 | 183.9 | 606.8 | 423.0 |
| joint torque | 3 | reverse_jit | 8744.623 | 0.413 | 5 | 184.1 | 680.7 | 496.6 |
| joint torque | 3 | id_time_forward_jit | 6438.481 | 0.830 | 5 | 209.7 | 638.7 | 429.0 |
| joint torque | 3 | id_time_reverse_jit | 11436.633 | 0.361 | 5 | 210.1 | 694.2 | 484.2 |
| joint torque | 4 | numpy_full | 99.125 | 93.556 | 5 | 184.1 | 198.7 | 14.6 |
| joint torque | 4 | forward_jit | 7464.991 | 1.259 | 5 | 184.1 | 709.1 | 524.9 |
| joint torque | 4 | reverse_jit | 14086.346 | 0.478 | 5 | 184.0 | 872.9 | 688.8 |
| joint torque | 4 | id_time_forward_jit | 13266.783 | 1.241 | 5 | 210.2 | 946.6 | 736.4 |
| joint torque | 4 | id_time_reverse_jit | 25679.949 | 0.600 | 5 | 210.3 | 1022.8 | 812.5 |

## Agreement against NumPy analytic

Errors are maximum absolute and relative Frobenius differences over the two tested inputs. NumPy analytic is a comparison reference, not an exact oracle.

| Quantity | k | Method | Max abs | Relative Frobenius |
|---|---:|---|---:|---:|
| joint torque | 0 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 0 | forward_jit | 8.527e-14 | 5.529e-16 |
| joint torque | 0 | reverse_jit | 5.684e-14 | 5.121e-16 |
| joint torque | 0 | id_time_forward_jit | 8.527e-14 | 5.529e-16 |
| joint torque | 0 | id_time_reverse_jit | 5.684e-14 | 5.121e-16 |
| joint torque | 1 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 1 | forward_jit | 9.948e-14 | 6.286e-16 |
| joint torque | 1 | reverse_jit | 9.237e-14 | 7.149e-16 |
| joint torque | 1 | id_time_forward_jit | 9.948e-14 | 6.358e-16 |
| joint torque | 1 | id_time_reverse_jit | 1.066e-13 | 6.970e-16 |
| joint torque | 2 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 2 | forward_jit | 3.979e-13 | 1.279e-15 |
| joint torque | 2 | reverse_jit | 3.695e-13 | 1.327e-15 |
| joint torque | 2 | id_time_forward_jit | 3.979e-13 | 1.311e-15 |
| joint torque | 2 | id_time_reverse_jit | 3.695e-13 | 1.317e-15 |
| joint torque | 3 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 3 | forward_jit | 6.821e-13 | 1.722e-15 |
| joint torque | 3 | reverse_jit | 7.958e-13 | 1.895e-15 |
| joint torque | 3 | id_time_forward_jit | 7.958e-13 | 1.746e-15 |
| joint torque | 3 | id_time_reverse_jit | 6.821e-13 | 1.902e-15 |
| joint torque | 4 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 4 | forward_jit | 1.876e-12 | 2.526e-15 |
| joint torque | 4 | reverse_jit | 1.904e-12 | 2.757e-15 |
| joint torque | 4 | id_time_forward_jit | 1.791e-12 | 2.743e-15 |
| joint torque | 4 | id_time_reverse_jit | 2.089e-12 | 2.952e-15 |

## Output-value agreement

The time-differentiated quantities themselves are also checked at both inputs, outside all timing/memory measurements. These are not Jacobian errors.

| Quantity | k | Method | Max abs | Relative Frobenius |
|---|---:|---|---:|---:|
| joint torque | 0 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 0 | forward_jit | 5.684e-14 | 3.709e-16 |
| joint torque | 0 | reverse_jit | 5.684e-14 | 3.709e-16 |
| joint torque | 0 | id_time_forward_jit | 5.684e-14 | 3.709e-16 |
| joint torque | 0 | id_time_reverse_jit | 5.684e-14 | 3.709e-16 |
| joint torque | 1 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 1 | forward_jit | 2.132e-14 | 7.842e-16 |
| joint torque | 1 | reverse_jit | 2.132e-14 | 7.842e-16 |
| joint torque | 1 | id_time_forward_jit | 3.553e-14 | 7.842e-16 |
| joint torque | 1 | id_time_reverse_jit | 3.553e-14 | 7.842e-16 |
| joint torque | 2 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 2 | forward_jit | 3.979e-13 | 2.531e-15 |
| joint torque | 2 | reverse_jit | 3.979e-13 | 2.531e-15 |
| joint torque | 2 | id_time_forward_jit | 4.263e-13 | 2.711e-15 |
| joint torque | 2 | id_time_reverse_jit | 4.263e-13 | 2.711e-15 |
| joint torque | 3 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 3 | forward_jit | 3.908e-13 | 1.128e-14 |
| joint torque | 3 | reverse_jit | 3.908e-13 | 1.128e-14 |
| joint torque | 3 | id_time_forward_jit | 3.126e-13 | 9.022e-15 |
| joint torque | 3 | id_time_reverse_jit | 3.126e-13 | 9.022e-15 |
| joint torque | 4 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 4 | forward_jit | 4.832e-13 | 6.437e-15 |
| joint torque | 4 | reverse_jit | 4.832e-13 | 6.437e-15 |
| joint torque | 4 | id_time_forward_jit | 7.105e-13 | 6.568e-15 |
| joint torque | 4 | id_time_reverse_jit | 7.105e-13 | 6.568e-15 |
