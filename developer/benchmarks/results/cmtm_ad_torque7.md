# Jacobian CPU time and memory

Measured: 2026-09-17T01:10:29.484851+00:00; Linux-7.0.0-31-generic-x86_64-with-glibc2.39; CPU: x86_64.

7 DOF, float64, seed 20260916, gravity [0.3, -0.4, -9.81]. Model, inputs, environment, revision and raw timings are in JSON. A single random motion is measured repeatedly, not independent trials.

Every method/case uses a fresh sequential subprocess with JAX_PLATFORMS=cpu and OMP/OPENBLAS/MKL_NUM_THREADS=1. These settings do not guarantee single-threaded XLA. Process startup/import/model construction and common CPU runtime initialization are excluded. Cached analytic cases additionally exclude initial state preparation; full cases include motion import and state calculation without dictionary export. Numerical uses central differences, eps=1e-8. AD includes runtime input conversion, synchronized execution, and NumPy output conversion. Unprefixed forward/reverse AD computes analytic time derivatives directly on ordinary derivative coefficient series; eager may compute unused dynamics work. Methods prefixed time_ instead obtain velocity from FK directional JVPs, momentum rate from another JVP, and higher output derivatives from nested total-time JVPs before outer jacfwd/jacrev. Methods prefixed id_time_ start from ordinary ID(q,qdot,qddot), always motion order 3, and use total-time JVPs only for the higher torque derivatives. Methods prefixed cmtm_ instead construct explicit spatial lower block-Toeplitz CMTMs and use factorial-normalized coefficient vectors; only the outer motion Jacobian uses AD. The coefficient-series and explicit-CMTM paths are implementation representations of the same high-order algebra, not separate mathematical differentiation methods. Both use recurrences to construct transform coefficients. No trajectory is constructed: D_t f(x)=JVP(f,x,shift(x)) with x=(q,qdot,...); repeated AD differentiates the state-dependent direction too.

First call includes lazy initialization (and tracing/compilation for JIT). Warm medians exclude the first call and configured warmups. No JIT function closes over motion. A second seeded input is checked against NumPy analytic results outside measurement.

Peak RSS is the OS process-lifetime high-water mark, not live array size or device memory. It includes Python, libraries, compiler and allocator caches. Growth subtracts the baseline high-water mark after setup; it is NOT exact per-call allocation or a separately measured warm-only peak. Each memory measurement is one process run. Validation occurs after the memory snapshot.

Rust labels specify the requested state backend, not a guarantee of Rust derivative kernels. The selections retain explicit local frames from the accuracy table; current Rust fast paths require frame_name=None. Nonzero gravity also restricts spatial derivative kernels. These Rust-labelled measurements therefore include Python derivative fallback, including torque.

Completed: 35 / 35; failures: 0.

## Forward vs reverse AD

Speed ratio = forward median / reverse median; above 1 means reverse was faster. Peak RSS includes first call/compilation, even for JIT warm timings.

| Quantity | k | Mode | Forward ms | Reverse ms | Speed ratio | Forward peak MiB | Reverse peak MiB |
|---|---:|---|---:|---:|---:|---:|---:|
| joint torque | 0 | jit | 0.347 | 0.170 | 2.04 | 361.8 | 355.1 |
| joint torque | 0 | id_time_jit | 0.415 | 0.259 | 1.60 | 361.2 | 355.9 |
| joint torque | 0 | cmtm_jit | 0.459 | 0.365 | 1.26 | 371.5 | 373.7 |
| joint torque | 1 | jit | 0.492 | 0.216 | 2.28 | 424.7 | 432.1 |
| joint torque | 1 | id_time_jit | 0.468 | 0.218 | 2.14 | 417.7 | 424.7 |
| joint torque | 1 | cmtm_jit | 0.904 | 0.416 | 2.17 | 419.1 | 428.7 |
| joint torque | 2 | jit | 0.604 | 0.272 | 2.22 | 505.4 | 538.1 |
| joint torque | 2 | id_time_jit | 0.570 | 0.281 | 2.03 | 497.9 | 523.4 |
| joint torque | 2 | cmtm_jit | 1.411 | 0.522 | 2.70 | 472.3 | 498.9 |
| joint torque | 3 | jit | 0.841 | 0.416 | 2.02 | 606.8 | 683.1 |
| joint torque | 3 | id_time_jit | 0.699 | 0.355 | 1.97 | 644.3 | 692.2 |
| joint torque | 3 | cmtm_jit | 2.163 | 0.429 | 5.04 | 547.2 | 577.6 |
| joint torque | 4 | jit | 1.387 | 0.485 | 2.86 | 709.8 | 877.7 |
| joint torque | 4 | id_time_jit | 1.106 | 0.473 | 2.34 | 952.6 | 1004.8 |
| joint torque | 4 | cmtm_jit | 3.356 | 0.476 | 7.05 | 625.9 | 693.7 |

## All methods

| Quantity | k | Method | First ms | Warm median ms | n | Baseline MiB | Peak MiB | Growth MiB |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| joint torque | 0 | numpy_full | 29.622 | 23.643 | 5 | 184.0 | 187.7 | 3.8 |
| joint torque | 0 | forward_jit | 800.029 | 0.347 | 5 | 183.9 | 361.8 | 177.9 |
| joint torque | 0 | reverse_jit | 1046.964 | 0.170 | 5 | 184.0 | 355.1 | 171.2 |
| joint torque | 0 | id_time_forward_jit | 792.146 | 0.415 | 5 | 209.8 | 361.2 | 151.4 |
| joint torque | 0 | id_time_reverse_jit | 1048.947 | 0.259 | 5 | 209.8 | 355.9 | 146.1 |
| joint torque | 0 | cmtm_forward_jit | 799.814 | 0.459 | 5 | 250.6 | 371.5 | 121.0 |
| joint torque | 0 | cmtm_reverse_jit | 1102.565 | 0.365 | 5 | 250.8 | 373.7 | 122.9 |
| joint torque | 1 | numpy_full | 43.322 | 36.886 | 5 | 184.2 | 189.5 | 5.3 |
| joint torque | 1 | forward_jit | 1655.436 | 0.492 | 5 | 184.0 | 424.7 | 240.7 |
| joint torque | 1 | reverse_jit | 2496.575 | 0.216 | 5 | 184.0 | 432.1 | 248.1 |
| joint torque | 1 | id_time_forward_jit | 1504.795 | 0.468 | 5 | 209.9 | 417.7 | 207.8 |
| joint torque | 1 | id_time_reverse_jit | 2024.886 | 0.218 | 5 | 210.3 | 424.7 | 214.4 |
| joint torque | 1 | cmtm_forward_jit | 1444.170 | 0.904 | 5 | 250.9 | 419.1 | 168.2 |
| joint torque | 1 | cmtm_reverse_jit | 2237.470 | 0.416 | 5 | 250.5 | 428.7 | 178.2 |
| joint torque | 2 | numpy_full | 59.212 | 51.934 | 5 | 183.7 | 191.7 | 8.0 |
| joint torque | 2 | forward_jit | 3043.996 | 0.604 | 5 | 184.0 | 505.4 | 321.4 |
| joint torque | 2 | reverse_jit | 5073.364 | 0.272 | 5 | 184.2 | 538.1 | 353.9 |
| joint torque | 2 | id_time_forward_jit | 3108.310 | 0.570 | 5 | 210.0 | 497.9 | 287.9 |
| joint torque | 2 | id_time_reverse_jit | 4290.439 | 0.281 | 5 | 210.2 | 523.4 | 313.2 |
| joint torque | 2 | cmtm_forward_jit | 2435.491 | 1.411 | 5 | 251.2 | 472.3 | 221.1 |
| joint torque | 2 | cmtm_reverse_jit | 4017.519 | 0.522 | 5 | 250.7 | 498.9 | 248.1 |
| joint torque | 3 | numpy_full | 77.966 | 70.969 | 5 | 184.3 | 195.3 | 11.0 |
| joint torque | 3 | forward_jit | 4952.260 | 0.841 | 5 | 184.1 | 606.8 | 422.7 |
| joint torque | 3 | reverse_jit | 8868.324 | 0.416 | 5 | 183.7 | 683.1 | 499.4 |
| joint torque | 3 | id_time_forward_jit | 6498.018 | 0.699 | 5 | 209.9 | 644.3 | 434.4 |
| joint torque | 3 | id_time_reverse_jit | 11803.483 | 0.355 | 5 | 210.1 | 692.2 | 482.1 |
| joint torque | 3 | cmtm_forward_jit | 3513.198 | 2.163 | 5 | 250.2 | 547.2 | 296.9 |
| joint torque | 3 | cmtm_reverse_jit | 6267.448 | 0.429 | 5 | 250.5 | 577.6 | 327.0 |
| joint torque | 4 | numpy_full | 99.805 | 93.850 | 5 | 183.8 | 198.0 | 14.2 |
| joint torque | 4 | forward_jit | 7447.740 | 1.387 | 5 | 183.9 | 709.8 | 525.8 |
| joint torque | 4 | reverse_jit | 14249.259 | 0.485 | 5 | 184.2 | 877.7 | 693.6 |
| joint torque | 4 | id_time_forward_jit | 13306.242 | 1.106 | 5 | 210.0 | 952.6 | 742.6 |
| joint torque | 4 | id_time_reverse_jit | 26125.752 | 0.473 | 5 | 209.7 | 1004.8 | 795.2 |
| joint torque | 4 | cmtm_forward_jit | 5019.076 | 3.356 | 5 | 250.4 | 625.9 | 375.5 |
| joint torque | 4 | cmtm_reverse_jit | 9571.659 | 0.476 | 5 | 251.2 | 693.7 | 442.5 |

## Agreement against NumPy analytic

Errors are maximum absolute and relative Frobenius differences over the two tested inputs. NumPy analytic is a comparison reference, not an exact oracle.

| Quantity | k | Method | Max abs | Relative Frobenius |
|---|---:|---|---:|---:|
| joint torque | 0 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 0 | forward_jit | 8.527e-14 | 5.529e-16 |
| joint torque | 0 | reverse_jit | 5.684e-14 | 5.121e-16 |
| joint torque | 0 | id_time_forward_jit | 8.527e-14 | 5.529e-16 |
| joint torque | 0 | id_time_reverse_jit | 5.684e-14 | 5.121e-16 |
| joint torque | 0 | cmtm_forward_jit | 2.842e-14 | 2.976e-16 |
| joint torque | 0 | cmtm_reverse_jit | 3.908e-14 | 3.859e-16 |
| joint torque | 1 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 1 | forward_jit | 9.948e-14 | 6.286e-16 |
| joint torque | 1 | reverse_jit | 9.237e-14 | 7.149e-16 |
| joint torque | 1 | id_time_forward_jit | 9.948e-14 | 6.358e-16 |
| joint torque | 1 | id_time_reverse_jit | 1.066e-13 | 6.970e-16 |
| joint torque | 1 | cmtm_forward_jit | 1.137e-13 | 7.314e-16 |
| joint torque | 1 | cmtm_reverse_jit | 9.948e-14 | 8.018e-16 |
| joint torque | 2 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 2 | forward_jit | 3.979e-13 | 1.279e-15 |
| joint torque | 2 | reverse_jit | 3.695e-13 | 1.327e-15 |
| joint torque | 2 | id_time_forward_jit | 3.979e-13 | 1.311e-15 |
| joint torque | 2 | id_time_reverse_jit | 3.695e-13 | 1.317e-15 |
| joint torque | 2 | cmtm_forward_jit | 3.979e-13 | 1.414e-15 |
| joint torque | 2 | cmtm_reverse_jit | 3.979e-13 | 1.459e-15 |
| joint torque | 3 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 3 | forward_jit | 6.821e-13 | 1.722e-15 |
| joint torque | 3 | reverse_jit | 7.958e-13 | 1.895e-15 |
| joint torque | 3 | id_time_forward_jit | 7.958e-13 | 1.746e-15 |
| joint torque | 3 | id_time_reverse_jit | 6.821e-13 | 1.902e-15 |
| joint torque | 3 | cmtm_forward_jit | 7.248e-13 | 1.803e-15 |
| joint torque | 3 | cmtm_reverse_jit | 8.811e-13 | 1.995e-15 |
| joint torque | 4 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 4 | forward_jit | 1.876e-12 | 2.526e-15 |
| joint torque | 4 | reverse_jit | 1.904e-12 | 2.757e-15 |
| joint torque | 4 | id_time_forward_jit | 1.791e-12 | 2.743e-15 |
| joint torque | 4 | id_time_reverse_jit | 2.089e-12 | 2.952e-15 |
| joint torque | 4 | cmtm_forward_jit | 1.819e-12 | 2.703e-15 |
| joint torque | 4 | cmtm_reverse_jit | 2.302e-12 | 2.896e-15 |

## Output-value agreement

The time-differentiated quantities themselves are also checked at both inputs, outside all timing/memory measurements. These are not Jacobian errors.

| Quantity | k | Method | Max abs | Relative Frobenius |
|---|---:|---|---:|---:|
| joint torque | 0 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 0 | forward_jit | 5.684e-14 | 3.709e-16 |
| joint torque | 0 | reverse_jit | 5.684e-14 | 3.709e-16 |
| joint torque | 0 | id_time_forward_jit | 5.684e-14 | 3.709e-16 |
| joint torque | 0 | id_time_reverse_jit | 5.684e-14 | 3.709e-16 |
| joint torque | 0 | cmtm_forward_jit | 8.527e-14 | 4.627e-16 |
| joint torque | 0 | cmtm_reverse_jit | 8.527e-14 | 4.627e-16 |
| joint torque | 1 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 1 | forward_jit | 2.132e-14 | 7.842e-16 |
| joint torque | 1 | reverse_jit | 2.132e-14 | 7.842e-16 |
| joint torque | 1 | id_time_forward_jit | 3.553e-14 | 7.842e-16 |
| joint torque | 1 | id_time_reverse_jit | 3.553e-14 | 7.842e-16 |
| joint torque | 1 | cmtm_forward_jit | 1.776e-14 | 7.842e-16 |
| joint torque | 1 | cmtm_reverse_jit | 1.776e-14 | 7.842e-16 |
| joint torque | 2 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 2 | forward_jit | 3.979e-13 | 2.531e-15 |
| joint torque | 2 | reverse_jit | 3.979e-13 | 2.531e-15 |
| joint torque | 2 | id_time_forward_jit | 4.263e-13 | 2.711e-15 |
| joint torque | 2 | id_time_reverse_jit | 4.263e-13 | 2.711e-15 |
| joint torque | 2 | cmtm_forward_jit | 4.263e-13 | 2.711e-15 |
| joint torque | 2 | cmtm_reverse_jit | 4.263e-13 | 2.711e-15 |
| joint torque | 3 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 3 | forward_jit | 3.908e-13 | 1.128e-14 |
| joint torque | 3 | reverse_jit | 3.908e-13 | 1.128e-14 |
| joint torque | 3 | id_time_forward_jit | 3.126e-13 | 9.022e-15 |
| joint torque | 3 | id_time_reverse_jit | 3.126e-13 | 9.022e-15 |
| joint torque | 3 | cmtm_forward_jit | 4.121e-13 | 1.189e-14 |
| joint torque | 3 | cmtm_reverse_jit | 4.121e-13 | 1.189e-14 |
| joint torque | 4 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 4 | forward_jit | 4.832e-13 | 6.437e-15 |
| joint torque | 4 | reverse_jit | 4.832e-13 | 6.437e-15 |
| joint torque | 4 | id_time_forward_jit | 7.105e-13 | 6.568e-15 |
| joint torque | 4 | id_time_reverse_jit | 7.105e-13 | 6.568e-15 |
| joint torque | 4 | cmtm_forward_jit | 3.695e-13 | 1.182e-14 |
| joint torque | 4 | cmtm_reverse_jit | 3.695e-13 | 1.182e-14 |
