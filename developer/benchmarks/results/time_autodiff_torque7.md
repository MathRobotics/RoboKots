# Jacobian CPU time and memory

Measured: 2026-09-16T06:41:34.328597+00:00; Linux-7.0.0-31-generic-x86_64-with-glibc2.39; CPU: x86_64.

7 DOF, float64, seed 20260916, gravity [0.3, -0.4, -9.81]. Model, inputs, environment, revision and raw timings are in JSON. A single random motion is measured repeatedly, not independent trials.

Every method/case uses a fresh sequential subprocess with JAX_PLATFORMS=cpu and OMP/OPENBLAS/MKL_NUM_THREADS=1. These settings do not guarantee single-threaded XLA. Process startup/import/model construction and common CPU runtime initialization are excluded. Cached analytic cases additionally exclude initial state preparation; full cases include motion import and state calculation without dictionary export. Numerical uses central differences, eps=1e-8. AD includes runtime input conversion, synchronized execution, and NumPy output conversion. Unprefixed forward/reverse AD computes analytic time derivatives directly on ordinary derivative coefficient series; eager may compute unused dynamics work. Methods prefixed time_ instead obtain velocity from FK directional JVPs, momentum rate from another JVP, and higher output derivatives from nested total-time JVPs before outer jacfwd/jacrev. Methods prefixed id_time_ start from ordinary ID(q,qdot,qddot), always motion order 3, and use total-time JVPs only for the higher torque derivatives. Methods prefixed cmtm_ instead construct explicit spatial lower block-Toeplitz CMTMs and use factorial-normalized coefficient vectors; only the outer motion Jacobian uses AD. The coefficient-series and explicit-CMTM paths are implementation representations of the same high-order algebra, not separate mathematical differentiation methods. Both use recurrences to construct transform coefficients. No trajectory is constructed: D_t f(x)=JVP(f,x,shift(x)) with x=(q,qdot,...); repeated AD differentiates the state-dependent direction too.

First call includes lazy initialization (and tracing/compilation for JIT). Warm medians exclude the first call and configured warmups. No JIT function closes over motion. A second seeded input is checked against NumPy analytic results outside measurement.

Peak RSS is the OS process-lifetime high-water mark, not live array size or device memory. It includes Python, libraries, compiler and allocator caches. Growth subtracts the baseline high-water mark after setup; it is NOT exact per-call allocation or a separately measured warm-only peak. Each memory measurement is one process run. Validation occurs after the memory snapshot.

Rust labels specify the requested state backend, not a guarantee of Rust derivative kernels. The selections retain explicit local frames from the accuracy table; current Rust fast paths require frame_name=None. Nonzero gravity also restricts spatial derivative kernels. These Rust-labelled measurements therefore include Python derivative fallback, including torque.

Completed: 24 / 25; failures: 1.

## Forward vs reverse AD

Speed ratio = forward median / reverse median; above 1 means reverse was faster. Peak RSS includes first call/compilation, even for JIT warm timings.

| Quantity | k | Mode | Forward ms | Reverse ms | Speed ratio | Forward peak MiB | Reverse peak MiB |
|---|---:|---|---:|---:|---:|---:|---:|
| joint torque | 0 | jit | 0.380 | 0.237 | 1.60 | 357.3 | 356.3 |
| joint torque | 0 | time_jit | 0.367 | 0.239 | 1.54 | 389.0 | 409.1 |
| joint torque | 1 | jit | 0.436 | 0.339 | 1.29 | 431.4 | 431.0 |
| joint torque | 1 | time_jit | 0.381 | 0.317 | 1.20 | 464.3 | 575.4 |
| joint torque | 2 | jit | 0.508 | 0.274 | 1.85 | 506.5 | 535.0 |
| joint torque | 2 | time_jit | 0.497 | 0.831 | 0.60 | 615.8 | 1024.0 |
| joint torque | 3 | jit | 0.686 | 0.364 | 1.89 | 609.7 | 677.1 |
| joint torque | 3 | time_jit | 0.892 | 0.870 | 1.03 | 920.4 | 2739.4 |
| joint torque | 4 | jit | 1.314 | 0.480 | 2.74 | 711.1 | 868.7 |

## All methods

| Quantity | k | Method | First ms | Warm median ms | n | Baseline MiB | Peak MiB | Growth MiB |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| joint torque | 0 | numpy_full | 29.958 | 23.772 | 5 | 184.3 | 188.1 | 3.7 |
| joint torque | 0 | forward_jit | 792.139 | 0.380 | 5 | 184.1 | 357.3 | 173.2 |
| joint torque | 0 | reverse_jit | 1043.287 | 0.237 | 5 | 184.1 | 356.3 | 172.2 |
| joint torque | 0 | time_forward_jit | 1169.982 | 0.367 | 5 | 233.9 | 389.0 | 155.1 |
| joint torque | 0 | time_reverse_jit | 1940.715 | 0.239 | 5 | 233.8 | 409.1 | 175.2 |
| joint torque | 1 | numpy_full | 42.702 | 36.540 | 5 | 183.7 | 189.5 | 5.8 |
| joint torque | 1 | forward_jit | 1626.749 | 0.436 | 5 | 184.4 | 431.4 | 247.0 |
| joint torque | 1 | reverse_jit | 2471.650 | 0.339 | 5 | 184.0 | 431.0 | 246.9 |
| joint torque | 1 | time_forward_jit | 2479.290 | 0.381 | 5 | 233.8 | 464.3 | 230.5 |
| joint torque | 1 | time_reverse_jit | 7111.187 | 0.317 | 5 | 233.9 | 575.4 | 341.5 |
| joint torque | 2 | numpy_full | 58.132 | 51.530 | 5 | 184.0 | 191.8 | 7.8 |
| joint torque | 2 | forward_jit | 3036.111 | 0.508 | 5 | 184.4 | 506.5 | 322.2 |
| joint torque | 2 | reverse_jit | 5044.349 | 0.274 | 5 | 184.0 | 535.0 | 351.0 |
| joint torque | 2 | time_forward_jit | 5851.670 | 0.497 | 5 | 234.2 | 615.8 | 381.7 |
| joint torque | 2 | time_reverse_jit | 31051.305 | 0.831 | 5 | 233.8 | 1024.0 | 790.2 |
| joint torque | 3 | numpy_full | 76.227 | 70.399 | 5 | 183.9 | 195.0 | 11.0 |
| joint torque | 3 | forward_jit | 4913.756 | 0.686 | 5 | 184.2 | 609.7 | 425.6 |
| joint torque | 3 | reverse_jit | 8757.045 | 0.364 | 5 | 183.9 | 677.1 | 493.2 |
| joint torque | 3 | time_forward_jit | 16989.296 | 0.892 | 5 | 233.5 | 920.4 | 686.9 |
| joint torque | 3 | time_reverse_jit | 156028.711 | 0.870 | 5 | 234.1 | 2739.4 | 2505.3 |
| joint torque | 4 | numpy_full | 100.427 | 94.512 | 5 | 183.9 | 197.9 | 14.0 |
| joint torque | 4 | forward_jit | 7426.223 | 1.314 | 5 | 183.8 | 711.1 | 527.3 |
| joint torque | 4 | reverse_jit | 14114.390 | 0.480 | 5 | 183.9 | 868.7 | 684.8 |
| joint torque | 4 | time_forward_jit | 48388.549 | 1.025 | 5 | 233.9 | 1835.0 | 1601.0 |

## Agreement against NumPy analytic

Errors are maximum absolute and relative Frobenius differences over the two tested inputs. NumPy analytic is a comparison reference, not an exact oracle.

| Quantity | k | Method | Max abs | Relative Frobenius |
|---|---:|---|---:|---:|
| joint torque | 0 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 0 | forward_jit | 8.527e-14 | 5.529e-16 |
| joint torque | 0 | reverse_jit | 5.684e-14 | 5.121e-16 |
| joint torque | 0 | time_forward_jit | 4.974e-14 | 3.748e-16 |
| joint torque | 0 | time_reverse_jit | 8.527e-14 | 6.199e-16 |
| joint torque | 1 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 1 | forward_jit | 9.948e-14 | 6.286e-16 |
| joint torque | 1 | reverse_jit | 9.237e-14 | 7.149e-16 |
| joint torque | 1 | time_forward_jit | 8.527e-14 | 6.023e-16 |
| joint torque | 1 | time_reverse_jit | 9.237e-14 | 6.664e-16 |
| joint torque | 2 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 2 | forward_jit | 3.979e-13 | 1.279e-15 |
| joint torque | 2 | reverse_jit | 3.695e-13 | 1.327e-15 |
| joint torque | 2 | time_forward_jit | 3.411e-13 | 1.299e-15 |
| joint torque | 2 | time_reverse_jit | 3.553e-13 | 1.401e-15 |
| joint torque | 3 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 3 | forward_jit | 6.821e-13 | 1.722e-15 |
| joint torque | 3 | reverse_jit | 7.958e-13 | 1.895e-15 |
| joint torque | 3 | time_forward_jit | 9.095e-13 | 2.033e-15 |
| joint torque | 3 | time_reverse_jit | 8.384e-13 | 2.151e-15 |
| joint torque | 4 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 4 | forward_jit | 1.876e-12 | 2.526e-15 |
| joint torque | 4 | reverse_jit | 1.904e-12 | 2.757e-15 |
| joint torque | 4 | time_forward_jit | 2.132e-12 | 3.180e-15 |

## Failures

```json
[
  {
    "row": 7,
    "k": 4,
    "method": "time_reverse_jit",
    "error": "Command '['/home/ishigaki/src/RoboKots/.venv/bin/python', '-m', 'developer.benchmarks.jacobian_resources', '--worker', '--dof', '7', '--seed', '20260916', '--row', '7', '--k', '4', '--method', 'time_reverse_jit', '--repeats', '5', '--numerical-repeats', '1', '--warmup', '1', '--check-values']' timed out after 600.0 seconds",
    "stderr": "None"
  }
]
```

## Output-value agreement

The time-differentiated quantities themselves are also checked at both inputs, outside all timing/memory measurements. These are not Jacobian errors.

| Quantity | k | Method | Max abs | Relative Frobenius |
|---|---:|---|---:|---:|
| joint torque | 0 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 0 | forward_jit | 5.684e-14 | 3.709e-16 |
| joint torque | 0 | reverse_jit | 5.684e-14 | 3.709e-16 |
| joint torque | 0 | time_forward_jit | 8.527e-14 | 4.627e-16 |
| joint torque | 0 | time_reverse_jit | 8.527e-14 | 4.627e-16 |
| joint torque | 1 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 1 | forward_jit | 2.132e-14 | 7.842e-16 |
| joint torque | 1 | reverse_jit | 2.132e-14 | 7.842e-16 |
| joint torque | 1 | time_forward_jit | 2.132e-14 | 3.502e-16 |
| joint torque | 1 | time_reverse_jit | 2.132e-14 | 3.502e-16 |
| joint torque | 2 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 2 | forward_jit | 3.979e-13 | 2.531e-15 |
| joint torque | 2 | reverse_jit | 3.979e-13 | 2.531e-15 |
| joint torque | 2 | time_forward_jit | 4.263e-13 | 2.711e-15 |
| joint torque | 2 | time_reverse_jit | 4.263e-13 | 2.711e-15 |
| joint torque | 3 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 3 | forward_jit | 3.908e-13 | 1.128e-14 |
| joint torque | 3 | reverse_jit | 3.908e-13 | 1.128e-14 |
| joint torque | 3 | time_forward_jit | 3.268e-13 | 9.432e-15 |
| joint torque | 3 | time_reverse_jit | 3.268e-13 | 9.432e-15 |
| joint torque | 4 | numpy_full | 0.000e+00 | 0.000e+00 |
| joint torque | 4 | forward_jit | 4.832e-13 | 6.437e-15 |
| joint torque | 4 | reverse_jit | 4.832e-13 | 6.437e-15 |
| joint torque | 4 | time_forward_jit | 4.832e-13 | 7.619e-15 |
