# Selected spatial outputs

Mixed kinematics/dynamics.
5 warmups; 30 samples × 5 calls; medians in ms. Derivatives start from computed states; internal recomputation and Python boundary conversion are included.
Derivative primals are cached across repeated calls. State construction is separate. Same model and float64 inputs; mixed dynamics uses nonzero gravity, kinematics-only uses none. No JAX/JIT measurements: the mixed pose contract is outside the current dynamics AD API.

| Input | Operation | NumPy ms | Rust ms | NumPy / Rust |
| --- | --- | ---: | ---: | ---: |
| single | import_and_dynamics | 4.015621 | 0.016358 | 245.48x |
| single | dense | 7.829038 | 0.288275 | 27.16x |
| single | jvp | 7.472363 | 0.037354 | 200.04x |
| single | vjp | 7.717667 | 0.038275 | 201.64x |
| batch2 | import_and_dynamics | 5.223650 | 0.022104 | 236.32x |
| batch2 | dense | 9.379621 | 0.530379 | 17.68x |
| batch2 | jvp | 8.830096 | 0.061008 | 144.74x |
| batch2 | vjp | 35.717079 | 0.061379 | 581.91x |

| Accuracy reference | Max abs | Relative Frobenius |
| --- | ---: | ---: |
| numpy/single/numerical | 1.09e-07 | 7.62e-09 |
| numpy/single/jvp | 1.78e-15 | 7.99e-17 |
| numpy/single/vjp | 8.88e-16 | 1.15e-16 |
| numpy/batch2/numerical | 1.09e-07 | 5.77e-09 |
| numpy/batch2/jvp | 3.55e-15 | 1.59e-16 |
| numpy/batch2/vjp | 7.11e-15 | 2.47e-16 |
| rust/single/numerical | 1.09e-07 | 7.62e-09 |
| rust/single/jvp | 5.33e-15 | 2.25e-16 |
| rust/single/vjp | 3.55e-15 | 3.68e-16 |
| rust/batch2/numerical | 1.09e-07 | 5.77e-09 |
| rust/batch2/jvp | 1.07e-14 | 4.88e-16 |
| rust/batch2/vjp | 3.55e-15 | 1.77e-16 |
| rust/single/dense/numpy | 5.33e-15 | 4.13e-16 |
| rust/single/jvp/numpy | 4.44e-15 | 2.82e-16 |
| rust/single/vjp/numpy | 7.11e-15 | 8.12e-16 |
| rust/batch2/dense/numpy | 7.11e-15 | 4.1e-16 |
| rust/batch2/jvp/numpy | 1.55e-14 | 7.09e-16 |
| rust/batch2/vjp/numpy | 1.42e-14 | 5.81e-16 |

Selected world/pose outputs: NumPy vs Rust with optional matched baseline.

--compare checks numerical equality with a corrected same-workload baseline.
Pre-correction world/pose results are not equivalent and must not be used for
speedup claims. --kinematics-only excludes dynamics and gravity.


See the JSON for first-call timing, raw samples, outputs and environment.

## Same-workload before/after

| Workload | Before ms | After ms | Speedup |
| --- | ---: | ---: | ---: |
| numpy/single/import_and_dynamics | 4.121162 | 4.015621 | 1.03x |
| numpy/single/dense | 7.862279 | 7.829038 | 1.00x |
| numpy/single/jvp | 7.534912 | 7.472363 | 1.01x |
| numpy/single/vjp | 7.839917 | 7.717667 | 1.02x |
| numpy/batch2/import_and_dynamics | 5.205204 | 5.223650 | 1.00x |
| numpy/batch2/dense | 9.514350 | 9.379621 | 1.01x |
| numpy/batch2/jvp | 8.987592 | 8.830096 | 1.02x |
| numpy/batch2/vjp | 36.307721 | 35.717079 | 1.02x |
| rust/single/import_and_dynamics | 0.016179 | 0.016358 | 0.99x |
| rust/single/dense | 0.277225 | 0.288275 | 0.96x |
| rust/single/jvp | 0.036533 | 0.037354 | 0.98x |
| rust/single/vjp | 0.037271 | 0.038275 | 0.97x |
| rust/batch2/import_and_dynamics | 0.022596 | 0.022104 | 1.02x |
| rust/batch2/dense | 0.523225 | 0.530379 | 0.99x |
| rust/batch2/jvp | 0.060908 | 0.061008 | 1.00x |
| rust/batch2/vjp | 0.060454 | 0.061379 | 0.98x |

Cache counters are (kinematics primal evaluations, dynamics primal evaluations, cached samples):
{'rust/single': (0, 1, 1), 'rust/batch2': (0, 2, 2)}
