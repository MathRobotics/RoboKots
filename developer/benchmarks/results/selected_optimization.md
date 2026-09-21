# Selected spatial outputs

Mixed kinematics/dynamics.
5 warmups; 30 samples × 5 calls; medians in ms. Derivatives start from computed states; internal recomputation and Python boundary conversion are included.
Derivative primals are cached across repeated calls. State construction is separate. Same model and float64 inputs; mixed dynamics uses nonzero gravity, kinematics-only uses none. No JAX/JIT measurements: the mixed pose contract is outside the current dynamics AD API.

| Input | Operation | NumPy ms | Rust ms | NumPy / Rust |
| --- | --- | ---: | ---: | ---: |
| single | import_and_dynamics | 2.187071 | 0.008725 | 250.67x |
| single | dense | 4.328017 | 0.154062 | 28.09x |
| single | jvp | 4.131154 | 0.020413 | 202.38x |
| single | vjp | 4.243225 | 0.020800 | 204.00x |
| batch2 | import_and_dynamics | 2.827846 | 0.012267 | 230.53x |
| batch2 | dense | 5.083229 | 0.297537 | 17.08x |
| batch2 | jvp | 4.838792 | 0.034100 | 141.90x |
| batch2 | vjp | 19.563775 | 0.034312 | 570.16x |

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
| numpy/single/import_and_dynamics | 2.266796 | 2.187071 | 1.04x |
| numpy/single/dense | 4.434325 | 4.328017 | 1.02x |
| numpy/single/jvp | 4.268942 | 4.131154 | 1.03x |
| numpy/single/vjp | 4.461762 | 4.243225 | 1.05x |
| numpy/batch2/import_and_dynamics | 2.935175 | 2.827846 | 1.04x |
| numpy/batch2/dense | 8.011521 | 5.083229 | 1.58x |
| numpy/batch2/jvp | 7.671458 | 4.838792 | 1.59x |
| numpy/batch2/vjp | 25.372708 | 19.563775 | 1.30x |
| rust/single/import_and_dynamics | 0.008800 | 0.008725 | 1.01x |
| rust/single/dense | 0.155817 | 0.154062 | 1.01x |
| rust/single/jvp | 0.021387 | 0.020413 | 1.05x |
| rust/single/vjp | 0.020833 | 0.020800 | 1.00x |
| rust/batch2/import_and_dynamics | 0.012512 | 0.012267 | 1.02x |
| rust/batch2/dense | 0.297783 | 0.297537 | 1.00x |
| rust/batch2/jvp | 0.037279 | 0.034100 | 1.09x |
| rust/batch2/vjp | 0.035125 | 0.034312 | 1.02x |

Cache counters are (kinematics primal evaluations, dynamics primal evaluations, cached samples):
{'rust/single': (0, 1, 1), 'rust/batch2': (0, 2, 2)}
