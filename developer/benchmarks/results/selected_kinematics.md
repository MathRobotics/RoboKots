# Selected spatial outputs

Kinematics only.
5 warmups; 30 samples × 5 calls; medians in ms. Derivatives start from computed states; internal recomputation and Python boundary conversion are included.
Derivative primals are cached across repeated calls. State construction is separate. Same model and float64 inputs; mixed dynamics uses nonzero gravity, kinematics-only uses none. No JAX/JIT measurements: the mixed pose contract is outside the current dynamics AD API.

| Input | Operation | NumPy ms | Rust ms | NumPy / Rust |
| --- | --- | ---: | ---: | ---: |
| single | import_and_kinematics | 0.408283 | 0.003483 | 117.21x |
| single | dense | 0.259475 | 0.075304 | 3.45x |
| single | jvp | 0.253479 | 0.011792 | 21.50x |
| single | vjp | 0.255850 | 0.013121 | 19.50x |
| batch2 | import_and_kinematics | 0.795217 | 0.005396 | 147.37x |
| batch2 | dense | 0.324550 | 0.134879 | 2.41x |
| batch2 | jvp | 0.320192 | 0.018429 | 17.37x |
| batch2 | vjp | 0.322721 | 0.019846 | 16.26x |

| Accuracy reference | Max abs | Relative Frobenius |
| --- | ---: | ---: |
| numpy/single/numerical | 3.47e-09 | 4.07e-09 |
| numpy/single/jvp | 4.44e-16 | 1.9e-16 |
| numpy/single/vjp | 2.22e-16 | 1.39e-16 |
| numpy/batch2/numerical | 6.41e-09 | 4.38e-09 |
| numpy/batch2/jvp | 3.89e-16 | 1.84e-16 |
| numpy/batch2/vjp | 1.11e-16 | 8.59e-17 |
| rust/single/numerical | 3.47e-09 | 4.07e-09 |
| rust/single/jvp | 2.22e-16 | 1.36e-16 |
| rust/single/vjp | 5e-16 | 2.95e-16 |
| rust/batch2/numerical | 6.41e-09 | 4.38e-09 |
| rust/batch2/jvp | 5.55e-16 | 3.13e-16 |
| rust/batch2/vjp | 6.16e-16 | 3.63e-16 |
| rust/single/dense/numpy | 6.66e-16 | 5.67e-16 |
| rust/single/jvp/numpy | 9.99e-16 | 5.95e-16 |
| rust/single/vjp/numpy | 6.11e-16 | 4.05e-16 |
| rust/batch2/dense/numpy | 8.88e-16 | 5.63e-16 |
| rust/batch2/jvp/numpy | 8.88e-16 | 3.93e-16 |
| rust/batch2/vjp/numpy | 1.11e-15 | 5.21e-16 |

Selected world/pose outputs: NumPy vs Rust with optional matched baseline.

--compare checks numerical equality with a corrected same-workload baseline.
Pre-correction world/pose results are not equivalent and must not be used for
speedup claims. --kinematics-only excludes dynamics and gravity.


See the JSON for first-call timing, raw samples, outputs and environment.

Cache counters are (kinematics primal evaluations, dynamics primal evaluations, cached samples):
{'rust/single': (1, 0, 1), 'rust/batch2': (2, 0, 2)}
