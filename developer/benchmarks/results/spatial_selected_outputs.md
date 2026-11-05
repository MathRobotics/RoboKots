# Corrected spatial mixed outputs

5 warmups; 30 samples × 5 calls; medians in ms. Derivatives start from computed states; internal recomputation and Python boundary conversion are included.
State construction is separate. Same model, float64 inputs and nonzero gravity. No JAX/JIT measurements: the mixed pose contract is outside the current dynamics AD API.

| Input | Operation | NumPy ms | Rust ms | NumPy / Rust |
| --- | --- | ---: | ---: | ---: |
| single | import_and_dynamics | 2.266796 | 0.008800 | 257.59x |
| single | dense | 4.434325 | 0.155817 | 28.46x |
| single | jvp | 4.268942 | 0.021387 | 199.60x |
| single | vjp | 4.461762 | 0.020833 | 214.16x |
| batch2 | import_and_dynamics | 2.935175 | 0.012512 | 234.58x |
| batch2 | dense | 8.011521 | 0.297783 | 26.90x |
| batch2 | jvp | 7.671458 | 0.037279 | 205.78x |
| batch2 | vjp | 25.372708 | 0.035125 | 722.35x |

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

Corrected world/pose mixed outputs: NumPy vs Rust (computed-state timings).

The old implementation did not implement these semantics consistently, so its
world/pose timings are not presented as equivalent-work speedups. The separate
spatial_outputs_before/after reports measure the unchanged local workload.


See the JSON for first-call timing, raw samples, outputs and environment.
