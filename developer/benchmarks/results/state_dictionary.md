# State dictionary separation timing

This historical measurement predates centralizing export in `state_io.dictionary`.
It describes the source snapshot below, not the final refactored implementation.

Baseline: `6ec3bfb456acf2b5d05aef1f8f0b2803b88c4c0d`; current uncommitted source hash: `d6c45959b016930bac709eb71107e1380c2fea7995dc304e67896a0cb08cca96`.

Environment: AMD Ryzen 9 7940HS w/ Radeon 780M Graphics; Python 3.13.1; NumPy 2.4.6; mathrobo 0.0.4.
Measured: 2026-09-14T16:29:09+0900. Same Rust extension and virtualenv; one BLAS/OpenMP thread.

Order 5, float64, seed 71, world gravity [0.2, -0.3, -9.81]. Sample 3-DOF arm and generated 16-DOF humanoid tree.
ABBA sequential workers; each case has one first call, 3 warmups and 15 measured calls per worker.
First-call times (after model/state initialization) and individual samples are in the JSON. Module-import/model-construction costs are excluded.

Default operations include motion import, state calculation and selected value retrieval. Before defaults export a dictionary; after defaults do not.
`dynamics_no_dict` explicitly disables export on both revisions. `dynamics_export` includes a full dictionary export on both, including new snapshot copies.
`jacobian_cached` measures the dense torque_diff2 Jacobian with state already computed. Rust dispatch may use existing NumPy fallbacks.
No JAX AD/JIT timing or numerical differentiation timing is included. NumPy/Rust outputs are synchronous; Python/NumPy conversion costs are included.

| DOF | Batch | Backend | Operation | Before ms | After ms | Before/after | Max abs error | Rel. Frobenius error |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 3 | 1 | numpy | kinematics_default | 0.9164 | 0.8532 | 1.07 | 0.00e+00 | 0.00e+00 |
| 3 | 1 | numpy | dynamics_default | 3.8569 | 3.5845 | 1.08 | 0.00e+00 | 0.00e+00 |
| 3 | 1 | numpy | dynamics_no_dict | 3.7236 | 3.5820 | 1.04 | 0.00e+00 | 0.00e+00 |
| 3 | 1 | numpy | dynamics_export | 3.6970 | 3.6694 | 1.01 | 0.00e+00 | 0.00e+00 |
| 3 | 1 | numpy | jacobian_cached | 6.4845 | 6.4552 | 1.00 | 0.00e+00 | 0.00e+00 |
| 3 | 1 | rust | kinematics_default | 0.0841 | 0.0053 | 15.88 | 0.00e+00 | 0.00e+00 |
| 3 | 1 | rust | dynamics_default | 0.2896 | 0.0297 | 9.75 | 0.00e+00 | 0.00e+00 |
| 3 | 1 | rust | dynamics_no_dict | 0.0292 | 0.0287 | 1.02 | 0.00e+00 | 0.00e+00 |
| 3 | 1 | rust | dynamics_export | 0.2727 | 0.3028 | 0.90 | 0.00e+00 | 0.00e+00 |
| 3 | 1 | rust | jacobian_cached | 0.2114 | 0.2096 | 1.01 | 0.00e+00 | 0.00e+00 |
| 3 | 8 | numpy | kinematics_default | 1.7496 | 1.6963 | 1.03 | 0.00e+00 | 0.00e+00 |
| 3 | 8 | numpy | dynamics_default | 31.3130 | 30.3738 | 1.03 | 0.00e+00 | 0.00e+00 |
| 3 | 8 | numpy | dynamics_no_dict | 30.3981 | 30.2574 | 1.00 | 0.00e+00 | 0.00e+00 |
| 3 | 8 | numpy | dynamics_export | 31.1002 | 31.1854 | 1.00 | 0.00e+00 | 0.00e+00 |
| 3 | 8 | numpy | jacobian_cached | 51.8665 | 51.8205 | 1.00 | 0.00e+00 | 0.00e+00 |
| 3 | 8 | rust | kinematics_default | 0.1206 | 0.0172 | 7.03 | 0.00e+00 | 0.00e+00 |
| 3 | 8 | rust | dynamics_default | 0.3827 | 0.0540 | 7.09 | 0.00e+00 | 0.00e+00 |
| 3 | 8 | rust | dynamics_no_dict | 0.0535 | 0.0530 | 1.01 | 0.00e+00 | 0.00e+00 |
| 3 | 8 | rust | dynamics_export | 0.3619 | 0.4043 | 0.90 | 0.00e+00 | 0.00e+00 |
| 3 | 8 | rust | jacobian_cached | 1.5367 | 1.5267 | 1.01 | 0.00e+00 | 0.00e+00 |
| 16 | 1 | numpy | kinematics_default | 3.7137 | 3.5109 | 1.06 | 0.00e+00 | 0.00e+00 |
| 16 | 1 | numpy | dynamics_default | 16.2092 | 15.8323 | 1.02 | 0.00e+00 | 0.00e+00 |
| 16 | 1 | numpy | dynamics_no_dict | 15.7363 | 15.7155 | 1.00 | 0.00e+00 | 0.00e+00 |
| 16 | 1 | numpy | dynamics_export | 16.0330 | 16.2172 | 0.99 | 0.00e+00 | 0.00e+00 |
| 16 | 1 | numpy | jacobian_cached | 59.3726 | 58.7282 | 1.01 | 0.00e+00 | 0.00e+00 |
| 16 | 1 | rust | kinematics_default | 0.3337 | 0.0098 | 34.12 | 0.00e+00 | 0.00e+00 |
| 16 | 1 | rust | dynamics_default | 1.1251 | 0.0812 | 13.86 | 0.00e+00 | 0.00e+00 |
| 16 | 1 | rust | dynamics_no_dict | 0.0811 | 0.0813 | 1.00 | 0.00e+00 | 0.00e+00 |
| 16 | 1 | rust | dynamics_export | 1.0561 | 1.1615 | 0.91 | 0.00e+00 | 0.00e+00 |
| 16 | 1 | rust | jacobian_cached | 4.5606 | 4.5948 | 0.99 | 0.00e+00 | 0.00e+00 |
| 16 | 8 | numpy | kinematics_default | 7.5999 | 7.4136 | 1.03 | 0.00e+00 | 0.00e+00 |
| 16 | 8 | numpy | dynamics_default | 138.0560 | 135.8260 | 1.02 | 0.00e+00 | 0.00e+00 |
| 16 | 8 | numpy | dynamics_no_dict | 134.4858 | 134.4382 | 1.00 | 0.00e+00 | 0.00e+00 |
| 16 | 8 | numpy | dynamics_export | 137.9753 | 138.7862 | 0.99 | 0.00e+00 | 0.00e+00 |
| 16 | 8 | numpy | jacobian_cached | 473.6156 | 471.9097 | 1.00 | 0.00e+00 | 0.00e+00 |
| 16 | 8 | rust | kinematics_default | 0.4512 | 0.0528 | 8.54 | 0.00e+00 | 0.00e+00 |
| 16 | 8 | rust | dynamics_default | 1.4501 | 0.1837 | 7.89 | 0.00e+00 | 0.00e+00 |
| 16 | 8 | rust | dynamics_no_dict | 0.1824 | 0.1815 | 1.01 | 0.00e+00 | 0.00e+00 |
| 16 | 8 | rust | dynamics_export | 1.3889 | 1.5320 | 0.91 | 0.00e+00 | 0.00e+00 |
| 16 | 8 | rust | jacobian_cached | 35.9343 | 38.1480 | 0.94 | 0.00e+00 | 0.00e+00 |

Ratios near 1 should be interpreted as similar performance, not a demonstrated improvement/regression.

## Interpretation

Default Rust calls benefit strongly from skipping the formerly automatic export.
The explicit dictionary-free timings show essentially unchanged kernel/API costs.
Full Rust export is about 10–12% slower here and now includes independent snapshot
copies. All compared output arrays have zero maximum absolute and relative
Frobenius differences.

The 16-DOF Rust batch-8 Jacobian aggregate is 6.2% slower, but per-worker medians
were 35.95/35.86 ms before and 41.28/35.74 ms after. The change was not consistent
between workers; this measurement alone does not establish a sustained regression.

A separate untimed check of the current 16-DOF routes confirmed that NumPy
batch-8 dynamics uses the existing per-sample `StateBatch` fallback, while Rust
uses `RustBatchOutwardState`. Both scalar and batched Rust torque_diff2 Jacobians
completed with the generic NumPy Jacobian fallback disabled. This does not imply
that every other Rust output combination avoids fallback.

Reproduce: `.venv/bin/python -m developer.benchmarks.state_dictionary_compare`.
