# NumPy-only / Rust comparison

same-build NumPy vs Rust; all state/primal computation and input/output conversion included; model construction excluded; no cached state or Python fallback on Rust path

Seed 192; warmup 3; 9 samples × 3 calls; median milliseconds.

| Batch | Operation | NumPy ms | Rust ms | Max absolute difference |
|---|---|---:|---:|---:|
| [] | energy | 0.259139 | 0.007695 | 1.39e-17 |
| [] | energy_jvp | 1.451653 | 0.008792 | 0 |
| [] | energy_vjp | 1.410958 | 0.008514 | 5.55e-17 |
| [] | inverse_dynamics | 1.577014 | 0.006569 | 5.55e-17 |
| [] | forward_dynamics | 3.213042 | 0.008181 | 9.71e-16 |
| [6] | energy | 0.651653 | 0.024125 | 1.04e-17 |
| [6] | energy_jvp | 2.198542 | 0.025972 | 3.33e-16 |
| [6] | energy_vjp | 2.197347 | 0.025194 | 6.94e-17 |
| [6] | inverse_dynamics | 2.202444 | 0.007847 | 2.66e-15 |
| [6] | forward_dynamics | 19.094903 | 0.012417 | 6.93e-15 |

NumPy forward dynamics builds and solves a mass matrix; Rust uses ABA.
Relative Frobenius errors, first calls, and environment are in the adjacent JSON.
