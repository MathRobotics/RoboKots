# Time AD starting from FK vs ordinary ID

Preserved FK run: 2026-09-16T06:41:34.328597+00:00. New ID/coefficient run: 2026-09-16T09:30:08.908480+00:00.

The previous FK implementation and results are preserved, not rerun. Model, motions, gravity, seed, dtype/software/device environment, warmups and repeat counts were checked for equality. These are separate local runs, not simultaneous measurements.

All cases use the first actuated joint torque of the same 7-DOF rigid serial arm. k denotes ordinary torque time-derivative order. At k=4 the outer Jacobian is 1x49. All methods use dense outer AD. For the time-AD formulations, inner time AD is forward JVP in both outer modes.

Derivative coefficients + AD computes analytic high-order time derivatives directly on coefficient series before outer AD. This is an implementation representation of the same high-order algebra as explicit CMTM matrices, not a separate mathematical differentiation method. Both representations use coefficient recurrences. FK + time AD obtains velocity/momentum rate via AD from FK, then differentiates torque in time. Ordinary ID + time AD always evaluates ID(q,qdot,qddot) at motion order 3, then applies k total-time JVPs before outer AD. It does not reuse higher-order torque recurrences.

First time includes JIT tracing, compilation and execution. Warm time is the median of 5 calls after the first call and 1 warmups, including input conversion, synchronization and NumPy output. Peak RSS includes compilation and runtime caches, excludes post-measurement validation, and is not per-call memory. Missing results are never substituted by another method.

| k | Outer AD | Formulation | First s | Warm ms | Peak MiB |
|---:|---|---|---:|---:|---:|
| 0 | forward | Derivative coefficients + AD (new run) | 0.795 | 0.332 | 356.3 |
| 0 | forward | FK + time AD (previous run) | 1.170 | 0.367 | 389.0 |
| 0 | forward | Ordinary ID + time AD (new run) | 0.774 | 0.263 | 361.3 |
| 0 | reverse | Derivative coefficients + AD (new run) | 1.042 | 0.177 | 355.1 |
| 0 | reverse | FK + time AD (previous run) | 1.941 | 0.239 | 409.1 |
| 0 | reverse | Ordinary ID + time AD (new run) | 1.036 | 0.168 | 357.6 |
| 1 | forward | Derivative coefficients + AD (new run) | 1.634 | 0.371 | 424.8 |
| 1 | forward | FK + time AD (previous run) | 2.479 | 0.381 | 464.3 |
| 1 | forward | Ordinary ID + time AD (new run) | 1.496 | 0.440 | 420.4 |
| 1 | reverse | Derivative coefficients + AD (new run) | 2.436 | 0.215 | 433.6 |
| 1 | reverse | FK + time AD (previous run) | 7.111 | 0.317 | 575.4 |
| 1 | reverse | Ordinary ID + time AD (new run) | 2.028 | 0.227 | 421.1 |
| 2 | forward | Derivative coefficients + AD (new run) | 3.011 | 0.546 | 506.2 |
| 2 | forward | FK + time AD (previous run) | 5.852 | 0.497 | 615.8 |
| 2 | forward | Ordinary ID + time AD (new run) | 3.121 | 0.606 | 499.5 |
| 2 | reverse | Derivative coefficients + AD (new run) | 5.084 | 0.272 | 536.5 |
| 2 | reverse | FK + time AD (previous run) | 31.051 | 0.831 | 1024.0 |
| 2 | reverse | Ordinary ID + time AD (new run) | 4.217 | 0.274 | 517.2 |
| 3 | forward | Derivative coefficients + AD (new run) | 4.913 | 0.820 | 606.8 |
| 3 | forward | FK + time AD (previous run) | 16.989 | 0.892 | 920.4 |
| 3 | forward | Ordinary ID + time AD (new run) | 6.438 | 0.830 | 638.7 |
| 3 | reverse | Derivative coefficients + AD (new run) | 8.745 | 0.413 | 680.7 |
| 3 | reverse | FK + time AD (previous run) | 156.029 | 0.870 | 2739.4 |
| 3 | reverse | Ordinary ID + time AD (new run) | 11.437 | 0.361 | 694.2 |
| 4 | forward | Derivative coefficients + AD (new run) | 7.465 | 1.259 | 709.1 |
| 4 | forward | FK + time AD (previous run) | 48.389 | 1.025 | 1835.0 |
| 4 | forward | Ordinary ID + time AD (new run) | 13.267 | 1.241 | 946.6 |
| 4 | reverse | Derivative coefficients + AD (new run) | 14.086 | 0.478 | 872.9 |
| 4 | reverse | FK + time AD (previous run) | timeout | — | — |
| 4 | reverse | Ordinary ID + time AD (new run) | 25.680 | 0.600 | 1022.8 |

## Agreement

Each run checks two inputs against NumPy analytic values and Jacobians (see source reports). The following directly compares the new ID formulation with the preserved FK formulation. Relative Frobenius errors and full failure details are in the JSON.

| k | Outer AD | Max value difference | Max Jacobian difference |
|---:|---|---:|---:|
| 0 | forward | 2.842e-14 | 5.684e-14 |
| 0 | reverse | 2.842e-14 | 8.527e-14 |
| 1 | forward | 1.421e-14 | 4.263e-14 |
| 1 | reverse | 1.421e-14 | 8.527e-14 |
| 2 | forward | 0.000e+00 | 5.684e-14 |
| 2 | reverse | 0.000e+00 | 8.527e-14 |
| 3 | forward | 2.842e-14 | 1.421e-13 |
| 3 | reverse | 2.842e-14 | 1.990e-13 |
| 4 | forward | 2.274e-13 | 7.958e-13 |
