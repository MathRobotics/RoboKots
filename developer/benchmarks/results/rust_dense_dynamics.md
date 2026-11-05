# Rust dense dynamics Jacobian comparison

Measured UTC: 2026-09-30T09:03:32.951583+00:00

Baseline: `932a426422b165aa395875f3c6af3ef1c0d23070` (Python dispatch + release Rust extension). Optimized: current working-tree Python dispatch + installed release extension.

Synthetic serial revolute chains with alternating axes, rotated origins and a fixed tool. All active-joint outputs; local unless specified. Momentum/force each have 6 rows per joint; torque has 1. Dense Jacobians only, float64, gravity [0.2, -0.3, -9.81], seed 20260930 + DOF. Motion order is derivative+2 for momentum, derivative+3 for force/torque; no padding.

3 fresh-process rounds, alternating variant order. Per case and scope: one separate first call, 5 warmups, 15 samples × 3 calls. Reported medians pool all rounds. Milliseconds per call, including the entire batch. Input/output boundary costs included; model construction and imports excluded. No JAX/JIT or numerical differentiation is timed. Python derivative fallback is blocked.

state_ready starts with computed states and reuses warmed derivative workspaces. including_state alternates two motions and includes import, dynamics without dictionary materialization, and the Jacobian. First calls are API calls, not process startup/compilation. A compiled extension is reused throughout each process.

Environment: AMD Ryzen 9 7940HS w/ Radeon 780M Graphics; Linux-7.0.0-34-generic-x86_64-with-glibc2.39; Python 3.13.1; NumPy 2.4.6; rustc 1.93.0 (254b59607 2026-01-19). BLAS/OpenMP thread counts are set to one. Environment and raw samples are in the JSON.

| DOF | Output | Derivative | Frame | Batch | Shape | Ready old ms | Ready new ms | Speedup | Full old ms | Full new ms | Speedup |
|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 7 | momentum | 0 | local | [] | 42×14 | 0.1100 | 0.0528 | 2.08× | 0.1267 | 0.0697 | 1.82× |
| 7 | momentum | 1 | local | [] | 42×21 | 0.2735 | 0.1446 | 1.89× | 0.2919 | 0.1634 | 1.79× |
| 7 | momentum | 2 | local | [] | 42×28 | 0.5396 | 0.3097 | 1.74× | 0.5658 | 0.3303 | 1.71× |
| 7 | momentum | 3 | local | [] | 42×35 | 0.9523 | 0.5707 | 1.67× | 0.9752 | 0.5963 | 1.64× |
| 7 | momentum | 4 | local | [] | 42×42 | 1.5059 | 0.9174 | 1.64× | 1.5318 | 0.9458 | 1.62× |
| 7 | force | 0 | local | [] | 42×21 | 0.2769 | 0.1464 | 1.89× | 0.2950 | 0.1655 | 1.78× |
| 7 | force | 1 | local | [] | 42×28 | 0.5399 | 0.3099 | 1.74× | 0.5601 | 0.3341 | 1.68× |
| 7 | force | 2 | local | [] | 42×35 | 0.9510 | 0.5726 | 1.66× | 0.9758 | 0.6000 | 1.63× |
| 7 | force | 3 | local | [] | 42×42 | 1.4998 | 0.9370 | 1.60× | 1.5275 | 0.9675 | 1.58× |
| 7 | force | 4 | local | [] | 42×49 | 2.2741 | 1.4284 | 1.59× | 2.3024 | 1.4566 | 1.58× |
| 7 | torque | 0 | local | [] | 7×21 | 0.0285 | 0.0276 | 1.03× | 0.0444 | 0.0438 | 1.01× |
| 7 | torque | 1 | local | [] | 7×28 | 0.5614 | 0.3139 | 1.79× | 0.5844 | 0.3348 | 1.75× |
| 7 | torque | 2 | local | [] | 7×35 | 0.9740 | 0.5767 | 1.69× | 0.9981 | 0.6014 | 1.66× |
| 7 | torque | 3 | local | [] | 7×42 | 1.5186 | 0.9300 | 1.63× | 1.5469 | 0.9580 | 1.61× |
| 7 | torque | 4 | local | [] | 7×49 | 2.3082 | 1.4305 | 1.61× | 2.3367 | 1.4634 | 1.60× |
| 16 | momentum | 0 | local | [] | 96×32 | 0.4711 | 0.1961 | 2.40× | 0.4890 | 0.2162 | 2.26× |
| 16 | momentum | 1 | local | [] | 96×48 | 1.2472 | 0.6564 | 1.90× | 1.2789 | 0.6799 | 1.88× |
| 16 | momentum | 2 | local | [] | 96×64 | 2.6069 | 1.4781 | 1.76× | 2.6396 | 1.5056 | 1.75× |
| 16 | momentum | 3 | local | [] | 96×80 | 4.6625 | 2.8206 | 1.65× | 4.6848 | 2.8512 | 1.64× |
| 16 | momentum | 4 | local | [] | 96×96 | 7.4382 | 4.6070 | 1.61× | 7.4816 | 4.6489 | 1.61× |
| 16 | force | 0 | local | [] | 96×48 | 1.2648 | 0.6640 | 1.90× | 1.2905 | 0.6897 | 1.87× |
| 16 | force | 1 | local | [] | 96×64 | 2.6669 | 1.4869 | 1.79× | 2.6931 | 1.5148 | 1.78× |
| 16 | force | 2 | local | [] | 96×80 | 4.6821 | 2.8073 | 1.67× | 4.7003 | 2.8456 | 1.65× |
| 16 | force | 3 | local | [] | 96×96 | 7.4187 | 4.6441 | 1.60× | 7.4817 | 4.6539 | 1.61× |
| 16 | force | 4 | local | [] | 96×112 | 11.3176 | 7.1046 | 1.59× | 11.3988 | 7.1157 | 1.60× |
| 16 | torque | 0 | local | [] | 16×48 | 0.0760 | 0.0760 | 1.00× | 0.0958 | 0.0963 | 1.00× |
| 16 | torque | 1 | local | [] | 16×64 | 2.7051 | 1.4979 | 1.81× | 2.7415 | 1.5239 | 1.80× |
| 16 | torque | 2 | local | [] | 16×80 | 4.7176 | 2.8053 | 1.68× | 4.7543 | 2.8372 | 1.68× |
| 16 | torque | 3 | local | [] | 16×96 | 7.4785 | 4.6135 | 1.62× | 7.5323 | 4.6505 | 1.62× |
| 16 | torque | 4 | local | [] | 16×112 | 11.8446 | 7.1225 | 1.66× | 11.8602 | 7.1674 | 1.65× |
| 7 | momentum | 4 | world | [] | 42×42 | 1.8543 | 1.2778 | 1.45× | 1.8838 | 1.3072 | 1.44× |
| 7 | force | 4 | world | [] | 42×49 | 2.6809 | 1.8310 | 1.46× | 2.7117 | 1.8658 | 1.45× |
| 7 | torque | 4 | local | [2, 1] | 2×1×7×49 | 4.5294 | 2.8402 | 1.59× | 4.5781 | 2.8906 | 1.58× |

Maximum absolute difference across both motions/all rounds: 8.81073e-12.
Maximum relative Frobenius difference: 1.53917e-14.

Baseline outputs are a comparison reference, not an exact oracle. Independent NumPy/finite-difference regressions are separate from this timing run.

Reproduce: `/home/ishigaki/src/RoboKots/.venv/bin/python -u -m developer.benchmarks.rust_dense_dynamics_compare`
