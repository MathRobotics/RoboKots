# NumPy / Rust analytical Jacobian comparison

Same-build analytical backends; no JAX or numerical differentiation. Rust NumPy fallbacks and NumPy native dispatch blocked. Input/output boundary costs included. Model construction excluded. state_ready reuses current state and warmed derivative workspaces; including_state alternates two motions and includes import + state calculation + derivative. First_ms is first timed call, not cold compilation. Milliseconds per entire call (batch time is for six samples).

Model: synthetic serial 7-DOF revolute chain with fixed tool (7 DOF). Float64; seed 20260923; gravity [0.2, -0.3, -9.81].
Warmup 5; 15 samples × 3 calls; median milliseconds.

| Family | Order | Batch | Scope | Operation | NumPy ms | Rust ms | Speedup |
|---|---:|---|---|---|---:|---:|---:|
| kinematics | 3 | [] | state_ready | dense | 0.296194 | 0.008556 | 34.6× |
| kinematics | 3 | [] | state_ready | jvp | 0.287208 | 0.006486 | 44.3× |
| kinematics | 3 | [] | state_ready | vjp | 0.297417 | 0.007125 | 41.7× |
| kinematics | 3 | [] | including_state | dense | 1.937486 | 0.017542 | 110.5× |
| kinematics | 3 | [] | including_state | jvp | 1.934208 | 0.014667 | 131.9× |
| kinematics | 3 | [] | including_state | vjp | 1.949930 | 0.015375 | 126.8× |

Maximum absolute difference: 1.77636e-15.
Maximum relative Frobenius error: 6.73496e-16.
Full timing samples, first calls, environment, and shapes are in the adjacent JSON.
