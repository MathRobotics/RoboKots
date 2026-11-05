# NumPy / Rust analytical Jacobian comparison

Same-build analytical backends; no JAX or numerical differentiation. Rust NumPy fallbacks and NumPy native dispatch blocked. Input/output boundary costs included. Model construction excluded. state_ready reuses current state and warmed derivative workspaces; including_state alternates two motions and includes import + state calculation + derivative. First_ms is first timed call, not cold compilation. Milliseconds per entire call (batch time is for six samples).

Model: synthetic serial 7-DOF revolute chain with fixed tool (7 DOF). Float64; seed 20260923; gravity [0.2, -0.3, -9.81].
Warmup 5; 15 samples × 3 calls; median milliseconds.

| Family | Order | Batch | Scope | Operation | NumPy ms | Rust ms | Speedup |
|---|---:|---|---|---|---:|---:|---:|
| torque | 3 | [] | state_ready | dense | 5.368958 | 0.021111 | 254.3× |
| torque | 3 | [] | state_ready | jvp | 5.146111 | 0.015000 | 343.1× |
| torque | 3 | [] | state_ready | vjp | 5.368805 | 0.056695 | 94.7× |
| torque | 3 | [] | including_state | dense | 16.902945 | 0.030972 | 545.7× |
| torque | 3 | [] | including_state | jvp | 16.604903 | 0.025139 | 660.5× |
| torque | 3 | [] | including_state | vjp | 16.831778 | 0.067486 | 249.4× |

Maximum absolute difference: 6.39488e-14.
Maximum relative Frobenius error: 2.55272e-15.
Full timing samples, first calls, environment, and shapes are in the adjacent JSON.
