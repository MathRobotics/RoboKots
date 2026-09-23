# NumPy / Rust analytical Jacobian comparison

Same-build analytical backends; no JAX or numerical differentiation. Rust NumPy fallbacks and NumPy native dispatch blocked. Input/output boundary costs included. Model construction excluded. state_ready reuses current state and warmed derivative workspaces; including_state alternates two motions and includes import + state calculation + derivative. First_ms is first timed call, not cold compilation. Milliseconds per entire call (batch time is for six samples).

Model: synthetic serial 7-DOF revolute chain with fixed tool (7 DOF). Float64; seed 20260923; gravity [0.2, -0.3, -9.81].
Warmup 5; 15 samples × 3 calls; median milliseconds.

| Family | Order | Batch | Scope | Operation | NumPy ms | Rust ms | Speedup |
|---|---:|---|---|---|---:|---:|---:|
| torque | 3 | [] | state_ready | dense | 5.120875 | 0.020833 | 245.8× |
| torque | 3 | [] | state_ready | jvp | 5.013597 | 0.013417 | 373.7× |
| torque | 3 | [] | state_ready | vjp | 5.186778 | 0.014986 | 346.1× |
| torque | 3 | [] | including_state | dense | 16.526389 | 0.031625 | 522.6× |
| torque | 3 | [] | including_state | jvp | 16.206403 | 0.025375 | 638.7× |
| torque | 3 | [] | including_state | vjp | 16.451167 | 0.027167 | 605.6× |

Maximum absolute difference: 7.10543e-14.
Maximum relative Frobenius error: 2.20016e-15.
Full timing samples, first calls, environment, and shapes are in the adjacent JSON.
