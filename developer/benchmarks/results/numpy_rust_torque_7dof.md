# NumPy / Rust analytical Jacobian comparison

Same-build analytical backends; no JAX or numerical differentiation. Rust NumPy fallbacks and NumPy native dispatch blocked. Input/output boundary costs included. Model construction excluded. state_ready reuses current state and warmed derivative workspaces; including_state alternates two motions and includes import + state calculation + derivative. First_ms is first timed call, not cold compilation. Milliseconds per entire call (batch time is for six samples).

Model: synthetic serial 7-DOF revolute chain with fixed tool (7 DOF). Float64; seed 20260923; gravity [0.2, -0.3, -9.81].
Warmup 5; 15 samples × 3 calls; median milliseconds.

| Family | Order | Batch | Scope | Operation | NumPy ms | Rust ms | Speedup |
|---|---:|---|---|---|---:|---:|---:|
| torque | 3 | [] | state_ready | dense | 5.371250 | 0.020236 | 265.4× |
| torque | 3 | [] | state_ready | jvp | 5.177639 | 0.014514 | 356.7× |
| torque | 3 | [] | state_ready | vjp | 5.410097 | 0.055680 | 97.2× |
| torque | 3 | [] | including_state | dense | 17.015861 | 0.031514 | 539.9× |
| torque | 3 | [] | including_state | jvp | 16.688000 | 0.026500 | 629.7× |
| torque | 3 | [] | including_state | vjp | 16.839528 | 0.068472 | 245.9× |

Maximum absolute difference: 6.39488e-14.
Maximum relative Frobenius error: 2.55272e-15.
Full timing samples, first calls, environment, and shapes are in the adjacent JSON.
