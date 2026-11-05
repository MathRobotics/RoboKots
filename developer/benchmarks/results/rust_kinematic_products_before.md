# NumPy / Rust analytical Jacobian comparison

Same-build analytical backends; no JAX or numerical differentiation. Rust NumPy fallbacks and NumPy native dispatch blocked. Input/output boundary costs included. Model construction excluded. state_ready reuses current state and warmed derivative workspaces; including_state alternates two motions and includes import + state calculation + derivative. First_ms is first timed call, not cold compilation. Milliseconds per entire call (batch time is for six samples).

Model: synthetic serial 7-DOF revolute chain with fixed tool (7 DOF). Float64; seed 20260923; gravity [0.2, -0.3, -9.81].
Warmup 5; 15 samples × 3 calls; median milliseconds.

| Family | Order | Batch | Scope | Operation | NumPy ms | Rust ms | Speedup |
|---|---:|---|---|---|---:|---:|---:|
| kinematics | 3 | [] | state_ready | dense | 0.304917 | 0.011361 | 26.8× |
| kinematics | 3 | [] | state_ready | jvp | 0.285500 | 0.009069 | 31.5× |
| kinematics | 3 | [] | state_ready | vjp | 0.294125 | 0.009917 | 29.7× |
| kinematics | 3 | [] | including_state | dense | 1.975681 | 0.016653 | 118.6× |
| kinematics | 3 | [] | including_state | jvp | 1.979875 | 0.014361 | 137.9× |
| kinematics | 3 | [] | including_state | vjp | 1.987097 | 0.015764 | 126.1× |

Maximum absolute difference: 1.77636e-15.
Maximum relative Frobenius error: 6.73496e-16.
Full timing samples, first calls, environment, and shapes are in the adjacent JSON.
