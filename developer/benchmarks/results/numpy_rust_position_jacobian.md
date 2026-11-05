# NumPy / Rust analytical Jacobian comparison

Same-build analytical backends; no JAX or numerical differentiation. Rust NumPy fallbacks and NumPy native dispatch blocked. Input/output boundary costs included. Model construction excluded. state_ready reuses current state and warmed derivative workspaces; including_state alternates two motions and includes import + state calculation + derivative. First_ms is first timed call, not cold compilation. Milliseconds per entire call (batch time is for six samples).

Model: tests/test_model/branched_fixed.urdf (3 DOF). Float64; seed 20260923; gravity [0.2, -0.3, -9.81].
Warmup 5; 15 samples × 3 calls; median milliseconds.

| Family | Order | Batch | Scope | Operation | NumPy ms | Rust ms | Speedup |
|---|---:|---|---|---|---:|---:|---:|
| position | 1 | [] | state_ready | dense | 0.068986 | 0.005986 | 11.5× |
| position | 1 | [] | state_ready | jvp | 0.069917 | 0.004556 | 15.3× |
| position | 1 | [] | state_ready | vjp | 0.069292 | 0.004931 | 14.1× |
| position | 1 | [] | including_state | dense | 0.336708 | 0.010500 | 32.1× |
| position | 1 | [] | including_state | jvp | 0.334250 | 0.010153 | 32.9× |
| position | 1 | [] | including_state | vjp | 0.380958 | 0.009695 | 39.3× |

Maximum absolute difference: 6.7783e-17.
Maximum relative Frobenius error: 2.45167e-15.
Full timing samples, first calls, environment, and shapes are in the adjacent JSON.
