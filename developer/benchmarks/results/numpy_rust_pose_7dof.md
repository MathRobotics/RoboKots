# NumPy / Rust analytical Jacobian comparison

Same-build analytical backends; no JAX or numerical differentiation. Rust NumPy fallbacks and NumPy native dispatch blocked. Input/output boundary costs included. Model construction excluded. state_ready reuses current state and warmed derivative workspaces; including_state alternates two motions and includes import + state calculation + derivative. First_ms is first timed call, not cold compilation. Milliseconds per entire call (batch time is for six samples).

Model: synthetic serial 7-DOF revolute chain with fixed tool (7 DOF). Float64; seed 20260923; gravity [0.2, -0.3, -9.81].
Warmup 5; 15 samples × 3 calls; median milliseconds.

| Family | Order | Batch | Scope | Operation | NumPy ms | Rust ms | Speedup |
|---|---:|---|---|---|---:|---:|---:|
| pose | 1 | [] | state_ready | dense | 0.166945 | 0.008181 | 20.4× |
| pose | 1 | [] | state_ready | jvp | 0.171736 | 0.006875 | 25.0× |
| pose | 1 | [] | state_ready | vjp | 0.183528 | 0.007250 | 25.3× |
| pose | 1 | [] | including_state | dense | 0.751875 | 0.012931 | 58.1× |
| pose | 1 | [] | including_state | jvp | 0.743236 | 0.012486 | 59.5× |
| pose | 1 | [] | including_state | vjp | 0.765722 | 0.013167 | 58.2× |

Maximum absolute difference: 1.22125e-15.
Maximum relative Frobenius error: 7.42392e-16.
Full timing samples, first calls, environment, and shapes are in the adjacent JSON.
