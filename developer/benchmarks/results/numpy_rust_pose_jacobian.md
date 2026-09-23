# NumPy / Rust analytical Jacobian comparison

Same-build analytical backends; no JAX or numerical differentiation. Rust NumPy fallbacks and NumPy native dispatch blocked. Input/output boundary costs included. Model construction excluded. state_ready reuses current state and warmed derivative workspaces; including_state alternates two motions and includes import + state calculation + derivative. First_ms is first timed call, not cold compilation. Milliseconds per entire call (batch time is for six samples).

Model: tests/test_model/branched_fixed.urdf (3 DOF). Float64; seed 20260923; gravity [0.2, -0.3, -9.81].
Warmup 5; 15 samples × 3 calls; median milliseconds.

| Family | Order | Batch | Scope | Operation | NumPy ms | Rust ms | Speedup |
|---|---:|---|---|---|---:|---:|---:|
| pose | 1 | [] | state_ready | dense | 0.083583 | 0.006917 | 12.1× |
| pose | 1 | [] | state_ready | jvp | 0.084833 | 0.005694 | 14.9× |
| pose | 1 | [] | state_ready | vjp | 0.088028 | 0.006139 | 14.3× |
| pose | 1 | [] | including_state | dense | 0.356486 | 0.011708 | 30.4× |
| pose | 1 | [] | including_state | jvp | 0.348236 | 0.011028 | 31.6× |
| pose | 1 | [] | including_state | vjp | 0.357736 | 0.011722 | 30.5× |

Maximum absolute difference: 2.22045e-16.
Maximum relative Frobenius error: 1.93505e-16.
Full timing samples, first calls, environment, and shapes are in the adjacent JSON.
