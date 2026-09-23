# NumPy / Rust analytical Jacobian comparison

Same-build analytical backends; no JAX or numerical differentiation. Rust NumPy fallbacks and NumPy native dispatch blocked. Input/output boundary costs included. Model construction excluded. state_ready reuses current state and warmed derivative workspaces; including_state alternates two motions and includes import + state calculation + derivative. First_ms is first timed call, not cold compilation. Milliseconds per entire call (batch time is for six samples).

Model: tests/test_model/branched_fixed.urdf (3 DOF). Float64; seed 20260923; gravity [0.2, -0.3, -9.81].
Warmup 5; 15 samples × 3 calls; median milliseconds.

| Family | Order | Batch | Scope | Operation | NumPy ms | Rust ms | Speedup |
|---|---:|---|---|---|---:|---:|---:|
| kinematics | 3 | [] | state_ready | dense | 1.298222 | 0.213528 | 6.1× |
| kinematics | 3 | [] | state_ready | jvp | 1.310458 | 0.036792 | 35.6× |
| kinematics | 3 | [] | state_ready | vjp | 1.305320 | 0.040139 | 32.5× |
| kinematics | 3 | [] | including_state | dense | 5.001292 | 0.252653 | 19.8× |
| kinematics | 3 | [] | including_state | jvp | 4.662014 | 0.078264 | 59.6× |
| kinematics | 3 | [] | including_state | vjp | 5.497208 | 0.067528 | 81.4× |
| torque | 3 | [] | state_ready | dense | 14.690819 | 0.031444 | 467.2× |
| torque | 3 | [] | state_ready | jvp | 14.403139 | 0.026861 | 536.2× |
| torque | 3 | [] | state_ready | vjp | 15.023903 | 0.057236 | 262.5× |
| torque | 3 | [] | including_state | dense | 14.937042 | 0.036070 | 414.1× |
| torque | 3 | [] | including_state | jvp | 14.988750 | 0.037903 | 395.5× |
| torque | 3 | [] | including_state | vjp | 16.444819 | 0.080458 | 204.4× |
| mixed | 3 | [] | state_ready | dense | 5.986778 | 0.166889 | 35.9× |
| mixed | 3 | [] | state_ready | jvp | 5.820278 | 0.027111 | 214.7× |
| mixed | 3 | [] | state_ready | vjp | 6.070875 | 0.028319 | 214.4× |
| mixed | 3 | [] | including_state | dense | 15.030403 | 0.195556 | 76.9× |
| mixed | 3 | [] | including_state | jvp | 15.331917 | 0.048667 | 315.0× |
| mixed | 3 | [] | including_state | vjp | 14.967806 | 0.049708 | 301.1× |
| mixed | 3 | [6] | state_ready | dense | 7.590417 | 0.883653 | 8.6× |
| mixed | 3 | [6] | state_ready | jvp | 7.236069 | 0.110875 | 65.3× |
| mixed | 3 | [6] | state_ready | vjp | 71.702833 | 0.104458 | 686.4× |
| mixed | 3 | [6] | including_state | dense | 19.145736 | 0.883278 | 21.7× |
| mixed | 3 | [6] | including_state | jvp | 18.931180 | 0.152597 | 124.1× |
| mixed | 3 | [6] | including_state | vjp | 77.135458 | 0.148167 | 520.6× |
| mixed | 6 | [] | state_ready | dense | 11.564097 | 0.641792 | 18.0× |
| mixed | 6 | [] | state_ready | jvp | 10.950167 | 0.046667 | 234.6× |
| mixed | 6 | [] | state_ready | vjp | 11.171556 | 0.044056 | 253.6× |
| mixed | 6 | [] | including_state | dense | 37.978014 | 0.671222 | 56.6× |
| mixed | 6 | [] | including_state | jvp | 36.707083 | 0.073125 | 502.0× |
| mixed | 6 | [] | including_state | vjp | 37.197458 | 0.070500 | 527.6× |

Maximum absolute difference: 4.88498e-15.
Maximum relative Frobenius error: 1.39388e-15.
Full timing samples, first calls, environment, and shapes are in the adjacent JSON.
