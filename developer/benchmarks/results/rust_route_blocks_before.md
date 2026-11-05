# NumPy / Rust analytical Jacobian comparison

Same-build analytical backends; no JAX or numerical differentiation. Rust NumPy fallbacks and NumPy native dispatch blocked. Input/output boundary costs included. Model construction excluded. state_ready reuses current state and warmed derivative workspaces; including_state alternates two motions and includes import + state calculation + derivative. First_ms is first timed call, not cold compilation. Milliseconds per entire call (batch time is for six samples).

Model: tests/test_model/branched_fixed.urdf (3 DOF). Float64; seed 20260923; gravity [0.2, -0.3, -9.81].
Warmup 5; 15 samples × 3 calls; median milliseconds.

| Family | Order | Batch | Scope | Operation | NumPy ms | Rust ms | Speedup |
|---|---:|---|---|---|---:|---:|---:|
| kinematics | 3 | [] | state_ready | dense | 0.178306 | 0.047125 | 3.8× |
| kinematics | 3 | [] | state_ready | jvp | 0.173430 | 0.010181 | 17.0× |
| kinematics | 3 | [] | state_ready | vjp | 0.177458 | 0.010625 | 16.7× |
| kinematics | 3 | [] | including_state | dense | 0.918458 | 0.057222 | 16.1× |
| kinematics | 3 | [] | including_state | jvp | 0.928625 | 0.015389 | 60.3× |
| kinematics | 3 | [] | including_state | vjp | 0.934278 | 0.016125 | 57.9× |
| torque | 3 | [] | state_ready | dense | 3.051722 | 0.009944 | 306.9× |
| torque | 3 | [] | state_ready | jvp | 3.023194 | 0.009861 | 306.6× |
| torque | 3 | [] | state_ready | vjp | 3.063597 | 0.031250 | 98.0× |
| torque | 3 | [] | including_state | dense | 8.098444 | 0.020472 | 395.6× |
| torque | 3 | [] | including_state | jvp | 8.086278 | 0.020889 | 387.1× |
| torque | 3 | [] | including_state | vjp | 8.134653 | 0.042486 | 191.5× |
| mixed | 3 | [] | state_ready | dense | 3.209486 | 0.094083 | 34.1× |
| mixed | 3 | [] | state_ready | jvp | 3.197389 | 0.014875 | 215.0× |
| mixed | 3 | [] | state_ready | vjp | 3.343681 | 0.015278 | 218.9× |
| mixed | 3 | [] | including_state | dense | 8.369486 | 0.103555 | 80.8× |
| mixed | 3 | [] | including_state | jvp | 8.363153 | 0.025889 | 323.0× |
| mixed | 3 | [] | including_state | vjp | 8.491750 | 0.027194 | 312.3× |
| mixed | 3 | [6] | state_ready | dense | 4.296736 | 0.485139 | 8.9× |
| mixed | 3 | [6] | state_ready | jvp | 4.136542 | 0.060722 | 68.1× |
| mixed | 3 | [6] | state_ready | vjp | 40.533320 | 0.056347 | 719.3× |
| mixed | 3 | [6] | including_state | dense | 10.899458 | 0.480986 | 22.7× |
| mixed | 3 | [6] | including_state | jvp | 10.778722 | 0.083847 | 128.6× |
| mixed | 3 | [6] | including_state | vjp | 43.510625 | 0.081819 | 531.8× |
| mixed | 6 | [] | state_ready | dense | 6.630639 | 0.358445 | 18.5× |
| mixed | 6 | [] | state_ready | jvp | 6.107278 | 0.024875 | 245.5× |
| mixed | 6 | [] | state_ready | vjp | 6.298792 | 0.024042 | 262.0× |
| mixed | 6 | [] | including_state | dense | 21.247014 | 0.368611 | 57.6× |
| mixed | 6 | [] | including_state | jvp | 20.677750 | 0.040306 | 513.0× |
| mixed | 6 | [] | including_state | vjp | 20.935375 | 0.040111 | 521.9× |

Maximum absolute difference: 4.88498e-15.
Maximum relative Frobenius error: 1.39388e-15.
Full timing samples, first calls, environment, and shapes are in the adjacent JSON.
