# NumPy / Rust analytical Jacobian comparison

Same-build analytical backends; no JAX or numerical differentiation. Rust NumPy fallbacks and NumPy native dispatch blocked. Input/output boundary costs included. Model construction excluded. state_ready reuses current state and warmed derivative workspaces; including_state alternates two motions and includes import + state calculation + derivative. First_ms is first timed call, not cold compilation. Milliseconds per entire call (batch time is for six samples).

Model: tests/test_model/branched_fixed.urdf (3 DOF). Float64; seed 20260923; gravity [0.2, -0.3, -9.81].
Warmup 5; 15 samples × 3 calls; median milliseconds.

| Family | Order | Batch | Scope | Operation | NumPy ms | Rust ms | Speedup |
|---|---:|---|---|---|---:|---:|---:|
| kinematics | 3 | [] | state_ready | dense | 0.172486 | 0.009042 | 19.1× |
| kinematics | 3 | [] | state_ready | jvp | 0.171208 | 0.007056 | 24.3× |
| kinematics | 3 | [] | state_ready | vjp | 0.175083 | 0.007986 | 21.9× |
| kinematics | 3 | [] | including_state | dense | 0.894472 | 0.014236 | 62.8× |
| kinematics | 3 | [] | including_state | jvp | 0.920861 | 0.012333 | 74.7× |
| kinematics | 3 | [] | including_state | vjp | 0.921028 | 0.013708 | 67.2× |
| torque | 3 | [] | state_ready | dense | 3.022972 | 0.009722 | 310.9× |
| torque | 3 | [] | state_ready | jvp | 3.006320 | 0.010417 | 288.6× |
| torque | 3 | [] | state_ready | vjp | 3.028153 | 0.030861 | 98.1× |
| torque | 3 | [] | including_state | dense | 8.061833 | 0.019653 | 410.2× |
| torque | 3 | [] | including_state | jvp | 8.079042 | 0.019889 | 406.2× |
| torque | 3 | [] | including_state | vjp | 8.083278 | 0.041694 | 193.9× |
| mixed | 3 | [] | state_ready | dense | 3.286417 | 0.086806 | 37.9× |
| mixed | 3 | [] | state_ready | jvp | 3.175931 | 0.015194 | 209.0× |
| mixed | 3 | [] | state_ready | vjp | 3.348319 | 0.015083 | 222.0× |
| mixed | 3 | [] | including_state | dense | 8.364278 | 0.094417 | 88.6× |
| mixed | 3 | [] | including_state | jvp | 8.475347 | 0.027458 | 308.7× |
| mixed | 3 | [] | including_state | vjp | 8.488070 | 0.027319 | 310.7× |
| mixed | 3 | [6] | state_ready | dense | 4.352181 | 0.462417 | 9.4× |
| mixed | 3 | [6] | state_ready | jvp | 4.169639 | 0.063222 | 66.0× |
| mixed | 3 | [6] | state_ready | vjp | 40.523528 | 0.054486 | 743.7× |
| mixed | 3 | [6] | including_state | dense | 10.992153 | 0.480903 | 22.9× |
| mixed | 3 | [6] | including_state | jvp | 10.791236 | 0.084583 | 127.6× |
| mixed | 3 | [6] | including_state | vjp | 43.582042 | 0.082222 | 530.1× |
| mixed | 6 | [] | state_ready | dense | 6.619917 | 0.344764 | 19.2× |
| mixed | 6 | [] | state_ready | jvp | 6.199458 | 0.025445 | 243.6× |
| mixed | 6 | [] | state_ready | vjp | 6.332764 | 0.023889 | 265.1× |
| mixed | 6 | [] | including_state | dense | 21.381083 | 0.366361 | 58.4× |
| mixed | 6 | [] | including_state | jvp | 20.720056 | 0.040500 | 511.6× |
| mixed | 6 | [] | including_state | vjp | 20.990667 | 0.039236 | 535.0× |

Maximum absolute difference: 4.88498e-15.
Maximum relative Frobenius error: 1.39388e-15.
Full timing samples, first calls, environment, and shapes are in the adjacent JSON.
