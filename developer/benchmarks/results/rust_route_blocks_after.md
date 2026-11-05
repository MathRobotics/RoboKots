# NumPy / Rust analytical Jacobian comparison

Same-build analytical backends; no JAX or numerical differentiation. Rust NumPy fallbacks and NumPy native dispatch blocked. Input/output boundary costs included. Model construction excluded. state_ready reuses current state and warmed derivative workspaces; including_state alternates two motions and includes import + state calculation + derivative. First_ms is first timed call, not cold compilation. Milliseconds per entire call (batch time is for six samples).

Model: tests/test_model/branched_fixed.urdf (3 DOF). Float64; seed 20260923; gravity [0.2, -0.3, -9.81].
Warmup 5; 15 samples × 3 calls; median milliseconds.

| Family | Order | Batch | Scope | Operation | NumPy ms | Rust ms | Speedup |
|---|---:|---|---|---|---:|---:|---:|
| kinematics | 3 | [] | state_ready | dense | 0.174333 | 0.009292 | 18.8× |
| kinematics | 3 | [] | state_ready | jvp | 0.177514 | 0.007000 | 25.4× |
| kinematics | 3 | [] | state_ready | vjp | 0.175972 | 0.007889 | 22.3× |
| kinematics | 3 | [] | including_state | dense | 0.910347 | 0.014208 | 64.1× |
| kinematics | 3 | [] | including_state | jvp | 0.910097 | 0.012111 | 75.1× |
| kinematics | 3 | [] | including_state | vjp | 0.922375 | 0.013222 | 69.8× |
| torque | 3 | [] | state_ready | dense | 2.999194 | 0.009889 | 303.3× |
| torque | 3 | [] | state_ready | jvp | 3.002111 | 0.009986 | 300.6× |
| torque | 3 | [] | state_ready | vjp | 3.077597 | 0.031042 | 99.1× |
| torque | 3 | [] | including_state | dense | 8.020986 | 0.020319 | 394.7× |
| torque | 3 | [] | including_state | jvp | 7.967292 | 0.021236 | 375.2× |
| torque | 3 | [] | including_state | vjp | 8.129083 | 0.041792 | 194.5× |
| mixed | 3 | [] | state_ready | dense | 3.198917 | 0.085736 | 37.3× |
| mixed | 3 | [] | state_ready | jvp | 3.131569 | 0.015556 | 201.3× |
| mixed | 3 | [] | state_ready | vjp | 3.310236 | 0.014945 | 221.5× |
| mixed | 3 | [] | including_state | dense | 8.324444 | 0.093667 | 88.9× |
| mixed | 3 | [] | including_state | jvp | 8.320736 | 0.026181 | 317.8× |
| mixed | 3 | [] | including_state | vjp | 8.435014 | 0.026861 | 314.0× |
| mixed | 3 | [6] | state_ready | dense | 4.256750 | 0.461917 | 9.2× |
| mixed | 3 | [6] | state_ready | jvp | 4.018861 | 0.060680 | 66.2× |
| mixed | 3 | [6] | state_ready | vjp | 40.272611 | 0.060805 | 662.3× |
| mixed | 3 | [6] | including_state | dense | 10.884403 | 0.485180 | 22.4× |
| mixed | 3 | [6] | including_state | jvp | 10.641917 | 0.086250 | 123.4× |
| mixed | 3 | [6] | including_state | vjp | 43.182055 | 0.080708 | 535.0× |
| mixed | 6 | [] | state_ready | dense | 6.545111 | 0.347694 | 18.8× |
| mixed | 6 | [] | state_ready | jvp | 6.102986 | 0.025250 | 241.7× |
| mixed | 6 | [] | state_ready | vjp | 6.350889 | 0.023347 | 272.0× |
| mixed | 6 | [] | including_state | dense | 21.363111 | 0.425542 | 50.2× |
| mixed | 6 | [] | including_state | jvp | 22.277361 | 0.039361 | 566.0× |
| mixed | 6 | [] | including_state | vjp | 21.371014 | 0.039695 | 538.4× |

Maximum absolute difference: 4.88498e-15.
Maximum relative Frobenius error: 1.39388e-15.
Full timing samples, first calls, environment, and shapes are in the adjacent JSON.
