# Rust kinematic route-block comparison

Fresh before/after release builds, three-DOF branched fixed/revolute model.
Input/output conversion included; model construction excluded. Median milliseconds,
5 warmups, 15 samples × 3 calls. See the paired JSON files for environment,
extension hashes, first calls, timing samples, seed, and correctness errors.

| Scope | Operation | Before ms | After ms | Speedup |
|---|---|---:|---:|---:|
| state_ready | dense | 0.047125 | 0.009292 | 5.07× |
| state_ready | jvp | 0.010181 | 0.007000 | 1.45× |
| state_ready | vjp | 0.010625 | 0.007889 | 1.35× |
| including_state | dense | 0.057222 | 0.014208 | 4.03× |
| including_state | jvp | 0.015389 | 0.012111 | 1.27× |
| including_state | vjp | 0.016125 | 0.013222 | 1.22× |

This workload selects world position, spatial velocity, and acceleration of a_tip;
the Jacobian is 15 × 9. `including_state` alternates two motions to invalidate
primal caches, and includes motion import + kinematics + derivative.

The new algorithm visits the selected ancestor routes, builds 6×6 Taylor
adjoint blocks and six-component screw series, then assembles/applies output
blocks. It does not build a full CMTM matrix. Dense kinematic Jacobians no
longer use identity-seeded tangents; JVP/VJP do not form a dense Jacobian.

The general selected-kinematics path (link/joint pose and local/world spatial
motion) uses these blocks at any supported order; dedicated local fast paths remain. Requests mixed with dynamics retain
the existing dynamics recurrence; dedicated torque kernels are unchanged.

After-build maximum absolute difference across benchmark outputs: 4.88498e-15.
Maximum relative Frobenius error: 1.39388e-15.

Absolute times differ from earlier sessions; this report uses a freshly
measured baseline rather than reusing the older NumPy/Rust comparison times.
