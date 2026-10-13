# NumPy analytical Jacobian breakdown

Measured UTC: 2026-10-01T08:49:10.618938+00:00; runtime source commit `32f0b0b9c25570c308cd36d8a7fbdfa178cbd555`.

Linux-7.0.0-34-generic-x86_64-with-glibc2.39; AMD Ryzen 9 7940HS w/ Radeon 780M Graphics; Python 3.13.1; NumPy 2.4.6; float64.

All active-joint dense torque Jacobians, derivatives 0 and 2, fixed motion orders 3 and 5. Same serial-model family, gravity [0.2,-0.3,-9.81], seed 20261001 + DOF as dynamics_complexity. Rust calls are blocked. Two motion inputs alternate; dictionary materialization is disabled. Model construction/imports are excluded. BLAS/OpenMP thread counts are one.

Uninstrumented phases: 5 samples after 2 warmups, measured before AND after profiling. Profile: one separate warmup, then 3 samples. No numerical/AD/JIT timings. Both inputs are verified bit-for-bit against uninstrumented outputs. JSON retains samples, shapes, operation counts, conventional dense FLOP estimates, function times and source hashes.

## Findings

At 64 DOF / derivative 2, construction is 53.9% and intercepted products are 43.8% of instrumented Jacobian time.

32→64 DOF: intercepted-product FLOPs increase 7.472×, product time 6.621×, construction time 3.670×. The matrix-product work approaches cubic scaling, while construction remains a substantial elapsed-time term.

Potential optimization targets visible in the product/routine tables: apply diagonal factorial factors as row/column scaling (partial_dynamics.py lines 23/137), apply spatial-inertia blocks without whole-body dense block-diagonal multiplication, and reuse repeated coordinate-to-link gradient construction within one Jacobian call. These are findings, not runtime changes implemented by this benchmark.

## Uninstrumented phase times (median ms)

The table pools the before/after phase samples; full is measured directly. Phase medians need not sum to the full median. State includes outward computation and API cache bookkeeping. Jacobian is the FIRST call on each freshly computed state, including lazy relative-CMTM/geometry cache construction. It differs from repeated state-ready calls that reuse those caches.

| Derivative | DOF | Import | State | Jacobian | Full | J after/before |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4 | 0.0780 | 2.4337 | 10.5416 | 13.0706 | 0.999 |
| 0 | 8 | 0.1938 | 4.4289 | 23.8822 | 28.5007 | 1.006 |
| 0 | 16 | 0.7335 | 8.3168 | 66.9713 | 76.0425 | 0.994 |
| 0 | 32 | 3.6572 | 16.0917 | 222.0954 | 241.8352 | 1.017 |
| 0 | 64 | 13.4041 | 31.8158 | 885.0429 | 929.5805 | 0.978 |
| 2 | 4 | 0.1074 | 4.6695 | 22.1472 | 26.9345 | 0.998 |
| 2 | 8 | 0.3684 | 8.6616 | 53.9656 | 62.9755 | 0.995 |
| 2 | 16 | 1.5435 | 16.3811 | 165.2666 | 183.1416 | 1.001 |
| 2 | 32 | 5.4249 | 32.2724 | 626.8057 | 664.4580 | 1.001 |
| 2 | 64 | 18.0326 | 63.6717 | 2800.1662 | 2882.0175 | 0.993 |

## Instrumented Jacobian times (median ms)

Construction: whole-body kinematics/dynamics/partial-matrix builders and their callees, including CMTM/block geometry and allocation; excludes intercepted 2D ndarray products. Products: explicit ndarray @ ndarray in loaded RoboKots module functions. Gravity: gravity derivative routines excluding nested construction/products. Assembly: dense derivative composition excluding its nested categories (e.g. additions/slicing). Other: output selection, dispatch and profiler bookkeeping. Categories are exclusive and sum to each raw sample's elapsed time. Individual category medians need not sum to total median.

External MathRobo internals/class methods are not rewritten; their small internal operations stay in the enclosing category. Thus construction is a measured routine group, not pure allocation time. Instrumentation overhead remains in these samples; do not use their total as production timing or rescale them to the uninstrumented total.

| Derivative | DOF | Total | Construction | Products | Gravity | Assembly | Other | Product share | Profile/plain J |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4 | 11.8713 | 8.9494 | 0.4336 | 2.2775 | 0.1458 | 0.0843 | 3.7% | 1.126 |
| 0 | 8 | 25.9996 | 20.8042 | 1.2143 | 3.7122 | 0.1694 | 0.1128 | 4.7% | 1.089 |
| 0 | 16 | 71.2007 | 58.8693 | 5.1575 | 6.6686 | 0.3214 | 0.1755 | 7.2% | 1.063 |
| 0 | 32 | 233.7824 | 191.7953 | 27.8223 | 13.1317 | 0.7478 | 0.2712 | 11.9% | 1.053 |
| 0 | 64 | 928.5506 | 718.9315 | 178.9780 | 27.8797 | 2.3149 | 0.4796 | 19.3% | 1.049 |
| 2 | 4 | 23.6819 | 17.3837 | 1.6251 | 4.4256 | 0.1624 | 0.0923 | 6.9% | 1.069 |
| 2 | 8 | 56.4896 | 42.5244 | 6.2998 | 7.3225 | 0.2226 | 0.1292 | 11.2% | 1.047 |
| 2 | 16 | 170.5174 | 125.2188 | 31.1710 | 13.4088 | 0.4406 | 0.1904 | 18.3% | 1.032 |
| 2 | 32 | 638.8704 | 420.3676 | 189.3498 | 26.2185 | 2.4095 | 0.2955 | 29.6% | 1.019 |
| 2 | 64 | 2860.0093 | 1542.7395 | 1253.6177 | 58.2593 | 5.5165 | 0.5314 | 43.8% | 1.021 |

## Most expensive intercepted matrix products

Top five per case; source line refers to the unchanged runtime source. FLOPs use 2mkn per m×k times k×n product, regardless of zero entries. They are arithmetic estimates, not hardware counters. Operator-only timing excludes the surrounding profiling bookkeeping.

| Derivative | DOF | Operation / dimensions | ms | Calls per sample | Estimated FLOPs |
|---:|---:|---|---:|---:|---:|
| 0 | 4 | `robokots/core/kernels/whole_body/gravity_derivatives.py:150: (6, 6) @ (6, 12)` | 0.0477 | 17 | 14688 |
| 0 | 4 | `robokots/core/kernels/whole_body/gravity_derivatives.py:226: (6, 6) @ (6, 12)` | 0.0311 | 12 | 10368 |
| 0 | 4 | `robokots/core/kernels/whole_body/partial_dynamics.py:23: (72, 72) @ (72, 72)` | 0.0215 | 2 | 1492992 |
| 0 | 4 | `robokots/core/kernels/whole_body/kinematics_derivatives.py:63: (18, 18) @ (18, 18)` | 0.0187 | 15 | 174960 |
| 0 | 4 | `robokots/core/kernels/whole_body/gravity_derivatives.py:167: (6, 6) @ (6, 12)` | 0.0183 | 22 | 19008 |
| 0 | 8 | `robokots/core/kernels/whole_body/gravity_derivatives.py:150: (6, 6) @ (6, 24)` | 0.1174 | 29 | 50112 |
| 0 | 8 | `robokots/core/kernels/whole_body/partial_dynamics.py:23: (120, 120) @ (120, 120)` | 0.1113 | 2 | 6912000 |
| 0 | 8 | `robokots/core/kernels/whole_body/partial_dynamics.py:137: (108, 108) @ (108, 108)` | 0.0853 | 2 | 5038848 |
| 0 | 8 | `robokots/core/kernels/whole_body/dynamics_derivatives.py:34: (120, 120) @ (120, 162)` | 0.0767 | 1 | 4665600 |
| 0 | 8 | `robokots/core/kernels/whole_body/gravity_derivatives.py:226: (6, 6) @ (6, 24)` | 0.0765 | 20 | 34560 |
| 0 | 16 | `robokots/core/kernels/whole_body/partial_dynamics.py:23: (216, 216) @ (216, 216)` | 0.5859 | 2 | 40310784 |
| 0 | 16 | `robokots/core/kernels/whole_body/dynamics_derivatives.py:34: (216, 216) @ (216, 306)` | 0.5722 | 1 | 28553472 |
| 0 | 16 | `robokots/core/kernels/whole_body/partial_dynamics.py:137: (204, 204) @ (204, 204)` | 0.5059 | 2 | 33958656 |
| 0 | 16 | `robokots/core/kernels/whole_body/partial_dynamics.py:34: (216, 216) @ (216, 216)` | 0.3974 | 1 | 20155392 |
| 0 | 16 | `robokots/core/kernels/whole_body/gravity_derivatives.py:150: (6, 6) @ (6, 48)` | 0.3527 | 53 | 183168 |
| 0 | 32 | `robokots/core/kernels/whole_body/partial_dynamics.py:23: (408, 408) @ (408, 408)` | 3.7907 | 2 | 271669248 |
| 0 | 32 | `robokots/core/kernels/whole_body/partial_dynamics.py:137: (396, 396) @ (396, 396)` | 3.4895 | 2 | 248396544 |
| 0 | 32 | `robokots/core/kernels/whole_body/dynamics_derivatives.py:34: (408, 408) @ (408, 594)` | 3.3419 | 1 | 197759232 |
| 0 | 32 | `robokots/core/kernels/whole_body/partial_dynamics.py:34: (408, 408) @ (408, 408)` | 2.2727 | 1 | 135834624 |
| 0 | 32 | `robokots/core/kernels/whole_body/partial_dynamics.py:148: (396, 408) @ (408, 408)` | 1.8751 | 1 | 131839488 |
| 0 | 64 | `robokots/core/kernels/whole_body/partial_dynamics.py:23: (792, 792) @ (792, 792)` | 28.3651 | 2 | 1987172352 |
| 0 | 64 | `robokots/core/kernels/whole_body/partial_dynamics.py:137: (780, 780) @ (780, 780)` | 26.4167 | 2 | 1898208000 |
| 0 | 64 | `robokots/core/kernels/whole_body/dynamics_derivatives.py:34: (792, 792) @ (792, 1170)` | 20.9927 | 1 | 1467797760 |
| 0 | 64 | `robokots/core/kernels/whole_body/partial_dynamics.py:34: (792, 792) @ (792, 792)` | 14.4298 | 1 | 993586176 |
| 0 | 64 | `robokots/core/kernels/whole_body/partial_dynamics.py:148: (780, 792) @ (792, 792)` | 13.8767 | 1 | 978531840 |
| 2 | 4 | `robokots/core/kernels/whole_body/gravity_derivatives.py:150: (18, 18) @ (18, 20)` | 0.3324 | 17 | 220320 |
| 2 | 4 | `robokots/core/kernels/whole_body/gravity_derivatives.py:226: (18, 18) @ (18, 20)` | 0.2330 | 12 | 155520 |
| 2 | 4 | `robokots/core/kernels/whole_body/partial_dynamics.py:23: (144, 144) @ (144, 144)` | 0.1864 | 2 | 11943936 |
| 2 | 4 | `robokots/core/kernels/whole_body/partial_dynamics.py:137: (120, 120) @ (120, 120)` | 0.1143 | 2 | 6912000 |
| 2 | 4 | `robokots/core/kernels/whole_body/dynamics_derivatives.py:34: (144, 144) @ (144, 150)` | 0.0994 | 1 | 6220800 |
| 2 | 8 | `robokots/core/kernels/whole_body/gravity_derivatives.py:150: (18, 18) @ (18, 40)` | 1.0953 | 29 | 751680 |
| 2 | 8 | `robokots/core/kernels/whole_body/partial_dynamics.py:23: (240, 240) @ (240, 240)` | 0.8060 | 2 | 55296000 |
| 2 | 8 | `robokots/core/kernels/whole_body/gravity_derivatives.py:226: (18, 18) @ (18, 40)` | 0.7388 | 20 | 518400 |
| 2 | 8 | `robokots/core/kernels/whole_body/partial_dynamics.py:137: (216, 216) @ (216, 216)` | 0.6003 | 2 | 40310784 |
| 2 | 8 | `robokots/core/kernels/whole_body/dynamics_derivatives.py:34: (240, 240) @ (240, 270)` | 0.4460 | 1 | 31104000 |
| 2 | 16 | `robokots/core/kernels/whole_body/partial_dynamics.py:23: (432, 432) @ (432, 432)` | 4.4960 | 2 | 322486272 |
| 2 | 16 | `robokots/core/kernels/whole_body/gravity_derivatives.py:150: (18, 18) @ (18, 80)` | 3.8937 | 53 | 2747520 |
| 2 | 16 | `robokots/core/kernels/whole_body/partial_dynamics.py:137: (408, 408) @ (408, 408)` | 3.8564 | 2 | 271669248 |
| 2 | 16 | `robokots/core/kernels/whole_body/dynamics_derivatives.py:34: (432, 432) @ (432, 510)` | 2.7284 | 1 | 190356480 |
| 2 | 16 | `robokots/core/kernels/whole_body/gravity_derivatives.py:226: (18, 18) @ (18, 80)` | 2.5912 | 36 | 1866240 |
| 2 | 32 | `robokots/core/kernels/whole_body/partial_dynamics.py:23: (816, 816) @ (816, 816)` | 30.7376 | 2 | 2173353984 |
| 2 | 32 | `robokots/core/kernels/whole_body/partial_dynamics.py:137: (792, 792) @ (792, 792)` | 27.3427 | 2 | 1987172352 |
| 2 | 32 | `robokots/core/kernels/whole_body/dynamics_derivatives.py:34: (816, 816) @ (816, 990)` | 18.8475 | 1 | 1318394880 |
| 2 | 32 | `robokots/core/kernels/whole_body/partial_dynamics.py:148: (792, 816) @ (816, 816)` | 15.5883 | 1 | 1054715904 |
| 2 | 32 | `robokots/core/kernels/whole_body/partial_dynamics.py:34: (816, 816) @ (816, 816)` | 15.5084 | 1 | 1086676992 |
| 2 | 64 | `robokots/core/kernels/whole_body/partial_dynamics.py:23: (1584, 1584) @ (1584, 1584)` | 214.4842 | 2 | 15897378816 |
| 2 | 64 | `robokots/core/kernels/whole_body/partial_dynamics.py:137: (1560, 1560) @ (1560, 1560)` | 204.1720 | 2 | 15185664000 |
| 2 | 64 | `robokots/core/kernels/whole_body/dynamics_derivatives.py:34: (1584, 1584) @ (1584, 1950)` | 131.6724 | 1 | 9785318400 |
| 2 | 64 | `robokots/core/kernels/whole_body/partial_dynamics.py:34: (1584, 1584) @ (1584, 1584)` | 107.4831 | 1 | 7948689408 |
| 2 | 64 | `robokots/core/kernels/whole_body/partial_dynamics.py:148: (1560, 1584) @ (1584, 1584)` | 105.7206 | 1 | 7828254720 |

## Construction/gravity function hotspots (inclusive)

Top six construction/gravity routine scopes, excluding outer assembly/selection wrappers. Parent/child times overlap; do not sum them. Inclusive routine times also contain nested products.

| Derivative | DOF | Function | Inclusive ms | Calls |
|---:|---:|---|---:|---:|
| 0 | 4 | `robokots.core.kernels.whole_body.gravity_derivatives.total_coord_to_joint_gravity_force_grad_mat` | 3.3337 | 1 |
| 0 | 4 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_coord_to_link_vel_grad_mat` | 3.2304 | 1 |
| 0 | 4 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_coord_to_link_tan_vel_grad_mat` | 2.9495 | 3 |
| 0 | 4 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_joint_tan_vel_to_link_vel_grad_mat` | 2.8461 | 1 |
| 0 | 4 | `robokots.core.kernels.whole_body.gravity_derivatives.total_link_pose_to_gravity_force_grad_matmul_rhs` | 2.4332 | 1 |
| 0 | 4 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_joint_tan_vel_to_link_tan_vel_grad_mat` | 2.2606 | 3 |
| 0 | 8 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_coord_to_link_vel_grad_mat` | 8.2878 | 1 |
| 0 | 8 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_coord_to_link_tan_vel_grad_mat` | 7.6698 | 3 |
| 0 | 8 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_joint_tan_vel_to_link_vel_grad_mat` | 7.5550 | 1 |
| 0 | 8 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_joint_tan_vel_to_link_tan_vel_grad_mat` | 6.4006 | 3 |
| 0 | 8 | `robokots.core.kernels.whole_body.gravity_derivatives.total_coord_to_joint_gravity_force_grad_mat` | 6.2727 | 1 |
| 0 | 8 | `robokots.core.kernels.whole_body.gravity_derivatives.total_link_pose_to_gravity_force_grad_matmul_rhs` | 4.0480 | 1 |
| 0 | 16 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_coord_to_link_vel_grad_mat` | 25.6581 | 1 |
| 0 | 16 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_joint_tan_vel_to_link_vel_grad_mat` | 24.0530 | 1 |
| 0 | 16 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_coord_to_link_tan_vel_grad_mat` | 23.4496 | 3 |
| 0 | 16 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_joint_tan_vel_to_link_tan_vel_grad_mat` | 20.8953 | 3 |
| 0 | 16 | `robokots.core.kernels.whole_body.gravity_derivatives.total_coord_to_joint_gravity_force_grad_mat` | 14.3847 | 1 |
| 0 | 16 | `robokots.core.kernels.whole_body.gravity_derivatives.total_link_pose_to_gravity_force_grad_matmul_rhs` | 7.5140 | 1 |
| 0 | 32 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_coord_to_link_vel_grad_mat` | 88.3594 | 1 |
| 0 | 32 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_joint_tan_vel_to_link_vel_grad_mat` | 84.3702 | 1 |
| 0 | 32 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_coord_to_link_tan_vel_grad_mat` | 82.7064 | 3 |
| 0 | 32 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_joint_tan_vel_to_link_tan_vel_grad_mat` | 77.0848 | 3 |
| 0 | 32 | `robokots.core.kernels.whole_body.gravity_derivatives.total_coord_to_joint_gravity_force_grad_mat` | 39.0679 | 1 |
| 0 | 32 | `robokots.core.kernels.whole_body.gravity_derivatives.total_link_pose_to_gravity_force_grad_matmul_rhs` | 15.6838 | 1 |
| 0 | 64 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_coord_to_link_tan_vel_grad_mat` | 340.6222 | 3 |
| 0 | 64 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_coord_to_link_vel_grad_mat` | 330.4514 | 1 |
| 0 | 64 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_joint_tan_vel_to_link_tan_vel_grad_mat` | 324.4451 | 3 |
| 0 | 64 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_joint_tan_vel_to_link_vel_grad_mat` | 316.5328 | 1 |
| 0 | 64 | `robokots.core.kernels.whole_body.gravity_derivatives.total_coord_to_joint_gravity_force_grad_mat` | 121.4798 | 1 |
| 0 | 64 | `robokots.core.kernels.whole_body.partial_dynamics.total_partial_link_tan_vel_to_joint_momentum_grad_mat` | 43.0077 | 1 |
| 2 | 4 | `robokots.core.kernels.whole_body.gravity_derivatives.total_coord_to_joint_gravity_force_grad_mat` | 7.8675 | 1 |
| 2 | 4 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_coord_to_link_tan_vel_grad_mat` | 7.0383 | 3 |
| 2 | 4 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_coord_to_link_vel_grad_mat` | 6.0366 | 1 |
| 2 | 4 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_joint_tan_vel_to_link_tan_vel_grad_mat` | 5.9094 | 3 |
| 2 | 4 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_joint_tan_vel_to_link_vel_grad_mat` | 5.4329 | 1 |
| 2 | 4 | `robokots.core.kernels.whole_body.gravity_derivatives.total_link_pose_to_gravity_force_grad_matmul_rhs` | 5.0862 | 1 |
| 2 | 8 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_coord_to_link_tan_vel_grad_mat` | 19.3651 | 3 |
| 2 | 8 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_joint_tan_vel_to_link_tan_vel_grad_mat` | 17.1112 | 3 |
| 2 | 8 | `robokots.core.kernels.whole_body.gravity_derivatives.total_coord_to_joint_gravity_force_grad_mat` | 16.7468 | 1 |
| 2 | 8 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_coord_to_link_vel_grad_mat` | 15.9480 | 1 |
| 2 | 8 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_joint_tan_vel_to_link_vel_grad_mat` | 14.6814 | 1 |
| 2 | 8 | `robokots.core.kernels.whole_body.gravity_derivatives.total_link_pose_to_gravity_force_grad_matmul_rhs` | 9.3392 | 1 |
| 2 | 16 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_coord_to_link_tan_vel_grad_mat` | 62.1492 | 3 |
| 2 | 16 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_joint_tan_vel_to_link_tan_vel_grad_mat` | 56.8641 | 3 |
| 2 | 16 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_coord_to_link_vel_grad_mat` | 49.9872 | 1 |
| 2 | 16 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_joint_tan_vel_to_link_vel_grad_mat` | 46.9437 | 1 |
| 2 | 16 | `robokots.core.kernels.whole_body.gravity_derivatives.total_coord_to_joint_gravity_force_grad_mat` | 43.9420 | 1 |
| 2 | 16 | `robokots.core.kernels.whole_body.gravity_derivatives.total_link_pose_to_gravity_force_grad_matmul_rhs` | 20.2683 | 1 |
| 2 | 32 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_coord_to_link_tan_vel_grad_mat` | 221.7475 | 3 |
| 2 | 32 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_joint_tan_vel_to_link_tan_vel_grad_mat` | 205.8720 | 3 |
| 2 | 32 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_coord_to_link_vel_grad_mat` | 175.1671 | 1 |
| 2 | 32 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_joint_tan_vel_to_link_vel_grad_mat` | 165.5484 | 1 |
| 2 | 32 | `robokots.core.kernels.whole_body.gravity_derivatives.total_coord_to_joint_gravity_force_grad_mat` | 134.0369 | 1 |
| 2 | 32 | `robokots.core.kernels.whole_body.gravity_derivatives.total_link_pose_to_gravity_force_grad_matmul_rhs` | 51.3648 | 1 |
| 2 | 64 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_coord_to_link_tan_vel_grad_mat` | 883.2522 | 3 |
| 2 | 64 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_joint_tan_vel_to_link_tan_vel_grad_mat` | 811.2031 | 3 |
| 2 | 64 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_coord_to_link_vel_grad_mat` | 665.6025 | 1 |
| 2 | 64 | `robokots.core.kernels.whole_body.kinematics_derivatives.total_joint_tan_vel_to_link_vel_grad_mat` | 620.3845 | 1 |
| 2 | 64 | `robokots.core.kernels.whole_body.gravity_derivatives.total_coord_to_joint_gravity_force_grad_mat` | 465.1151 | 1 |
| 2 | 64 | `robokots.core.kernels.whole_body.partial_dynamics.total_partial_link_tan_vel_to_joint_momentum_grad_mat` | 232.0964 | 1 |

## Fixed-order scaling

Adjacent-size ratios use pooled uninstrumented samples for state/Jacobian. Products/construction use instrumented samples. Effective α=log(T₂/T₁)/log(d₂/d₁). FLOP scaling verifies only the intercepted product work, not every operation in the algorithm.

| Derivative | DOFs | Quantity | Ratio | Effective α |
|---:|---|---|---:|---:|
| 0 | 4→8 | state | 1.820 | 0.864 |
| 0 | 4→8 | jacobian | 2.266 | 1.180 |
| 0 | 4→8 | construction | 2.325 | 1.217 |
| 0 | 4→8 | products | 2.800 | 1.486 |
| 0 | 4→8 | product_flops | 5.100 | 2.351 |
| 0 | 8→16 | state | 1.878 | 0.909 |
| 0 | 8→16 | jacobian | 2.804 | 1.488 |
| 0 | 8→16 | construction | 2.830 | 1.501 |
| 0 | 8→16 | products | 4.247 | 2.087 |
| 0 | 8→16 | product_flops | 6.228 | 2.639 |
| 0 | 16→32 | state | 1.935 | 0.952 |
| 0 | 16→32 | jacobian | 3.316 | 1.730 |
| 0 | 16→32 | construction | 3.258 | 1.704 |
| 0 | 16→32 | products | 5.394 | 2.431 |
| 0 | 16→32 | product_flops | 7.008 | 2.809 |
| 0 | 32→64 | state | 1.977 | 0.983 |
| 0 | 32→64 | jacobian | 3.985 | 1.995 |
| 0 | 32→64 | construction | 3.748 | 1.906 |
| 0 | 32→64 | products | 6.433 | 2.685 |
| 0 | 32→64 | product_flops | 7.473 | 2.902 |
| 2 | 4→8 | state | 1.855 | 0.891 |
| 2 | 4→8 | jacobian | 2.437 | 1.285 |
| 2 | 4→8 | construction | 2.446 | 1.291 |
| 2 | 4→8 | products | 3.876 | 1.955 |
| 2 | 4→8 | product_flops | 5.092 | 2.348 |
| 2 | 8→16 | state | 1.891 | 0.919 |
| 2 | 8→16 | jacobian | 3.062 | 1.615 |
| 2 | 8→16 | construction | 2.945 | 1.558 |
| 2 | 8→16 | products | 4.948 | 2.307 |
| 2 | 8→16 | product_flops | 6.224 | 2.638 |
| 2 | 16→32 | state | 1.970 | 0.978 |
| 2 | 16→32 | jacobian | 3.793 | 1.923 |
| 2 | 16→32 | construction | 3.357 | 1.747 |
| 2 | 16→32 | products | 6.075 | 2.603 |
| 2 | 16→32 | product_flops | 7.006 | 2.809 |
| 2 | 32→64 | state | 1.973 | 0.980 |
| 2 | 32→64 | jacobian | 4.467 | 2.159 |
| 2 | 32→64 | construction | 3.670 | 1.876 |
| 2 | 32→64 | products | 6.621 | 2.727 |
| 2 | 32→64 | product_flops | 7.472 | 2.902 |
