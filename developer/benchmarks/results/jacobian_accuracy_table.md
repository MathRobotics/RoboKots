# Jacobian accuracy table

New reproducible experiment; not a reconstruction of the historical table.

Measured: 2026-09-16T04:43:31.075731+00:00

Environment: Linux-7.0.0-31-generic-x86_64-with-glibc2.39; Python 3.13.1; NumPy 2.4.6; JAX 0.10.1; [CpuDevice(id=0)]; float64.

Generated 7-DOF serial rigid arm, alternating x/y/z revolute axes. Last link and first actuated joint (subtree quantities). Exact model and motion are in JSON.

Gravity: [0.3, -0.4, -9.81]; seed: 20260916; samples: 1; motion distribution: N(0, 0.4²).

k is the ordinary kth time derivative of the named quantity, not the order of differentiation with respect to motion. k=0 is the undifferentiated quantity. Velocity/momentum require motion order k+2; force/torque require k+3. Jacobian columns are owner-major q, qdot, ... through that required order.

NumPy analytic vs central differences (public numerical=True, eps=1e-8) vs JAX forward/reverse AD (jacfwd/jacrev, no JIT). Dynamics AD uses the same pure function as jacobian_autodiff(); velocity AD uses its recursive velocity series. No Rust or JIT timing comparison is performed here.

Maximum absolute difference: max(abs(A-B)). Relative Frobenius: norm(A-B)/max(norm(B), 1e-30). Each cell takes the maximum over completed samples; the second named method is the relative-error denominator, not an exact oracle.

Completed cases: 40 / 40.

## numerical vs analytic: max_abs

| Quantity | k = 0 | k = 1 | k = 2 | k = 3 | k = 4 |
|---|---:|---:|---:|---:|---:|
| link velocity / local | 1.176e-08 | 8.202e-09 | 1.283e-08 | 7.605e-08 | 1.898e-07 |
| link momentum / local | 2.890e-08 | 5.766e-08 | 8.080e-08 | 3.913e-07 | 9.046e-07 |
| link momentum / world | 8.630e-08 | 1.234e-07 | 1.871e-07 | 1.090e-06 | 2.625e-06 |
| joint momentum / local | 1.557e-07 | 3.888e-07 | 5.151e-07 | 1.508e-06 | 5.786e-06 |
| joint momentum / world | 2.005e-07 | 6.516e-07 | 9.542e-07 | 2.276e-06 | 4.673e-06 |
| link force / local | 8.904e-07 | 4.209e-07 | 9.370e-07 | 1.301e-06 | 4.224e-06 |
| joint force / local | 7.394e-06 | 3.391e-06 | 4.923e-06 | 1.166e-05 | 3.066e-05 |
| joint torque | 5.053e-06 | 2.367e-06 | 2.552e-06 | 7.291e-06 | 2.045e-05 |

## numerical vs analytic: relative_frobenius

| Quantity | k = 0 | k = 1 | k = 2 | k = 3 | k = 4 |
|---|---:|---:|---:|---:|---:|
| link velocity / local | 5.259e-09 | 7.055e-09 | 7.315e-09 | 2.615e-08 | 2.875e-08 |
| link momentum / local | 3.749e-09 | 8.286e-09 | 8.228e-09 | 2.857e-08 | 3.118e-08 |
| link momentum / world | 5.016e-09 | 9.553e-09 | 8.220e-09 | 2.001e-08 | 2.518e-08 |
| joint momentum / local | 6.093e-09 | 1.663e-08 | 1.810e-08 | 2.633e-08 | 3.111e-08 |
| joint momentum / world | 5.873e-09 | 1.420e-08 | 1.292e-08 | 1.806e-08 | 2.281e-08 |
| link force / local | 1.740e-08 | 8.215e-09 | 1.540e-08 | 1.174e-08 | 2.040e-08 |
| joint force / local | 3.991e-08 | 1.626e-08 | 1.920e-08 | 2.362e-08 | 3.844e-08 |
| joint torque | 4.315e-08 | 1.748e-08 | 1.784e-08 | 3.042e-08 | 3.687e-08 |

## autodiff_forward vs analytic: max_abs

| Quantity | k = 0 | k = 1 | k = 2 | k = 3 | k = 4 |
|---|---:|---:|---:|---:|---:|
| link velocity / local | 3.608e-16 | 7.216e-16 | 1.665e-15 | 2.658e-15 | 8.438e-15 |
| link momentum / local | 1.776e-15 | 3.664e-15 | 8.882e-15 | 1.329e-14 | 4.263e-14 |
| link momentum / world | 3.553e-15 | 7.105e-15 | 1.776e-14 | 4.086e-14 | 1.421e-13 |
| joint momentum / local | 8.882e-15 | 1.421e-14 | 2.487e-14 | 5.951e-14 | 2.558e-13 |
| joint momentum / world | 1.421e-14 | 1.421e-14 | 2.132e-14 | 1.146e-13 | 3.268e-13 |
| link force / local | 1.421e-14 | 1.421e-14 | 2.487e-14 | 7.105e-14 | 3.908e-13 |
| joint force / local | 5.684e-14 | 1.705e-13 | 1.705e-13 | 3.979e-13 | 1.556e-12 |
| joint torque | 5.684e-14 | 2.842e-14 | 5.684e-14 | 1.705e-13 | 7.816e-13 |

## autodiff_forward vs analytic: relative_frobenius

| Quantity | k = 0 | k = 1 | k = 2 | k = 3 | k = 4 |
|---|---:|---:|---:|---:|---:|
| link velocity / local | 2.598e-16 | 4.295e-16 | 6.169e-16 | 8.829e-16 | 1.206e-15 |
| link momentum / local | 3.139e-16 | 5.364e-16 | 7.595e-16 | 1.084e-15 | 1.368e-15 |
| link momentum / world | 2.516e-16 | 4.017e-16 | 6.293e-16 | 8.880e-16 | 1.166e-15 |
| joint momentum / local | 2.829e-16 | 4.603e-16 | 7.471e-16 | 1.171e-15 | 1.745e-15 |
| joint momentum / world | 2.719e-16 | 3.307e-16 | 4.225e-16 | 8.595e-16 | 1.207e-15 |
| link force / local | 2.359e-16 | 2.976e-16 | 3.682e-16 | 5.590e-16 | 1.175e-15 |
| joint force / local | 2.710e-16 | 4.001e-16 | 4.564e-16 | 7.380e-16 | 1.558e-15 |
| joint torque | 3.904e-16 | 2.555e-16 | 3.659e-16 | 5.619e-16 | 1.138e-15 |

## autodiff_reverse vs analytic: max_abs

| Quantity | k = 0 | k = 1 | k = 2 | k = 3 | k = 4 |
|---|---:|---:|---:|---:|---:|
| link velocity / local | 4.441e-16 | 7.216e-16 | 1.554e-15 | 2.887e-15 | 8.438e-15 |
| link momentum / local | 1.665e-15 | 3.553e-15 | 7.994e-15 | 1.410e-14 | 4.441e-14 |
| link momentum / world | 3.553e-15 | 6.217e-15 | 1.599e-14 | 4.263e-14 | 1.279e-13 |
| joint momentum / local | 1.066e-14 | 1.421e-14 | 3.553e-14 | 5.684e-14 | 2.558e-13 |
| joint momentum / world | 1.421e-14 | 1.421e-14 | 2.842e-14 | 8.882e-14 | 2.913e-13 |
| link force / local | 1.421e-14 | 1.421e-14 | 1.954e-14 | 7.105e-14 | 3.695e-13 |
| joint force / local | 5.684e-14 | 1.137e-13 | 1.137e-13 | 4.547e-13 | 1.521e-12 |
| joint torque | 4.263e-14 | 5.684e-14 | 9.237e-14 | 2.842e-13 | 5.826e-13 |

## autodiff_reverse vs analytic: relative_frobenius

| Quantity | k = 0 | k = 1 | k = 2 | k = 3 | k = 4 |
|---|---:|---:|---:|---:|---:|
| link velocity / local | 2.745e-16 | 4.244e-16 | 6.166e-16 | 8.764e-16 | 1.228e-15 |
| link momentum / local | 2.779e-16 | 5.030e-16 | 7.328e-16 | 1.080e-15 | 1.417e-15 |
| link momentum / world | 2.782e-16 | 4.058e-16 | 5.601e-16 | 8.252e-16 | 1.141e-15 |
| joint momentum / local | 2.399e-16 | 4.107e-16 | 7.205e-16 | 1.130e-15 | 1.767e-15 |
| joint momentum / world | 3.148e-16 | 3.141e-16 | 4.447e-16 | 7.892e-16 | 1.207e-15 |
| link force / local | 2.510e-16 | 3.393e-16 | 3.632e-16 | 5.892e-16 | 1.170e-15 |
| joint force / local | 2.882e-16 | 3.343e-16 | 4.004e-16 | 7.488e-16 | 1.579e-15 |
| joint torque | 4.375e-16 | 4.912e-16 | 5.473e-16 | 7.231e-16 | 1.152e-15 |

## numerical vs autodiff_forward: max_abs

| Quantity | k = 0 | k = 1 | k = 2 | k = 3 | k = 4 |
|---|---:|---:|---:|---:|---:|
| link velocity / local | 1.176e-08 | 8.202e-09 | 1.283e-08 | 7.605e-08 | 1.898e-07 |
| link momentum / local | 2.890e-08 | 5.766e-08 | 8.080e-08 | 3.913e-07 | 9.046e-07 |
| link momentum / world | 8.630e-08 | 1.234e-07 | 1.871e-07 | 1.090e-06 | 2.625e-06 |
| joint momentum / local | 1.557e-07 | 3.888e-07 | 5.151e-07 | 1.508e-06 | 5.786e-06 |
| joint momentum / world | 2.005e-07 | 6.516e-07 | 9.542e-07 | 2.276e-06 | 4.673e-06 |
| link force / local | 8.904e-07 | 4.209e-07 | 9.370e-07 | 1.301e-06 | 4.224e-06 |
| joint force / local | 7.394e-06 | 3.391e-06 | 4.923e-06 | 1.166e-05 | 3.066e-05 |
| joint torque | 5.053e-06 | 2.367e-06 | 2.552e-06 | 7.291e-06 | 2.045e-05 |

## numerical vs autodiff_forward: relative_frobenius

| Quantity | k = 0 | k = 1 | k = 2 | k = 3 | k = 4 |
|---|---:|---:|---:|---:|---:|
| link velocity / local | 5.259e-09 | 7.055e-09 | 7.315e-09 | 2.615e-08 | 2.875e-08 |
| link momentum / local | 3.749e-09 | 8.286e-09 | 8.228e-09 | 2.857e-08 | 3.118e-08 |
| link momentum / world | 5.016e-09 | 9.553e-09 | 8.220e-09 | 2.001e-08 | 2.518e-08 |
| joint momentum / local | 6.093e-09 | 1.663e-08 | 1.810e-08 | 2.633e-08 | 3.111e-08 |
| joint momentum / world | 5.873e-09 | 1.420e-08 | 1.292e-08 | 1.806e-08 | 2.281e-08 |
| link force / local | 1.740e-08 | 8.215e-09 | 1.540e-08 | 1.174e-08 | 2.040e-08 |
| joint force / local | 3.991e-08 | 1.626e-08 | 1.920e-08 | 2.362e-08 | 3.844e-08 |
| joint torque | 4.315e-08 | 1.748e-08 | 1.784e-08 | 3.042e-08 | 3.687e-08 |

## numerical vs autodiff_reverse: max_abs

| Quantity | k = 0 | k = 1 | k = 2 | k = 3 | k = 4 |
|---|---:|---:|---:|---:|---:|
| link velocity / local | 1.176e-08 | 8.202e-09 | 1.283e-08 | 7.605e-08 | 1.898e-07 |
| link momentum / local | 2.890e-08 | 5.766e-08 | 8.080e-08 | 3.913e-07 | 9.046e-07 |
| link momentum / world | 8.630e-08 | 1.234e-07 | 1.871e-07 | 1.090e-06 | 2.625e-06 |
| joint momentum / local | 1.557e-07 | 3.888e-07 | 5.151e-07 | 1.508e-06 | 5.786e-06 |
| joint momentum / world | 2.005e-07 | 6.516e-07 | 9.542e-07 | 2.276e-06 | 4.673e-06 |
| link force / local | 8.904e-07 | 4.209e-07 | 9.370e-07 | 1.301e-06 | 4.224e-06 |
| joint force / local | 7.394e-06 | 3.391e-06 | 4.923e-06 | 1.166e-05 | 3.066e-05 |
| joint torque | 5.053e-06 | 2.367e-06 | 2.552e-06 | 7.291e-06 | 2.045e-05 |

## numerical vs autodiff_reverse: relative_frobenius

| Quantity | k = 0 | k = 1 | k = 2 | k = 3 | k = 4 |
|---|---:|---:|---:|---:|---:|
| link velocity / local | 5.259e-09 | 7.055e-09 | 7.315e-09 | 2.615e-08 | 2.875e-08 |
| link momentum / local | 3.749e-09 | 8.286e-09 | 8.228e-09 | 2.857e-08 | 3.118e-08 |
| link momentum / world | 5.016e-09 | 9.553e-09 | 8.220e-09 | 2.001e-08 | 2.518e-08 |
| joint momentum / local | 6.093e-09 | 1.663e-08 | 1.810e-08 | 2.633e-08 | 3.111e-08 |
| joint momentum / world | 5.873e-09 | 1.420e-08 | 1.292e-08 | 1.806e-08 | 2.281e-08 |
| link force / local | 1.740e-08 | 8.215e-09 | 1.540e-08 | 1.174e-08 | 2.040e-08 |
| joint force / local | 3.991e-08 | 1.626e-08 | 1.920e-08 | 2.362e-08 | 3.844e-08 |
| joint torque | 4.315e-08 | 1.748e-08 | 1.784e-08 | 3.042e-08 | 3.687e-08 |

## autodiff_reverse vs autodiff_forward: max_abs

| Quantity | k = 0 | k = 1 | k = 2 | k = 3 | k = 4 |
|---|---:|---:|---:|---:|---:|
| link velocity / local | 3.331e-16 | 4.441e-16 | 3.331e-16 | 8.882e-16 | 1.776e-15 |
| link momentum / local | 8.882e-16 | 1.332e-15 | 1.776e-15 | 3.553e-15 | 7.105e-15 |
| link momentum / world | 3.553e-15 | 3.553e-15 | 8.882e-15 | 1.421e-14 | 3.553e-14 |
| joint momentum / local | 7.105e-15 | 7.105e-15 | 1.066e-14 | 3.020e-14 | 5.684e-14 |
| joint momentum / world | 1.421e-14 | 1.421e-14 | 2.842e-14 | 5.684e-14 | 7.638e-14 |
| link force / local | 1.066e-14 | 1.421e-14 | 1.421e-14 | 5.684e-14 | 8.527e-14 |
| joint force / local | 7.341e-14 | 6.772e-14 | 1.137e-13 | 2.274e-13 | 3.695e-13 |
| joint torque | 3.553e-14 | 3.197e-14 | 8.527e-14 | 1.421e-13 | 2.274e-13 |

## autodiff_reverse vs autodiff_forward: relative_frobenius

| Quantity | k = 0 | k = 1 | k = 2 | k = 3 | k = 4 |
|---|---:|---:|---:|---:|---:|
| link velocity / local | 1.337e-16 | 1.642e-16 | 1.438e-16 | 2.346e-16 | 2.292e-16 |
| link momentum / local | 1.498e-16 | 1.711e-16 | 2.081e-16 | 2.084e-16 | 2.253e-16 |
| link momentum / world | 1.729e-16 | 1.854e-16 | 2.459e-16 | 2.840e-16 | 3.562e-16 |
| joint momentum / local | 2.149e-16 | 2.563e-16 | 3.187e-16 | 3.609e-16 | 4.182e-16 |
| joint momentum / world | 2.482e-16 | 2.359e-16 | 3.055e-16 | 3.445e-16 | 3.554e-16 |
| link force / local | 2.102e-16 | 2.278e-16 | 2.371e-16 | 2.881e-16 | 3.101e-16 |
| joint force / local | 3.421e-16 | 3.178e-16 | 4.177e-16 | 5.021e-16 | 5.738e-16 |
| joint torque | 3.380e-16 | 3.153e-16 | 3.295e-16 | 5.404e-16 | 5.065e-16 |
