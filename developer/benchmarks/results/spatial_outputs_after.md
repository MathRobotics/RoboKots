# Local mixed-output Rust comparison

5 warmups; 30 samples × 5 calls; medians in ms/call.
Separate processes with the same workload and thread settings. Small differences may be measurement noise.

| Workload | Before ms | After ms | Speedup |
| --- | ---: | ---: | ---: |
| numpy/single/import_and_dynamics | 1.879421 | 1.832604 | 1.03x |
| numpy/single/dense | 5.683746 | 4.981104 | 1.14x |
| numpy/single/jvp | 3.630358 | 3.446921 | 1.05x |
| numpy/single/vjp | 3.701754 | 3.483725 | 1.06x |
| numpy/batch2/import_and_dynamics | 2.571413 | 2.425179 | 1.06x |
| numpy/batch2/dense | 8.023717 | 7.081238 | 1.13x |
| numpy/batch2/jvp | 4.751950 | 3.941033 | 1.21x |
| numpy/batch2/vjp | 22.186050 | 20.589213 | 1.08x |
| rust/single/import_and_dynamics | 0.008467 | 0.008321 | 1.02x |
| rust/single/dense | 0.111758 | 0.112392 | 0.99x |
| rust/single/jvp | 0.015846 | 0.016208 | 0.98x |
| rust/single/vjp | 0.015275 | 0.015612 | 0.98x |
| rust/batch2/import_and_dynamics | 0.011412 | 0.011479 | 0.99x |
| rust/batch2/dense | 0.221888 | 0.215287 | 1.03x |
| rust/batch2/jvp | 0.028442 | 0.027704 | 1.03x |
| rust/batch2/vjp | 0.026092 | 0.026662 | 0.98x |

| Output | Max abs difference | Relative Frobenius error |
| --- | ---: | ---: |
| numpy/single/dense | 0 | 0 |
| numpy/single/jvp | 0 | 0 |
| numpy/single/vjp | 0 | 0 |
| numpy/single/state | 0 | 0 |
| numpy/batch2/dense | 0 | 0 |
| numpy/batch2/jvp | 0 | 0 |
| numpy/batch2/vjp | 0 | 0 |
| numpy/batch2/state | 0 | 0 |
| rust/single/dense | 0 | 0 |
| rust/single/jvp | 0 | 0 |
| rust/single/vjp | 0 | 0 |
| rust/single/state | 0 | 0 |
| rust/batch2/dense | 0 | 0 |
| rust/batch2/jvp | 0 | 0 |
| rust/batch2/vjp | 0 | 0 |
| rust/batch2/state | 0 | 0 |

| Rust vs NumPy | Max abs difference | Relative Frobenius error |
| --- | ---: | ---: |
| rust/single/dense | 7.11e-15 | 6.53e-16 |
| rust/single/jvp | 6.22e-15 | 4.97e-16 |
| rust/single/vjp | 1.07e-14 | 6e-16 |
| rust/single/state | 6.66e-16 | 3.34e-16 |
| rust/batch2/dense | 7.11e-15 | 6e-16 |
| rust/batch2/jvp | 7.11e-15 | 6.17e-16 |
| rust/batch2/vjp | 7.55e-15 | 4.54e-16 |
| rust/batch2/state | 6.66e-16 | 2.3e-16 |

Public dense/JVP/VJP start with computed states; any internal recurrence recomputation is timed. Import/conversion plus state building is measured separately. NumPy is a reference, not an exact solution. Local regression: both runs use the unified selected Rust recurrence. No JAX/JIT timing. First calls follow shape discovery, not cold process startup.

Full model/seed/output selection, raw samples, numerical outputs, direct-vs-dense errors and extension hashes are in the JSON files.
