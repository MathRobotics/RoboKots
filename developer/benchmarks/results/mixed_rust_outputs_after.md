# Local mixed-output Rust comparison

5 warmups; 30 samples × 5 calls; medians in ms/call.
Separate processes with the same workload and thread settings. Small differences may be measurement noise.

| Workload | Before ms | After ms | Speedup |
| --- | ---: | ---: | ---: |
| numpy/single/import_and_dynamics | 1.850821 | 1.946496 | 0.95x |
| numpy/single/dense | 5.133054 | 5.112342 | 1.00x |
| numpy/single/jvp | 3.473221 | 3.526929 | 0.98x |
| numpy/single/vjp | 3.611883 | 3.674583 | 0.98x |
| numpy/batch2/import_and_dynamics | 2.408021 | 2.550492 | 0.94x |
| numpy/batch2/dense | 7.369004 | 7.382108 | 1.00x |
| numpy/batch2/jvp | 3.935717 | 4.068992 | 0.97x |
| numpy/batch2/vjp | 20.859167 | 21.849604 | 0.95x |
| rust/single/import_and_dynamics | 0.008221 | 0.008712 | 0.94x |
| rust/single/dense | 3.047600 | 0.114929 | 26.52x |
| rust/single/jvp | 2.234696 | 0.015725 | 142.11x |
| rust/single/vjp | 0.035321 | 0.014971 | 2.36x |
| rust/batch2/import_and_dynamics | 0.011354 | 0.012050 | 0.94x |
| rust/batch2/dense | 4.008504 | 0.213729 | 18.76x |
| rust/batch2/jvp | 2.052817 | 0.027563 | 74.48x |
| rust/batch2/vjp | 0.056171 | 0.025971 | 2.16x |

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
| rust/single/dense | 4e-15 | 4.61e-16 |
| rust/single/jvp | 6.22e-15 | 6.21e-16 |
| rust/single/vjp | 1.5e-15 | 9.32e-17 |
| rust/single/state | 0 | 0 |
| rust/batch2/dense | 4e-15 | 4.97e-16 |
| rust/batch2/jvp | 5.83e-15 | 6.6e-16 |
| rust/batch2/vjp | 1.78e-15 | 9.08e-17 |
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

Public dense/JVP/VJP start with computed states; any internal recurrence recomputation is timed. Import/conversion plus state building is measured separately. NumPy is a reference, not an exact solution. Before: Rust state, Python dense/JVP, composed Rust VJP. After: all mixed derivatives use the selected Rust recurrence. No JAX/JIT timing. First calls follow shape discovery, not cold process startup.

Full model/seed/output selection, raw samples, numerical outputs, direct-vs-dense errors and extension hashes are in the JSON files.
