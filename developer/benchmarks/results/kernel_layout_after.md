# Kernel layout comparison

5 warmups, 30 samples × 5 calls. Medians in ms/call.
Before/after are separate processes; small differences may be measurement noise.

| Workload | Before ms | After ms | After / before |
| --- | ---: | ---: | ---: |
| numpy/single/import_and_dynamics | 1.834596 | 1.845204 | 1.006 |
| numpy/single/dense | 4.904908 | 4.850742 | 0.989 |
| numpy/single/jvp | 3.443621 | 3.406333 | 0.989 |
| numpy/single/vjp | 3.436742 | 3.431317 | 0.998 |
| numpy/batch2/import_and_dynamics | 2.470846 | 2.420642 | 0.980 |
| numpy/batch2/dense | 6.985883 | 6.926821 | 0.992 |
| numpy/batch2/jvp | 3.941608 | 3.845117 | 0.976 |
| numpy/batch2/vjp | 20.574367 | 20.482496 | 0.996 |
| rust/single/import_and_dynamics | 0.008288 | 0.008237 | 0.994 |
| rust/single/dense | 2.915379 | 2.934175 | 1.006 |
| rust/single/jvp | 2.183958 | 2.171496 | 0.994 |
| rust/single/vjp | 0.035717 | 0.035583 | 0.996 |
| rust/batch2/import_and_dynamics | 0.011792 | 0.011592 | 0.983 |
| rust/batch2/dense | 3.867325 | 3.871200 | 1.001 |
| rust/batch2/jvp | 1.956638 | 1.961150 | 1.002 |
| rust/batch2/vjp | 0.058071 | 0.056621 | 0.975 |

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

State computation and input conversion included only in import_and_dynamics. Dense/JVP/VJP start with computed states and use public dispatch; internal scalar-state fallback work is included. Products are checked with the dense facade path disabled outside timing. Rust labels identify state storage: dense/JVP use Python analytic derivatives over Rust views; VJP uses Rust direct kernels for this selection. First calls follow state setup and product-shape discovery; import/JIT costs are not measured.

Model, gravity, state selections, seed, environment, raw timings, outputs and direct-vs-dense checks are in the JSON files.
