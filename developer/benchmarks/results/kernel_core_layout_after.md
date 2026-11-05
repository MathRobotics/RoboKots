# Kernel layout comparison

5 warmups, 30 samples × 5 calls. Medians in ms/call.
The baseline is the saved post-consolidation measurement from a previous session.
Before/after are separate processes; timing differences may include changes in machine conditions between sessions.

| Workload | Before ms | After ms | After / before |
| --- | ---: | ---: | ---: |
| numpy/single/import_and_dynamics | 1.845204 | 1.915821 | 1.038 |
| numpy/single/dense | 4.850742 | 4.953888 | 1.021 |
| numpy/single/jvp | 3.406333 | 3.444171 | 1.011 |
| numpy/single/vjp | 3.431317 | 3.513738 | 1.024 |
| numpy/batch2/import_and_dynamics | 2.420642 | 2.466183 | 1.019 |
| numpy/batch2/dense | 6.926821 | 7.141808 | 1.031 |
| numpy/batch2/jvp | 3.845117 | 3.981092 | 1.035 |
| numpy/batch2/vjp | 20.482496 | 20.991029 | 1.025 |
| rust/single/import_and_dynamics | 0.008237 | 0.008537 | 1.036 |
| rust/single/dense | 2.934175 | 2.984975 | 1.017 |
| rust/single/jvp | 2.171496 | 2.206071 | 1.016 |
| rust/single/vjp | 0.035583 | 0.035554 | 0.999 |
| rust/batch2/import_and_dynamics | 0.011592 | 0.011629 | 1.003 |
| rust/batch2/dense | 3.871200 | 3.967567 | 1.025 |
| rust/batch2/jvp | 1.961150 | 2.049983 | 1.045 |
| rust/batch2/vjp | 0.056621 | 0.056083 | 0.991 |

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
