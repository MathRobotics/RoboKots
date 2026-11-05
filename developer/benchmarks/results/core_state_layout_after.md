# Core state layout comparison

Same model, float64 motion order 4, seed 71, world gravity [0, 0, -9.81].
5 warmups; 30 samples × 5 calls; medians in ms/call.
Before/after run in separate processes. Small timing changes may be measurement noise.

| Workload | Before ms | After ms | After / before |
| --- | ---: | ---: | ---: |
| numpy/single/import_and_dynamics | 1.851383 | 1.904438 | 1.029 |
| numpy/single/read_values | 0.013358 | 0.012833 | 0.961 |
| numpy/single/export | 0.066267 | 0.065767 | 0.992 |
| numpy/single/cache_hit | 0.000817 | 0.000821 | 1.005 |
| numpy/batch_2x3/import_and_dynamics | 2.834992 | 2.959033 | 1.044 |
| numpy/batch_2x3/read_values | 0.014137 | 0.014267 | 1.009 |
| numpy/batch_2x3/export | 0.079842 | 0.079642 | 0.997 |
| rust/single/import_and_dynamics | 0.008350 | 0.008167 | 0.978 |
| rust/single/read_values | 0.005283 | 0.005246 | 0.993 |
| rust/single/export | 0.065287 | 0.066000 | 1.011 |
| rust/batch_2x3/import_and_dynamics | 0.016050 | 0.016054 | 1.000 |
| rust/batch_2x3/read_values | 0.006112 | 0.006204 | 1.015 |
| rust/batch_2x3/export | 0.077146 | 0.075413 | 0.978 |
| list_batch/read_values | 0.067288 | 0.065700 | 0.976 |
| list_batch/read_parts | 0.063988 | 0.063500 | 0.992 |
| list_batch/export | 0.651925 | 0.656038 | 1.006 |

Maximum absolute differences and relative Frobenius errors:

| Output | Max abs | Relative Frobenius |
| --- | ---: | ---: |
| numpy/single | 0 | 0 |
| numpy/batch_2x3 | 0 | 0 |
| rust/single | 0 | 0 |
| rust/batch_2x3 | 0 | 0 |
| list_batch | 0 | 0 |

Import/conversion included only in import_and_dynamics. Reads/export use computed states. First timing is first invocation of each workload, not process import time. List batch is explicitly assembled from NumPy scalar states. No JAX/JIT measurements.

Environment and individual timings are recorded in the JSON files.
