# Dense dynamics complexity audit

Measured UTC: 2026-10-01T08:03:48.834417+00:00; source commit `32f0b0b9c25570c308cd36d8a7fbdfa178cbd555`.

Linux-7.0.0-34-generic-x86_64-with-glibc2.39; AMD Ryzen 9 7940HS w/ Radeon 780M Graphics; Python 3.13.1; NumPy 2.4.6; JAX 0.10.1; [CpuDevice(id=0)].

Current implementations, float64, serial revolute chains with alternating axes and a fixed tool. All active-joint torques, dense shape d × (p d). Gravity [0.2, -0.3, -9.81], seed 20261001 + DOF; p=3 for torque, p=5 for its second time derivative. Orders are held fixed during each DOF sweep. No batch/JVP/VJP. Rust analytic fallback is blocked; NumPy cannot invoke the Rust model.

Ready excludes state calculation; full includes motion import and state calculation and alternates two inputs to invalidate caches. Numerical uses central differences eps=1e-8 and includes all perturbed state evaluations; Rust scales each step by max(1, |x_i|), NumPy uses a fixed step. NumPy and Rust numerical are explicitly separated. AD eager uses pure jacfwd/jacrev without jit, independent of the public API's current jit=True default. JIT input is a runtime argument; warm/first include input conversion, synchronized execution and host output conversion. Construction/import costs are excluded; JIT first includes tracing and compilation. Compiled outputs are also checked at a second input.

Analytic: 7 samples ready / next even count full, two warmups. Numerical: 1 samples, no warmup (single measurements are noisy). Eager: 3 samples, one warmup. Warm JIT: 30 samples, two warmups. No competing benchmark process. Raw samples, errors, extension hash, environment and limits are in JSON.

## Findings

Compare each measured ratio with its fixed-order work model: doubling d predicts 8× for O(d³) and 4× for O(d²) arithmetic. Agreement in some methods and differences in others must be assessed separately; the work model is not an exact elapsed-time law.

Second time derivative, 8→16 DOF (same output and order):

| Method | Work model | 8 DOF ms | 16 DOF ms | Observed ratio |
|---|---|---:|---:|---:|
| numpy_full | O(d³) | 64.37274 | 189.71803 | 2.947× |
| rust_full | O(d²) | 0.80711 | 3.02160 | 3.744× |
| numpy_numerical | O(d³) | 5586.18679 | 42276.39697 | 7.568× |
| rust_numerical | O(d²) | 1.29541 | 3.46691 | 2.676× |
| forward_jit | O(d²) | 0.55597 | 1.93692 | 3.484× |
| reverse_jit | O(d²) | 0.43814 | 1.31214 | 2.995× |

## Median milliseconds

| Time derivative | DOF | Shape | Method | ms |
|---:|---:|---|---|---:|
| 0 | 4 | 4×12 | numpy_ready | 4.72764 |
| 0 | 4 | 4×12 | numpy_full | 13.28278 |
| 0 | 4 | 4×12 | numpy_numerical | 238.61629 |
| 0 | 4 | 4×12 | rust_ready | 0.01810 |
| 0 | 4 | 4×12 | rust_full | 0.03397 |
| 0 | 4 | 4×12 | rust_numerical | 0.30893 |
| 0 | 4 | 4×12 | forward_eager | 176.23278 |
| 0 | 4 | 4×12 | forward_jit_first | 539.75074 |
| 0 | 4 | 4×12 | forward_jit | 0.12434 |
| 0 | 4 | 4×12 | reverse_eager | 434.09729 |
| 0 | 4 | 4×12 | reverse_jit_first | 650.85565 |
| 0 | 4 | 4×12 | reverse_jit | 0.10526 |
| 0 | 8 | 8×24 | numpy_ready | 9.29305 |
| 0 | 8 | 8×24 | numpy_full | 29.12655 |
| 0 | 8 | 8×24 | numpy_numerical | 1724.46407 |
| 0 | 8 | 8×24 | rust_ready | 0.03728 |
| 0 | 8 | 8×24 | rust_full | 0.06261 |
| 0 | 8 | 8×24 | rust_numerical | 0.60582 |
| 0 | 8 | 8×24 | forward_eager | 314.53820 |
| 0 | 8 | 8×24 | forward_jit_first | 821.83464 |
| 0 | 8 | 8×24 | forward_jit | 0.20019 |
| 0 | 8 | 8×24 | reverse_eager | 789.51633 |
| 0 | 8 | 8×24 | reverse_jit_first | 1166.92492 |
| 0 | 8 | 8×24 | reverse_jit | 0.24895 |
| 0 | 16 | 16×48 | numpy_ready | 23.25502 |
| 0 | 16 | 16×48 | numpy_full | 78.24297 |
| 0 | 16 | 16×48 | numpy_numerical | 12976.22554 |
| 0 | 16 | 16×48 | rust_ready | 0.08278 |
| 0 | 16 | 16×48 | rust_full | 0.10387 |
| 0 | 16 | 16×48 | rust_numerical | 1.38157 |
| 0 | 16 | 16×48 | forward_eager | 596.65464 |
| 0 | 16 | 16×48 | forward_jit_first | 1650.19832 |
| 0 | 16 | 16×48 | forward_jit | 0.36373 |
| 0 | 16 | 16×48 | reverse_eager | 1522.75277 |
| 0 | 16 | 16×48 | reverse_jit_first | 2329.69152 |
| 0 | 16 | 16×48 | reverse_jit | 0.43363 |
| 0 | 32 | 32×96 | numpy_ready | 75.10864 |
| 0 | 32 | 32×96 | numpy_full | 249.33447 |
| 0 | 32 | 32×96 | rust_ready | 0.24335 |
| 0 | 32 | 32×96 | rust_full | 0.27412 |
| 2 | 4 | 4×20 | numpy_ready | 8.78788 |
| 2 | 4 | 4×20 | numpy_full | 27.43416 |
| 2 | 4 | 4×20 | numpy_numerical | 760.16683 |
| 2 | 4 | 4×20 | rust_ready | 0.22463 |
| 2 | 4 | 4×20 | rust_full | 0.24896 |
| 2 | 4 | 4×20 | rust_numerical | 0.55795 |
| 2 | 4 | 4×20 | forward_eager | 788.47895 |
| 2 | 4 | 4×20 | forward_jit_first | 1903.14464 |
| 2 | 4 | 4×20 | forward_jit | 0.27577 |
| 2 | 4 | 4×20 | reverse_eager | 2356.82799 |
| 2 | 4 | 4×20 | reverse_jit_first | 3195.02757 |
| 2 | 4 | 4×20 | reverse_jit | 0.30025 |
| 2 | 8 | 8×40 | numpy_ready | 19.57731 |
| 2 | 8 | 8×40 | numpy_full | 64.37274 |
| 2 | 8 | 8×40 | numpy_numerical | 5586.18679 |
| 2 | 8 | 8×40 | rust_ready | 0.79725 |
| 2 | 8 | 8×40 | rust_full | 0.80711 |
| 2 | 8 | 8×40 | rust_numerical | 1.29541 |
| 2 | 8 | 8×40 | forward_eager | 1435.26927 |
| 2 | 8 | 8×40 | forward_jit_first | 3807.66028 |
| 2 | 8 | 8×40 | forward_jit | 0.55597 |
| 2 | 8 | 8×40 | reverse_eager | 4269.38538 |
| 2 | 8 | 8×40 | reverse_jit_first | 6405.09413 |
| 2 | 8 | 8×40 | reverse_jit | 0.43814 |
| 2 | 16 | 16×80 | numpy_ready | 63.90722 |
| 2 | 16 | 16×80 | numpy_full | 189.71803 |
| 2 | 16 | 16×80 | numpy_numerical | 42276.39697 |
| 2 | 16 | 16×80 | rust_ready | 2.98438 |
| 2 | 16 | 16×80 | rust_full | 3.02160 |
| 2 | 16 | 16×80 | rust_numerical | 3.46691 |
| 2 | 16 | 16×80 | forward_eager | 2707.39532 |
| 2 | 16 | 16×80 | forward_jit_first | 7854.50663 |
| 2 | 16 | 16×80 | forward_jit | 1.93692 |
| 2 | 16 | 16×80 | reverse_eager | 8304.10485 |
| 2 | 16 | 16×80 | reverse_jit_first | 13149.85270 |
| 2 | 16 | 16×80 | reverse_jit | 1.31214 |
| 2 | 32 | 32×160 | numpy_ready | 274.46064 |
| 2 | 32 | 32×160 | numpy_full | 683.95054 |
| 2 | 32 | 32×160 | rust_ready | 11.77490 |
| 2 | 32 | 32×160 | rust_full | 11.85127 |

## Adjacent-size scaling

Effective exponent α = log(T₂/T₁)/log(d₂/d₁); doubling predicts 4× for quadratic work, 8× for cubic. These finite-size timing exponents do not prove asymptotic complexity. Compilation time is excluded from this table.

| Derivative | DOFs | Method | Time ratio | Effective α |
|---:|---|---|---:|---:|
| 0 | 4→8 | forward_eager | 1.785 | 0.836 |
| 0 | 4→8 | forward_jit | 1.610 | 0.687 |
| 0 | 4→8 | numpy_full | 2.193 | 1.133 |
| 0 | 4→8 | numpy_numerical | 7.227 | 2.853 |
| 0 | 4→8 | numpy_ready | 1.966 | 0.975 |
| 0 | 4→8 | reverse_eager | 1.819 | 0.863 |
| 0 | 4→8 | reverse_jit | 2.365 | 1.242 |
| 0 | 4→8 | rust_full | 1.843 | 0.882 |
| 0 | 4→8 | rust_numerical | 1.961 | 0.972 |
| 0 | 4→8 | rust_ready | 2.059 | 1.042 |
| 0 | 8→16 | forward_eager | 1.897 | 0.924 |
| 0 | 8→16 | forward_jit | 1.817 | 0.862 |
| 0 | 8→16 | numpy_full | 2.686 | 1.426 |
| 0 | 8→16 | numpy_numerical | 7.525 | 2.912 |
| 0 | 8→16 | numpy_ready | 2.502 | 1.323 |
| 0 | 8→16 | reverse_eager | 1.929 | 0.948 |
| 0 | 8→16 | reverse_jit | 1.742 | 0.801 |
| 0 | 8→16 | rust_full | 1.659 | 0.730 |
| 0 | 8→16 | rust_numerical | 2.280 | 1.189 |
| 0 | 8→16 | rust_ready | 2.220 | 1.151 |
| 0 | 16→32 | numpy_full | 3.187 | 1.672 |
| 0 | 16→32 | numpy_ready | 3.230 | 1.691 |
| 0 | 16→32 | rust_full | 2.639 | 1.400 |
| 0 | 16→32 | rust_ready | 2.940 | 1.556 |
| 2 | 4→8 | forward_eager | 1.820 | 0.864 |
| 2 | 4→8 | forward_jit | 2.016 | 1.012 |
| 2 | 4→8 | numpy_full | 2.346 | 1.230 |
| 2 | 4→8 | numpy_numerical | 7.349 | 2.877 |
| 2 | 4→8 | numpy_ready | 2.228 | 1.156 |
| 2 | 4→8 | reverse_eager | 1.811 | 0.857 |
| 2 | 4→8 | reverse_jit | 1.459 | 0.545 |
| 2 | 4→8 | rust_full | 3.242 | 1.697 |
| 2 | 4→8 | rust_numerical | 2.322 | 1.215 |
| 2 | 4→8 | rust_ready | 3.549 | 1.827 |
| 2 | 8→16 | forward_eager | 1.886 | 0.916 |
| 2 | 8→16 | forward_jit | 3.484 | 1.801 |
| 2 | 8→16 | numpy_full | 2.947 | 1.559 |
| 2 | 8→16 | numpy_numerical | 7.568 | 2.920 |
| 2 | 8→16 | numpy_ready | 3.264 | 1.707 |
| 2 | 8→16 | reverse_eager | 1.945 | 0.960 |
| 2 | 8→16 | reverse_jit | 2.995 | 1.582 |
| 2 | 8→16 | rust_full | 3.744 | 1.904 |
| 2 | 8→16 | rust_numerical | 2.676 | 1.420 |
| 2 | 8→16 | rust_ready | 3.743 | 1.904 |
| 2 | 16→32 | numpy_full | 3.605 | 1.850 |
| 2 | 16→32 | numpy_ready | 4.295 | 2.103 |
| 2 | 16→32 | rust_full | 3.922 | 1.972 |
| 2 | 16→32 | rust_ready | 3.946 | 1.980 |

## Agreement

All measured methods are checked against NumPy analytic output, a comparison reference rather than an exact oracle. First, warm and eager AD outputs are included.

| Methods | Maximum absolute difference | Maximum relative Frobenius error |
|---|---:|---:|
| numerical | 3.55342e-05 | 7.17594e-08 |
| analytic/AD | 1.71894e-10 | 2.81874e-14 |

## Interpretation and implementation cost

For fixed p and L=O(d) links, the scalar NumPy analytic path builds whole-body dense matrices and multiplies them (conventional dense multiplication O(d³)); Rust analytical propagates dense derivatives along the tree (O(d²)); both AD directions require O(d²) arithmetic for this all-joint dense output (forward has pd tangent directions, reverse d output directions; a primal evaluation is O(d)). These are arithmetic-work models, not exact wall-clock time laws.

NumPy numerical runs a separate finite-difference pass for every selected joint. The present central-difference helper also evaluates the unperturbed state once: d(2pd+1) full state evaluations, each O(d), hence O(d³). Rust numerical shares 2pd full state evaluations across all selected outputs, hence O(d²). Output allocation alone requires Ω(pd²) space/work for a dense d×pd matrix; this lower bound alone does not classify the implementation's full computation.

The observed finite-size exponents should be read separately for each method and scope. Python dispatch, small-array work, allocation, caches, vectorized tangent operations, compiler optimization and CPU utilization contribute different terms to elapsed time. NumPy analytic and AD can therefore appear below the asymptotic arithmetic exponent at these sizes. A non-cubic/non-quadratic timing exponent is not evidence that the underlying dense algorithm has a different asymptotic complexity. JIT compilation is a separate cost and its scaling is not inferred from warm execution.

Historical dynamics_autodiff reports used earlier dispatch, platforms and AD defaults. They cannot be combined with this run as a single scaling curve. Historical jacobian_resources selects individual outputs, whereas this run selects all joints. The saved rust_dense_dynamics run varies time derivative order as well as DOF; only equal-order rows can be used to estimate DOF scaling. This audit does not establish high-order p asymptotics or large-d asymptotics for AD.
