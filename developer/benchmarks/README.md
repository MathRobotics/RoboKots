# Developer Benchmarks

These scripts are for local performance investigation and are not part of the
normal RoboKots runtime path.

## State Cache Scope Experiment

```bash
.venv/bin/python -m developer.benchmarks.state_cache_scope --build /tmp/robokots-cache-probe
.venv/bin/python -u -m developer.benchmarks.state_cache_scope --extension /tmp/robokots-cache-probe/probe.so
```

Builds an isolated Rust extension from the current sources, without installing
it or changing production kernels. Compares existing recomputation, eager full
state, lazy completion of the order-3 zero-gravity state, and lazy full
recomputation. Measures dynamics, first/repeated JVP/VJP, and numeric buffer
capacities independently. Build uses the existing Cargo dependency cache
(`--offline`). See [the report](results/state_cache_scope.md) and
[raw timings/environment/source hashes](results/state_cache_scope.json).
Benchmark-only Rust additions are in `cache_probe/`; their prepared kernels
reuse the primal state without recomputing dynamics or kinematics.

## High-order Production Comparison

Preserve the old release extension before rebuilding, then compare two binaries:

```bash
.venv/bin/python -m developer.benchmarks.high_order_production \
  --baseline /path/to/baseline.so --optimized /path/to/optimized.so
```

Each binary runs in a fresh process. Three rounds alternate binary order, with
5 warmups and 30 samples of 5 evaluations each. Measures raw persistent-workspace
updates and public `import_motions` + `dynamics`, with cache invalidation on every
call; includes order 3 as a regression check. Checks all local link/joint
momentum and force derivatives against the baseline. Results and binary/source
hashes are saved in [the report](results/high_order_production.md) and
[JSON](results/high_order_production.json). These are production API timings;
`high_order_speed` above remains an isolated kernel experiment and reconstructs
the pre-optimization baseline inside its temporary source copy.

## High-order Dynamics Speed Experiment

```bash
.venv/bin/python -m developer.benchmarks.high_order_speed --build /tmp/robokots-high-order-probe
.venv/bin/python -u -m developer.benchmarks.high_order_speed --extension /tmp/robokots-high-order-probe/probe.so
```

Uses an isolated extension to compare orders 4/5/6/8: direct spatial-velocity
series propagation, reuse of wrench transport blocks for gravity, and their
combination. Preserves complete state outputs and checks all semantic buffers
against existing Rust plus independent NumPy momentum/force values. Measures
kinematics and full dynamics separately, without changing production dispatch.
See [the report](results/high_order_speed.md) and
[raw results/environment/source hashes](results/high_order_speed.json).
Prototype sources are in `high_order_probe/`.

## Runtime Benchmark

```bash
uv run python -m developer.benchmarks.runtime
```

Measures kinematics, dynamics, Jacobian, numerical Jacobian, Jacobian
vector/matrix products, Jacobian-transpose vector/matrix products, and cached
state update runtime on the sample model.

## State Dictionary Separation

```bash
.venv/bin/python -m developer.benchmarks.state_dictionary_compare
```

Compares commit `6ec3bfb` with the current working tree in sequential, isolated
Python workers using the same Rust extension and dependencies. Separates default
state calculation, explicit dictionary-free calculation, dictionary export, and
state-precomputed dense Jacobians. Medians, first-call timings, raw samples,
environment, and output differences are saved in
[the report](results/state_dictionary.md) and [raw JSON](results/state_dictionary.json).

## Batch API Benchmark

```bash
uv run python -m developer.benchmarks.batch_api
```

Compares batched API runtime with repeated single-sample execution for
kinematics, dynamics, Jacobian, and matvec operations on the sample model.

## High-DOF RoboKots Baseline

```bash
uv run python -m developer.benchmarks.robokots_baseline
```

This measures RoboKots batched execution against repeated single-sample
execution on generated high-DOF models.

## Jacobian Developer Utilities

### AD including time derivatives (evaluation only)

`time_autodiff.py` provides `make_time_ad_value(robot, state, order, gravity)`.
Unlike the production JAX implementation, it does **not** use high-order
time-derivative recurrences or binomial product formulas. It uses independent
motion entries `x=(q,qdot,qddot,...)`, with no trajectory or explicit time
parameter. The total derivative is `D_t f(x)=JVP(f,x,shift(x))` where
`shift(x)=(qdot,qddot,...)`. Repeated differentiation includes the dependence
of this direction on x (otherwise acceleration terms would be lost).
The final direction slot is zero-padded; required input-order checks ensure
that unavailable higher derivatives cannot affect the requested output.
Time AD obtains body velocity from
forward-kinematic poses, and momentum rate from momentum. The physical
force balance, gravity, spatial transformations and subtree accumulation are
still explicit physics equations. Additional time derivatives are nested
total-time JVPs; the outer motion Jacobian uses `jacfwd` or `jacrev`.
In particular, `time_reverse_*` means **outer reverse AD over inner forward
time AD**, not reverse mode for both. The outer Jacobian is dense.

This is a benchmark implementation, not a public `Kots` API. It supports
fixed-base rigid trees with fixed/revolute/prismatic joints, local link
velocity and its derivatives, local/world link/joint momentum and force,
and joint torque. It shares elementary spatial algebra/model constants,
but does not call the existing time-series dynamics implementation. No
custom analytic derivative rules, graph coloring or explicit branch pruning
are used.

`make_fk_value(robot, link_name, derivative_order, order)` also exposes FK
pose time derivatives directly, returning a 4x4 matrix. Its outer Jacobian
has shape `(4,4,dof*order)`: this is an **elementwise pose Jacobian**, not the
six-dimensional tangent Jacobian returned by `Kots.jacobian(...frame...)`.
For ID, use `make_time_ad_value` with a joint `torque` / `torque_diffN` state.

```python
import jax
from robokots.kots import StateType
from developer.benchmarks.time_autodiff import make_fk_value, make_time_ad_value

jax.config.update("jax_enable_x64", True)
# kots has already loaded a rigid model and sufficient motion orders.
fk_acceleration = make_fk_value(kots.robot_, kots.link_name_list()[-1],
                                derivative_order=2, order=3)
pose_second_derivative = fk_acceleration(kots.motion(3))
pose_second_derivative_jacobian = jax.jacfwd(fk_acceleration)(kots.motion(3))

joint = next(j.name for j in kots.robot_.joints if j.dof)
id_rate = make_time_ad_value(kots.robot_, StateType("joint", joint, "torque_diff1"),
                             order=4, gravity=[0, 0, -9.81])
torque_rate = id_rate(kots.motion(4))
torque_rate_jacobian = jax.jit(jax.jacrev(id_rate))(kots.motion(4))
```

Example comparing the two AD formulations on the same 7-DOF inputs:

```bash
.venv/bin/python -u -m developer.benchmarks.jacobian_resources \
  --derivatives 0 --rows 0 1 5 7 \
  --methods numpy_full forward_eager reverse_eager forward_jit reverse_jit \
    time_forward_eager time_reverse_eager time_forward_jit time_reverse_jit \
  --output developer/benchmarks/results/time_autodiff_resources
```

Use `--derivatives 0 1 2 3 4` to extend the comparison. Nested AD can grow
very expensive at high derivative orders; begin with a small model and use
`--timeout` to bound **each worker's entire lifetime**, including validation.
Timeouts/errors are recorded rather than replaced by recurrence results.
Peak RSS includes compilation. The ordinary resource benchmark defaults
remain unchanged; `time_*` methods are opt-in.

To check torque through the fourth time derivative on the 7-DOF model,
including both output values and their motion Jacobians:

```bash
.venv/bin/python -u -m developer.benchmarks.jacobian_resources \
  --dof 7 --derivatives 0 1 2 3 4 --rows 7 \
  --methods numpy_full forward_jit reverse_jit time_forward_jit time_reverse_jit \
  --repeats 5 --warmup 1 --timeout 600 --check-values \
  --output developer/benchmarks/results/time_autodiff_torque7
```

This selects the first actuated joint's scalar torque, including the
seven-joint downstream dynamics, not all seven output torques at once.
At k=4 the Jacobian is 1x49 and the inputs extend through the sixth time
derivative of q. Timing/memory cover **Jacobian generation**, including
the inner time AD; `--check-values` additionally checks the differentiated
torque values outside those measurements. Two independently seeded motion
inputs are compared against NumPy analytic. The timeout includes all worker
setup, measurement, and validation work.

### Time AD starting from ordinary inverse dynamics

The separate `id_time_autodiff.py` implementation preserves the earlier
FK-based experiment unchanged. `make_id_time_ad_value(robot, state, order,
gravity)` takes one joint `torque` / `torque_diffN` selection. Its base function
is the existing JAX ID at **motion order 3**, using only `q,qdot,qddot` and
undifferentiated torque. Higher time derivatives are repeated total-time JVPs;
outer `jacfwd`/`jacrev` produces the motion Jacobian. No high-order torque
recurrence is used. Methods prefixed `id_time_` are opt-in.

```bash
.venv/bin/python -u -m developer.benchmarks.jacobian_resources \
  --dof 7 --derivatives 0 1 2 3 4 --rows 7 \
  --methods numpy_full forward_jit reverse_jit id_time_forward_jit id_time_reverse_jit \
  --repeats 5 --warmup 1 --timeout 600 --check-values \
  --output developer/benchmarks/results/id_time_autodiff_torque7
.venv/bin/python -m developer.benchmarks.compare_time_ad_formulations
```

The comparison reads the preserved `time_autodiff_torque7.json` and the new
`id_time_autodiff_torque7.json`, verifies matching inputs and environment, and
writes `time_ad_formulations.md` / `.json`. Prior timings are explicitly
labelled as previous-run measurements; old timeouts stay missing, not filled
with a new method's timings. Values and Jacobians are also compared directly
between the two formulations at both saved inputs. Source reports are retained.

### Explicit CMTM matrices followed by AD

`cmtm_autodiff.py` is an evaluation-only JAX implementation of spatial CMTM
algebra. It constructs lower block-Toeplitz inverse-adjoint and wrench
transforms, propagates factorial-normalized velocity/momentum/force
coefficients, and converts the selected torque coefficient back to an ordinary
time derivative. **Only the outer motion Jacobian uses AD**; higher time
derivatives do not use nested JVPs. The matrices use dense storage, not a sparse
AD solver or graph coloring. This is not a call to mathrobo's NumPy-cached CMTM
object: matrix coefficients are tested against that implementation independently.

The existing `forward_jit`/`reverse_jit` implementation instead operates on
ordinary derivative series using binomial-weighted recurrences, without
constructing the full CMTM matrices. Label these implementations **derivative
coefficients + AD** and **explicit CMTM matrices + AD**, respectively. Both
compute high-order time derivatives analytically and apply AD only to the outer
motion Jacobian. Both also use recurrences to construct transform coefficients:
"recurrence versus CMTM" is not a distinction between mathematical methods.
They represent the same high-order algebra using different normalization,
storage and evaluation arrangements. Performance differences reflect these
implementations and JAX compilation, not a mathematical advantage of CMTM.
Block-matrix construction is not assumed faster. Existing CLI method identifiers
are retained so saved measurements and reproduction commands remain consistent.
The new methods support a single local joint torque/torque derivative on rigid,
fixed-base trees with fixed, revolute and prismatic joints, for evaluation only.

```bash
.venv/bin/python -u -m developer.benchmarks.jacobian_resources \
  --dof 7 --derivatives 0 1 2 3 4 --rows 7 \
  --methods numpy_full forward_jit reverse_jit id_time_forward_jit id_time_reverse_jit cmtm_forward_jit cmtm_reverse_jit \
  --repeats 5 --warmup 1 --timeout 600 --check-values \
  --output developer/benchmarks/results/cmtm_ad_torque7
```

This measures the Jacobian of the **first actuated joint's torque**, not all
seven torques: at derivative order 4 it has shape `(1, 49)`. All seven
formulations run afresh under the same conditions. Reports include initial JIT
cost, five-call warm medians, process peak RSS including compilation, and
value/Jacobian errors at two inputs. Earlier FK-based time-AD results remain
in their original reports; they are not silently mixed into this fresh run.

For isolated CPU timing and process peak-memory measurements:

```bash
.venv/bin/python -u -m developer.benchmarks.jacobian_resources
```

Defaults: the same 7-DOF model/seed as the accuracy table, local link velocity,
link momentum, link force and first-actuated-joint torque, at `k=0,4`.
Each case/method runs sequentially in a fresh process. NumPy/Rust-requested
analytic paths have full-state and cached-state variants; numerical central
differences and forward/reverse JAX AD (eager and JIT) are compared too.
Explicit local selections may bypass Rust derivative fast paths, including
torque: Rust labels are state-backend requests, not pure-Rust kernel timings.
First-call/JIT compilation cost is separate from warm medians (five calls;
numerical one call; one additional warmup). JAX execution is synchronized.
Memory is CPU process-lifetime peak RSS and growth from a pre-call high-water
baseline, including runtime/compiler caches, **not per-call allocation or
warm-only memory**. Cached-state preparation is excluded from growth.
Two motion inputs are checked against NumPy analytic, outside measurements.
JSON and Markdown are written to `results/jacobian_resources.*`.
Use `--derivatives 0 1 2 3 4 --rows 0 1 2 3 4 5 6 7` for the full accuracy-table
coverage, `--repeats 20` for more timing samples, or `--help` for other options.
Failures/timeouts are recorded explicitly and cause a nonzero exit status.

For a table-style accuracy comparison of velocity, link/joint momentum
(local/world), link/joint force (local), and joint torque, run:

```bash
.venv/bin/python -u -m developer.benchmarks.jacobian_accuracy_table
```

This new experiment is **not a reproduction of the historical paper table**.
The default is a generated 7-DOF serial arm, one seeded random motion, and
all ordinary time derivative orders `k=0,1,2,3,4` (velocity through crackle).
It compares NumPy analytic, central numerical differences (`eps=1e-8`), and
float64 JAX forward/reverse AD (`jacfwd`/`jacrev`) without JIT. Twelve tables report pairwise maximum absolute
and relative Frobenius differences. This is an accuracy benchmark, not a timing
benchmark; Rust/JIT performance remains in `dynamics_autodiff_compare` below.
The last link and first actuated joint are selected explicitly, so joint
quantities include the downstream subtree. The old table's tilde notation is
not assumed to mean local/world.

Results are saved to `results/jacobian_accuracy_table.md` and `.json`, including
the exact generated model, input motion, gravity, seed, environment, revision,
working-tree status, per-sample errors, and Jacobian shapes. Multiple samples
are aggregated by taking the maximum per cell. Use `--samples 3` for more
inputs, `--dof 2 --max-derivative 0 --output /tmp/jacobian_smoke` for a small
run, or `--help` for other options. Each completed derivative order is saved;
interrupted reports explicitly show missing cells as `pending`.

The public dynamics API also supports both modes:
`kots.jacobian_autodiff(state, mode="forward")` (default) and
`kots.jacobian_autodiff(state, mode="reverse")`. Both support batching and
`list_output=True`, return NumPy arrays, and do not use JIT. The table benchmark
additionally differentiates the internal velocity series; the public AD API
remains limited to dynamics quantities.

```bash
uv run python -m developer.benchmarks.jacobian_compare
uv run python -m developer.benchmarks.jacobian_dof_sweep
uv run python -m developer.benchmarks.jacobian_transpose_matvec_compare
uv run python -u -m developer.benchmarks.dynamics_autodiff_compare
```

These compare analytic, numerical, and JAX autodiff Jacobians, including a DOF
sweep utility for scaling checks. The transpose matvec comparison measures the
direct `jacobian_transpose_mul` API against explicit `jacobian(...).T @ vec`.
`jacobian_compare` also checks link/joint momentum, force and joint torque,
including all time derivatives available at its configured motion order and
nonzero gravity. These use the public `jacobian_autodiff()` method. The DOF
sweep currently measures the velocity, acceleration and jerk cases only.

`dynamics_autodiff_compare` measures dense dynamics Jacobians with NumPy/Rust
analytic derivatives, the existing numerical-difference API, eager JAX and
JIT-compiled JAX. It separates cached analytic timings, state-inclusive timings,
and first-call JIT cost, synchronizes JAX results, and writes a Markdown report
and raw JSON to `developer/benchmarks/results/dynamics_autodiff.*`. Models cover
the sample arm, a branched tree, and generated 8/16-DOF arms. Use `--repeats`
to change analytic repetitions and `--output` to change the output prefix.

The [recorded comparison](results/dynamics_autodiff.md) includes the measurement
environment; [raw measurements](results/dynamics_autodiff.json) are stored alongside
it. These values are a measurement snapshot, not performance requirements.

## Pinocchio Comparison

```bash
uv run --extra developer python -m developer.benchmarks.pinocchio_compare
uv run --extra developer python -m developer.benchmarks.fast_minimal_compare
```

Pinocchio is optional and intentionally not listed as a normal project
dependency. If it is not installed, the script prints a skip message and exits.

The Pinocchio comparison measures runtime categories on a generated model with
the same topology. The outputs are not exactly equivalent to RoboKots CMTM
state/Jacobian outputs, so use the numbers as a performance reference rather
than a strict numerical equivalence test.

The minimal fast comparison strips RoboKots semantics down to q/v/a array-only
kernels for FK, inverse dynamics, and joint Jacobian timing. It is intended to
measure the lower bound for a compiled fast path, not to replace public APIs.

## Forward Dynamics (ABA) Comparison

```bash
uv run --extra developer python -m developer.benchmarks.forward_dynamics_compare
```

This directly compares fixed-base ABA with Pinocchio at 16 and 64 DoF. It
reports one-shot and reusable-workspace RoboKots calls separately; reusable
storage is the fair counterpart to Pinocchio's persistent `Data` object. The
`InwardCache` result additionally reuses an ABA mass factorization for many
efforts at the same `q/v/gravity`. The batch result compares RoboKots' native
batch API with a Python loop over Pinocchio's scalar ABA API.

## Fixed Rust Comparison

```bash
uv run --extra developer python -m developer.benchmarks.fixed_rust_compare --profile quick
uv run --extra developer python -m developer.benchmarks.fixed_rust_compare --profile full
```

This is the stable comparison suite for Rust optimization work. It uses fixed
DOF counts, batch sizes, random seed, and output columns so results can be
compared across implementation changes.

The core section compares Pinocchio, the Rust Pinocchio-like q/v/a path, CMTM
full dynamics, CMTM torque-only dynamics, and higher-order CMTM dynamics. The
public section compares Python and Rust RoboKots API calls for torque and
second-order torque derivatives, including a mixed `total_joint` output with
joint coord/velocity/acceleration/jerk and torque/dtorque/ddtorque. The selected
section measures already-computed state kernels (`STATE`, `J`, `JM8`, and
`JTM4`) for total torque, local link dynamics, and the mixed `total_joint`
case, so regressions in Rust fast paths are visible without state update
overhead. The mixed selected case is intentionally downsampled in `quick`
because `torque_diff1/torque_diff2` still exercise the general outward
transpose/Jacobian paths; use `full` when you need the 64-DOF mixed selected
numbers. Results are written to CSV with a matching JSON metadata file.

## Fused Trajectory VJP

```bash
uv run python -m developer.benchmarks.fused_vjp_trajectory
```

This uses the direct-collocation workload of 509 steps and 69 DoF by default.
It checks that the sum of four separate VJPs (`torque`, `torque_diff1`,
`torque_diff2`, and total kinetic energy) matches one
`jacobian_transpose_mul_many` result, then reports both timings.  State
construction is outside the timed region.  Use `--rhs-cols 8` to model a
multi-column parameter VJP.
