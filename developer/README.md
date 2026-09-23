# Developer Tools

This directory contains local benchmark and investigation tools. These tools are
not part of RoboKots' normal runtime path.

For the staged plan to make Rust independent and avoid expanding full models in
Python, see [Rust改善方針](rust_roadmap.md) and the [Python依存の棚卸し](python_dependency_audit.md).
The roadmap describes planned work, not implemented behavior.

## Install Developer Dependencies

Pinocchio is optional and is not installed with the default RoboKots
dependencies. To install the developer extra with `uv`:

```bash
uv sync --extra developer
```

For pip-based editable installs:

```bash
python -m pip install -e ".[developer]"
```

The Pinocchio Python package is distributed on PyPI as `pin`, but it is imported
from Python as `pinocchio`.

## Benchmarks

All performance and comparison scripts live in `developer/benchmarks`.
See `developer/benchmarks/README.md` for the full benchmark list and notes.

Run the regular runtime benchmark:

```bash
uv run python -m developer.benchmarks.runtime
```

The printed baseline values are machine- and dependency-version specific. Use
them as a rough reference only unless the CPU, Python, NumPy, JAX, and power
settings match the baseline environment.

Run the batch API benchmark:

```bash
uv run python -m developer.benchmarks.batch_api
```

Run the high-DOF RoboKots baseline:

```bash
uv run python -m developer.benchmarks.robokots_baseline
```

Run Jacobian comparison and DOF sweep utilities:

```bash
uv run python -m developer.benchmarks.jacobian_compare
uv run python -m developer.benchmarks.jacobian_dof_sweep
```

Run the optional Pinocchio comparison:

```bash
uv run --extra developer python -m developer.benchmarks.pinocchio_compare
```

If Pinocchio is not installed, the comparison script exits with a skip message.
The comparison measures runtime categories on generated models with the same
topology; it is not a strict numerical equivalence test for RoboKots CMTM
outputs.

## API Implementation Boundaries

### Core state module names

`robokots.core.batch_shape` manages leading batch axes and trailing feature
axes: validation, flattening, broadcasting, and restoring output shapes.
Its mapping helper evaluates samples sequentially in Python. The former
`robokots.core.batch` import path has been removed.

Use these implementation paths for new code:

| Implementation module | Responsibility | Removed path |
| --- | --- | --- |
| `robokots.core.state.spec` | State selection, quantity definitions, orders and dimensions | `robokots.core.state_spec` |
| `robokots.core.state.protocol` | Shared read-only backend protocol | `robokots.core.outward_protocol`, `robokots.core.outward_data` |
| `robokots.core.state.tensor` | Backend-independent state/Jacobian array views | `robokots.core.state_tensor` |
| `robokots.core.state.batch` | State collection, batch shape, and validation | `robokots.core.state_batch` |
| `robokots.api.state_cache` | Freshness checks, invalidation, and cached computation | `robokots.core.state_cache` |
| `robokots.outward.data` | NumPy/mathrobo computational state storage | `robokots.core.outward_state` |
| `robokots.core.state.access` | Protocol-based state reads and value transformations | `robokots.core.state_access`, `robokots.outward.access` |
| `robokots.state_io.dictionary` | Dictionary export and serialized-state extraction | `robokots.core.state_dict`, `robokots.core.state_dict_utils` |
| `robokots.state_io.jsonl` | JSON Lines serialization | `robokots.core.state_json`, `robokots.core.state_jsonl` |

The old modules have been removed. Update direct imports, dynamic import strings,
and monkeypatch targets to the implementation paths above. Imports from
`robokots.kots` (including `Kots` and `StateType`) remain unchanged.

Polars table helpers are available through
`from robokots.contrib.polars import RobotDF, RobotState`. The compatibility
modules `robokots.core.state_table` and `robokots.core.dataframe`, as well as
the `RobotDF` and `RobotState` exports from `robokots.core`, have been removed.

Pickles containing removed module paths no longer load by default. Export trusted
data with the version that wrote it and reconstruct it using the current types.
The historical compatibility release (commit `32fa548`) covers earlier migrations,
not the state-package moves above. Array layouts and JSONL formats are unchanged.

### State containers and execution management

`core/state/` groups specifications, reader protocols, typed output arrays, and
the `StateBatch` collection. It does not import outward implementations or API
orchestration. The package lazily exports `StateType`, `OutwardDataView`,
`StateValueProvider`, `StateTensor`, `JacobianTensor`, and `StateBatch`.

`StateBatch` stores scalar state objects and their batch shape, validates the
sample count, and copies the input list. Its former `state_info()` and
`state_info_list()` methods now live as internal helpers in `api/state.py`;
callers use the corresponding `Kots` methods. `state_io` can still export the
container directly without depending on the API layer. Native batched states
continue to use their existing vectorized paths.

`outward/data.py` owns concrete NumPy/mathrobo storage, including its local
derived-value memoization and `state_sample()`, which extracts a batch sample
into an `OutwardState`. Shared read and value-transformation helpers live in
`core/state/access.py`; they depend on `OutwardDataView` and mathrobo value types,
not on concrete outward containers. Import `state_sample` directly from
`robokots.outward.data`; the old `outward.access` module has been removed.
Rust storage and workspaces remain under `outward/rust/`. `api/state_cache.py`
owns revision-based recomputation and the `update_outward_state()` helper,
which is no longer exported by `outward` or `outward.values`.

`robokots.core` lazily exports only types owned by core. Import `OutwardState`
and `ArrayOutwardState` from `robokots.outward.data`, and `StateCache` from
`robokots.api.state_cache`; their former `robokots.core` exports are removed.
`target.py`, `time_grid.py`, and `viz.py` retain their existing responsibilities.

Before/after timings and numerical comparisons are documented in the
[state layout benchmark](benchmarks/README.md#core-state-layout).

### Shared computation kernels

The former `core/models/` implementations are organized under `core/kernels/`.
Core contains both shared data definitions and foundational computation kernels.
Kernels compute local joint/link quantities and assemble whole-body operators;
they read states through `core/state/access.py` and `OutwardDataView`, without
importing concrete outward storage or API orchestration. The intermediate
`outward/kernels/` location is removed, with no compatibility exports.

| Former module under `core.models` | Module under `core.kernels` |
| --- | --- |
| `kinematics.kinematics`, `kinematics.kinematics_matrix` | `joint` |
| `kinematics.base` joint adapters | `joint` |
| `kinematics.base` soft-link adapters, `kinematics.kinematics_soft_link` | `soft_link` |
| `kinematics.kinematics_jax` | `kinematics_jax` |
| `dynamics.base` | `inertia` |
| `dynamics.dynamics` | `dynamics` |
| `dynamics.dynamics_matrix` | `dynamics_derivatives` |
| `cmtm_apply` | `cmtm_apply` |
| `whole_body.basic` | `whole_body.operators` |
| `whole_body.topology_layout` | `whole_body.topology` |
| `whole_body.total_kinematics_mat` | `whole_body.kinematics` |
| `whole_body.total_kinematics_grad_mat` | `whole_body.kinematics_derivatives` |
| `whole_body.total_dynamics_mat` | `whole_body.dynamics` |
| `whole_body.total_dynamics_grad_mat` | `whole_body.dynamics_derivatives` |
| `whole_body.total_partial_grad_mat` | `whole_body.partial_dynamics` |
| `whole_body.total_gravity_grad_mat` | `whole_body.gravity_derivatives` |

`JointData`/`SoftLinkData` and their converters belong with their local kernels.
They remain computation adapters, separate from `core.robot` structure types.
`partial_dynamics` includes both momentum and force partial derivatives.
Dense matrices and direct products stay together by physical operation; JVP/VJP
paths are not replaced by dense matrix construction. The numerical definitions,
factorial normalization, gravity conventions, and CMTM apply selection are unchanged.

There are no old-path compatibility modules or re-exports. Import local kernels
explicitly, for example `from robokots.core.kernels.joint import JointData`.
`core.kernels.whole_body` retains lazy exports of its own operator functions.
Importing NumPy kernels does not select the JAX implementation. Existing public
`Kots` methods are unchanged. Kernel tests live under `tests/core/kernels/`.
See the [kernel layout benchmark](benchmarks/README.md#kernel-layout) for
separate state-building, dense Jacobian, JVP, and VJP measurements. The subsequent
[move into core](benchmarks/README.md#move-into-core) is measured separately.

Whole-body state annotations use `OutwardDataView` rather than `dict`.
Pass world-frame gravity explicitly when selecting calculation conditions.
For existing callers that omit it, gravity helpers use the optional
`state.gravity` attribute, or zero when absent; gravity is not a required
attribute of the common reader protocol. Concrete state creation and sample
extraction remain in `outward/data.py`.

### Computational state and export

Computation reads `OutwardState`, `ArrayOutwardState`, or Rust state views through
`core.state.access` and backend methods. JAX kinematics also returns an
`OutwardState`. Jacobians and numerical reference calculations do not reconstruct
computational state from flat dictionaries. Low-level computational functions
expect state objects; dictionary inputs are no longer supported.

`kinematics()` and `dynamics()` default to `materialize_dict=False` and return
their computed state. Explicit `materialize_dict=True` returns an exported
snapshot. `to_state_dict()` exports the current state and `update_state_dict()`
computes then exports it. `Kots.state_dict_` and `state_dict_source_` have been
removed; callers should use `state_info()` for queries or `to_state_dict()` for
output. Exports own their arrays and are never used as computational caches.
State objects and Rust adapters do not implement `to_state_dict()`.
`state_io.dictionary.export_state_dict(robot, state)` owns dictionary construction;
it reads only `cmtm()` and `quantity_series()` from the state. The latter returns
stored derivatives as `(..., order, dimension)` and raises `KeyError` for missing
quantities. These read methods do not promise independent array copies.
The dictionary-returning `build_kinematics_state()` and
`build_dynamics_cmtm_state()` are explicit export wrappers.

Polars and JSONL consume these snapshots at the output boundary. Dictionary
restoration helpers live in `state_io.dictionary` and do not cache derived
objects from mutable snapshots. State-object caches continue to serve computation.

### Ownership and failed updates

Motion imports own their input arrays. `Kots.motions()` returns a copy; use
`import_motions()` or `import_motion_array()` to change motion. The underlying
`RobotMotions` setters advance the revision after successful changes.
`Kots.state_info()` and `state_info_list(..., list_output=True)` return detached
values; edits do not affect subsequent queries. Computational state readers
remain reference-based. State objects returned by `dynamics()/kinematics()` are
not promised to be snapshots: Rust workspaces can be reused on later updates.
Direct edits to internal attributes (such as `motions_.motions` or
`outward_state_`) bypass the public ownership and revision contract.

`dynamics()` commits gravity only after successful calculation and requested
dictionary export. Input validation failures preserve the previous state.
NumPy calculation/export failures also preserve it. If a Rust compute/export
operation fails, its reusable workspace is evicted and the current computed
state is invalidated, because the workspace may have been partially changed.
The previous gravity setting remains in effect; call `update_state()` or
`dynamics()` again before querying state. Previously retained raw state handles
must not be used after such a failure.

### JSONL validation

`iter_jsonl_rows()` preserves both `times` and `steps` when supplied. Their
lengths must match the states; mismatches raise `ValueError` during iteration.
The keys `t`, `step` and `schema_version` are reserved for explicit arguments
and cannot appear in state or metadata. Metadata/payload key collisions and
collisions after string conversion also raise `ValueError`.
Earlier rows may already have been yielded before a length error is detected.
The existing `write_jsonl()` writer is not transactional: it may leave a partial
file on an iteration or serialization error.

### State selection, batch exports, and fallback contracts

- `state_info_list()` always packs numeric values as `(..., state_dim)`, also
  for single samples and mixed quantities. Selection order is preserved.
  An empty selection returns `batch_shape + (0,)`; `list_output=True` returns
  an empty list. Frame matrices are flattened in row-major order (16 values
  for a 4x4 frame); this is a stored-value layout, not the six-dimensional
  tangent representation used by frame Jacobians. `list_output=True` keeps
  the individual value representations.
- `StateBatch` validates positive integer batch dimensions and the sample
  count at construction. Empty batches are rejected, including motion imports.
  An empty **selection** is supported; an empty **batch** is not.
- `to_state_dict()` and `materialize_dict=True` always return a dictionary,
  including flexible-link and JAX batches. Every array preserves the original
  batch axes. Batched samples must expose identical keys and value shapes.
  To build sample-wise JSONL rows, explicitly index these leading axes; a
  flattened `list[dict]` is no longer returned implicitly.
- `OutwardDataView` is the common computational reader implemented by NumPy,
  array-backed, JAX-produced, and Rust states. `StateValueProvider` separately
  describes the optional optimized `state_value()` lookup.
- `RobotState.state_vecs_traj()` infers component counts from the stored
  vectors, rather than assuming three. For different joint DOFs, pass
  `list_output=True` to receive one `(time, component)` array per owner.
  Empty/invalid trajectories fail explicitly rather than guessing dimensions.
- `StateCache` builders must accept `build_state(x_all, time=..., required=...)`.
  Exceptions from inside a builder are not retried with different signatures.
- Batched/Rust derivative fallbacks catch only `NotImplementedError`, meaning
  an explicitly unsupported path. `RuntimeError`, `ValueError`, `TypeError`,
  and `AttributeError` propagate. Shape errors are no longer treated as lack
  of backend support. Enable DEBUG logging for `robokots.api.state`,
  `robokots.api.derivatives`, and `robokots.api.rust_derivatives` to see reasons
  for exception-triggered fallbacks. Capability checks may still select the
  next implementation without raising an exception.

Removing exception-based shape fallbacks also exposed native batch indexing
issues. Torque projection now broadcasts the joint tangent over derivative
axes explicitly; world-force Jacobian slices preserve batch axes. Batched
dynamics reverse products reuse per-sample computational readers and do not
materialize dense Jacobians in the generic reverse kernel.

### Facade and computation

`robokots.kots.Kots` remains the public facade. Its implementation is being
split incrementally without adding another user-visible state container.

- `robokots.api.inward`: array-oriented RNEA/ABA and `InwardCache` creation.
- `robokots.api.outward`: kinematics/dynamics orchestration and backend
  validation.
- `robokots.api.state`: semantic state construction, `StateCache`, batch state,
  and explicit state export.
- `robokots.api.rust_backend`: Rust kernel dispatch and Rust outward workspace
  lifetime/cache management.
- `robokots.api.derivatives`: public Jacobian/JVP/VJP APIs, numerical
  fallback, batch-shape handling, and target derivative helpers.
- `robokots.api.fast_derivatives`: specialized joint-motion/joint-torque
  NumPy paths.
- `robokots.api.rust_derivatives`: Rust RNEA, CMTM, link-local derivative
  kernels.
- `robokots.api.whole_body`: whole-body quantities and kinetic-energy value/JVP/VJP
  dispatch for NumPy and Rust.
- `Kots`: the stable public facade plus model, motion, semantic state cache,
  targets, and visualization helpers.

`StateCache` holds semantic outward/CMTM state. Rust and inward workspaces are
algorithm-specific numerical storage and must not be inserted into that cache.

### Native Rust model and Python boundary

`robokots/_rust/src/model.rs` defines `RobotModel`, `LinkModel`, and
`JointModel`. `RustCompiledRobot::from_model()` validates topology and compiles
these ordinary Rust values. Link zero is the root; every other link must be
connected exactly once, and joints must appear in parent-before-child order.
The native constructor rejects invalid indices, disconnected links, cycles,
and invalid traversal order before accessing model arrays.

`types.rs` owns native model/state containers and their factories. Python
classes in `py_api.rs` hold these values directly; wrapping does not clone
buffers or add a separate heap allocation. Dictionary/NumPy conversion and
Python exceptions remain at this boundary. The existing Python class names,
methods, array layouts, and supported joint subsets are retained.

The default Cargo feature `python` builds the PyO3 extension. Both PyO3 and
NumPy are optional dependencies, so the native model and kernels can also be
built and tested without Python:

```bash
cargo test --offline --manifest-path robokots/_rust/Cargo.toml --no-default-features
cargo tree --offline --manifest-path robokots/_rust/Cargo.toml --no-default-features
```

Native state operations are now public: outward/batch compute methods, local
matrix/vector and local/world motion/wrench getters, selected JVP/VJP application,
and ABA preparation/solve. Selected inputs and outputs use flattened row-major
arrays, with explicit batch size and RHS column count. Dense Jacobians can be
obtained by supplying an identity RHS; product methods do not build one.
See [the external-crate integration tests](../robokots/_rust/tests/native_api.rs)
for a complete model/state/finite-difference/JVP/VJP/ABA example without Python.

ModelInfo holds names, joint kinds, DOFs, motion offsets and parent/child indices.
Its immutable native storage is shared by model clones. Python obtains the
metadata in one transfer and caches an immutable RobotModelInfo by compiled
model identity. Rust state adapters and Rust-specific derivative selection use
this metadata instead of retaining the full RobotStruct. State reads and exports
also use the metadata for raw Rust states. The default input path retains the
existing RobotStruct behavior; native-first input is described below.

Performance and regression results for this step are recorded in
[the native API comparison](benchmarks/results/native_api_comparison.md).

LinkModel and JointModel now require a name in Rust. Names must be nonempty and
unique within each owner type. The low-level Python dictionary constructor
retains support for missing names by assigning link_N/joint_N. Native CMTM
factories explicitly reject prismatic models; their RNEA/ABA support is retained.
Minimal dynamics permits torque reads but rejects uncomputed momentum/force
values. Native getters return owned vectors, so callers cannot mutate storage.

### Native-first model input

```python
k = Kots.from_json_file("model.json", order=4, backend="rust")
# Also available on from_json_data(...) and from_urdf_file(...).
k.import_motions(motion)
k.dynamics()  # Rust is the default for instances loaded with backend="rust".
```

`backend="rust"` bypasses RobotStruct/LinkStruct/JointStruct construction and
RobotStruct.to_dict. JSON text is parsed by Rust using serde_json; decoded
Python dictionaries cross PyO3 directly without JSON serialization. URDF still
uses the existing Python XML reader/topological normalization, then passes its
decoded data directly to Rust. Neither path expands the Python numerical model.
Canonical schema 0.0.2 validation and ID sorting live in model_input.rs; the
native constructor validates names, topology and finite numerical values.
Joint IDs must retain the existing parent-before-child order and link 0 is root.

Kots keeps a private input snapshot and lightweight metadata. A full Python model
is generated once on demand for `k.robot_`, NumPy/JAX computation, or other
features that require it. File input is not read again. Rust-supported state and
selected derivative operations, RNEA/ABA and set_order do not require expansion.
An unsupported derivative fallback can still materialize a Python model; this
is not a guarantee that every public Kots method runs entirely in Rust.
The native-first model is a snapshot: to change the model, create a new Kots;
directly editing the lazily generated Python model does not update Rust caches.

The new input mode requires dim=3 and lib="numpy". Fixed/revolute rigid models
support CMTM, while prismatic models support RNEA/ABA and explicitly reject
CMTM state computation. Input construction without a backend retains the
Python model path; on-demand calculations follow the backend policy below.
The low-level legacy `from_model_data` binding retains its permissive defaults;
new canonical inputs use `from_json` or `from_input_data`.

Rust callers can use `RustCompiledRobot::from_json` or `from_json_file` without
Python. URDF parsing is not yet available from the Rust-only crate. serde_json
is now a native dependency; PyO3 and NumPy remain optional.

[Input-path performance comparison](benchmarks/results/native_model_input_comparison.md)
includes model preparation, first dynamics and repeated computations.

This remains an experimental Rust API: many kernels remain crate-private.
Mathrobo-compatible views, native URDF reading,
validation beyond the supported rigid-model subset and coordinated model replacement are still
separate follow-up work. Python-free unit and integration tests validate
construction, indexing, state lifetime, matrix-free products and numerical
consistency. Existing Python numerical and API tests cover the binding path.

### Native world values and typed calculation API

`RustOutwardData` and `RustBatchOutwardData` expose `world_link_vec` and
`world_joint_vec`. Their key_order is 2 for velocity, 3 for acceleration, and so
on; values contain angular then linear components. Batch native getters accept
an explicit sample index. Joint outputs transform the relative joint motion
with the child-link frame; they do not substitute the child's absolute motion.
Higher derivatives include the moving frame's derivatives and retain ordinary
(non-factorial-normalized) derivative values. Cached order-3/minimal dynamics
and fixed joints obey the same convention.

The Python Rust adapter and public value dispatch now use those native getters;
NumPy retains its Python implementation. A framed joint `jerk` (`local`/`world`)
is spatial motion, whereas the existing unframed joint-coordinate selection
keeps its coordinate-derivative interpretation.

The experimental external Rust selected API now accepts `StateOutput` with
`StateOwner::{Link, Joint}`, `StateQuantity` and `ReferenceFrame::{Local, World}`.
It replaces the previous exported numeric tuple alias. The packed numeric
protocol remains crate-private for the existing Python binding. The output's
`derivative` is zero-based (SpatialMotion 0 = velocity); `width()` is its tangent
row count. Rotation/frame derivatives have 3/6 rows, distinct from matrix values.
See the [executable Rust example](../robokots/_rust/src/lib.rs) and
[integration tests](../robokots/_rust/tests/native_api.rs).

`RustCompiledRobot::inverse_dynamics`/`forward_dynamics` and their `_batch`
variants expose RNEA/ABA directly, with explicit world gravity and checked
lengths/finite inputs. Batch arrays are flattened (batch, dof). Python rnea/aba
methods delegate to these operations. Use `create_aba_data` for repeated cached
ABA solves; these convenience methods allocate temporary workspaces.

[World value performance and validation](benchmarks/results/world_state_comparison.md).

### Selected Rust dynamics derivatives

For dynamics requests with motion order at least 3, the Rust derivative adapter
can select link/joint momentum and force in local or world coordinates together
with joint torque, local/world spatial velocity and its higher time derivatives,
and position/orientation/frame tangents.
`None` and `"local"` denote local spatial outputs. Mixed owners, frames,
derivative orders, repeated selections, `total_joint` expansion, and leading
batch axes retain the public output ordering.

`robokots/_rust/src/dynamics_outputs.rs` selects results from the same primal
and tangent recurrence used by the torque-series API. World JVPs include both
wrench and moving-transform derivatives. World VJP seeds are accumulated with
local force/momentum/torque seeds before the common dynamics and kinematics
reverse pass. `jacobian_mul()` and `jacobian_transpose_mul()` use direct products;
only `jacobian()` supplies a full input basis to materialize a dense Jacobian.
Spatial outputs use the same tangent buffers; their VJP seeds join the
same reverse pass. Joint spatial motion uses relative joint CMTM vectors,
not child-link motion. The existing pure-torque and pure-kinematics dispatch
paths remain in use. Python still validates selections and converts arrays;
these mixed derivative requests no longer assemble Python analytic kernels.

The Python adapter is in `robokots/api/rust_derivatives.py`, and the batched PyO3
stateless entry points remain `dynamics_selected_tangent_batch` and
`dynamics_selected_transpose_batch`. The public adapter now uses
`RustSelectedWorkspace.apply`, created by `create_selected_workspace(order)`.
Pure kinematics (including orders 1 and 2) skips dynamics allocation and
computation; pose-only derivatives need just order 1.

The derivative workspace is bound to a compiled model and motion order and
holds only the latest batch, plus one tangent RHS-width allocation. It compares
motion values exactly per sample and gravity for dynamics; unchanged samples
reuse their primals across dense/JVP/VJP calls. Changing model/order replaces
the workspace, batch size or kinematics/dynamics mode replaces primal storage,
and changing RHS width replaces only the tangent storage. Returned arrays own
their memory. Invalid shapes/descriptors are rejected before cache mutation.
`cache_info()` exposes evaluation counts for regression tests and benchmarks.
This numerical workspace stays outside `StateCache`. The first derivative
call still prepares its own primal rather than borrowing a mutable outward
state; reverse scratch arrays are still allocated per invocation.
The supported model set remains the Rust CMTM fixed/revolute rigid-link subset.
Requests outside the selected-output contract retain their existing dispatch,
including joint coordinate outputs. This change does not extend model support.

See the [mixed-output before/after benchmark](benchmarks/README.md#mixed-rust-outputs).
NumPy spatial selection now uses the shared output operators in
`outward/diff/spatial_outputs.py` for dense, JVP and VJP requests.
`core/state/spec.py:StateOutput` defines owner, family, derivative index, width
and frame for NumPy and Rust; numeric Rust family codes stay in the adapter.
Spatial operators and route products broadcast over native batch axes rather
than slicing each sample. Matrix VJPs reuse computed states, including zero
and nonzero gravity. World-force-only selections now use direct products too.
The existing scalar dynamics reverse subroutine still handles individual batch
samples internally; it no longer forces spatial selections or state building
into that loop. Joint spatial
outputs select relative joint motion; a fixed joint therefore has zero relative
velocity even when its child link moves. This corrects the previous mixed
assembly's use of the preceding link's rows. Independent central differences
of state values cover selection order, duplicates, batch axes and higher orders.
Selected joint spatial derivatives currently cover fixed and one-DoF joints;
multidimensional relative joint derivatives raise `NotImplementedError`.
When world force outputs accompany spatial selections, NumPy also transports
local force variations and frame variations directly, including in reverse mode.

World spatial motion is the ordinary derivative series of `Ad(T) v_local`,
including derivatives of `T`. For a joint, `v_local` is relative joint motion
and `T` is its child's world transform; it is not the child's absolute velocity.
The same value conversion serves NumPy state reads and Rust state adapters.

Pose state values keep their owner transform (absolute link, relative joint).
The Jacobian uses tangent dimensions: `pos` and `rot` have 3 rows, `frame` 6.
None/local uses `vee(T^-1 dT)`; world frame uses `vee(dT T^-1)` and world rotation
uses its angular part. World position uses `dp`, while local position uses
`R^T dp`, preserving the existing local convention. `numerical=True` now uses
these same base-frame tangent conventions instead of differentiating flattened
rotation matrices. Position/orientation values themselves do not change when
the tangent frame is selected.

See the [corrected spatial-output measurements](benchmarks/README.md#spatial-world-and-pose-outputs)
for the new workload and the separate unchanged-local regression benchmark.

Run the fixed Rust comparison used for optimization work:

```bash
uv run --extra developer python -m developer.benchmarks.fixed_rust_compare --profile quick
```

### Rust high-order state computation

For motion order >= 4, `cmtm_generic.rs::kinematics_cmtm_high_order_into`
propagates ordinary spatial-velocity derivatives using factorial-scaled
relative inverse-rotation coefficients. This replaces 4x4 series composition
and velocity recovery for the supported fixed/revolute models.
`cmtm_series.rs::dynamics_cmtm_into` reuses momentum wrench-transport blocks
for the gravity series across the same joint. No persistent state-cache fields
are added; one temporary rotation buffer of `72 * (order - 1)` bytes is reused
across joints. The order-3 zero-gravity specialized path remains in place.

See the [production comparison](benchmarks/results/high_order_production.md)
for timings including public API costs. High-order state values and existing
analytic Jacobian/JVP/VJP paths are checked against NumPy and central differences
in `tests/outward/test_rust_high_order.py`.

### Shared Rust outward workspace

`RustOutwardData` and `RustBatchOutwardData` each use `dynamics.cmtm` as their
single kinematics allocation. At creation, `DynamicsCmtmWorkspace::kinematics_only`
leaves all dynamics-specific vectors empty. The first full or minimal dynamics
call invokes `ensure_dynamics` to allocate them without replacing or copying the
CMTM state. Later kinematics calls invalidate dynamics values while retaining
its allocated capacity for reuse. Read access is controlled by the existing
`has_kinematics`, `has_dynamics`, and order-3 completion flags.

Transient derivative workspaces still use the fully allocated `new` constructor.
The raw object's private `_workspace_buffer_bytes()` diagnostic returns shared
kinematics and dynamics-only numerical capacities; it excludes model copies,
object headers, allocator overhead and Python outputs. See
[allocation and timing measurements](benchmarks/results/shared_workspace.md).
This does not change the existing cache of workspaces by order and batch shape.


### NumPy-only computation

Use `Kots.from_json_file(..., backend="numpy")` (also available for dict and
URDF input) to select NumPy for state computation and on-demand energy,
inverse dynamics, forward dynamics, and `create_inward_cache()`. This path
requires no RoboKots Rust extension. `lib="numpy"` alone selects the array
library and is not a backend policy.

Energy uses `0.5 * v.T @ I @ v` per rigid link; its analytic JVP/VJP use the
existing direct kinematics products without constructing a dense Jacobian.
NumPy inverse dynamics uses the outward recurrence. NumPy forward dynamics
constructs the mass matrix and solves it with `numpy.linalg.solve`; its cache
reuses the mass matrix and bias. These are different algorithms from Rust ABA
and can be substantially slower. Inverse/forward dynamics currently support
rigid fixed-base models with fixed/revolute/prismatic joints.

For on-demand energy and inward operations, an explicit method `backend`
argument wins; otherwise the latest successful state computation's backend
wins, followed by the input backend. Thus `dynamics(backend="numpy")` on a
native-first instance also makes subsequent energy/inward calls use NumPy.
State-building methods without a backend argument still follow the input
backend. An inward cache retains the backend selected at creation.
For legacy construction without an input backend or any computed state,
on-demand operations retain the Rust default. `forward_dynamics`'s
`reference` alias now also uses NumPy inverse dynamics, with no hidden Rust call.
An explicit `jacobian_autodiff()` still requests JAX.

### Rust kinematic Jacobians from ancestor-route blocks

The general selected-kinematics path uses the same route/block structure as
NumPy's CMTM formulation, without constructing a full CMTM matrix. For each
selected body and ancestor joint, it computes the Taylor coefficients of
`b_j = Ad(T_body^-1 T_child(j)) S_j`. The pose variation is
`eta = sum_j b_j delta_q_j`; body motion varies as
`delta_v = eta_dot + [v, eta]`. For absolute world motion this simplifies to
`delta(Ad(T) v) = Ad(T) eta_dot`. Joint relative world motion uses
`Ad(T_child) ([eta_child, v_joint] + delta_v_joint)` instead, preserving its
relative-motion meaning. Ordinary input/output derivatives are recovered with
factorials; the internal coefficient series remain Taylor-normalized.

`RustSelectedWorkspace::jacobian` assembles these small blocks directly for
kinematic selections, without an identity seed. `apply` and its transpose
consume the same blocks without a dense Jacobian. The implementation shares
route coefficients across selected outputs and uses only needed link frames.
Dedicated low-order local and torque paths remain in place; selections mixed
with dynamics retain the existing dynamics recurrence. Supported CMTM model
kinds are unchanged (fixed/revolute).

Reproducible measurements are in
[the route-block report](benchmarks/results/rust_route_blocks_comparison.md).

### Cached direct Rust Jacobian products

For torque-only selections with motion order 3, `RustSelectedWorkspace::apply`
uses a body-coordinate RNEA linearization for both Jv and Jᵀv. It propagates
directions or adjoints directly; it does not construct a dense Jacobian or
identity seeds. The local derivative blocks and scratch require storage linear
in the number of links and joints per batch sample. A motion or gravity change
invalidates that sample's linearization. Repeated, reordered, and fixed-joint
output rows preserve the selected-output contract.

Kinematic products cache the ancestor-route coefficients described above.
Motion, output selection, or batch-size changes invalidate the affected cache.
The kinematic cache size depends on the selected ancestor routes and motion
order. High-order torque and mixed dynamics selections retain their existing
direct-product algorithms.

The [Jv/Jᵀv comparison](benchmarks/results/rust_products_comparison.md) separates
warmed state reuse from state updates. The largest improvement is torque Jᵀv;
state-inclusive Jv is approximately unchanged in the seven-DOF measurement.
