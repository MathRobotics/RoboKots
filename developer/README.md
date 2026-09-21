# Developer Tools

This directory contains local benchmark and investigation tools. These tools are
not part of RoboKots' normal runtime path.

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
| `robokots.outward.access` | Direct computational state access | `robokots.core.state_access` |
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
derived-value memoization. Read helpers remain separate in `outward/access.py`.
Rust storage and workspaces remain under `outward/rust/`. `api/state_cache.py`
owns revision-based recomputation and the `update_outward_state()` helper,
which is no longer exported by `outward` or `outward.values`.

`robokots.core` lazily exports only types owned by core. Import `OutwardState`
and `ArrayOutwardState` from `robokots.outward.data`, and `StateCache` from
`robokots.api.state_cache`; their former `robokots.core` exports are removed.
`core/models/`, `target.py`, `time_grid.py`, and `viz.py` retain their existing
responsibilities in this change. Only imports within the models are updated.

Before/after timings and numerical comparisons are documented in the
[state layout benchmark](benchmarks/README.md#core-state-layout).

### Computational state and export

Computation reads `OutwardState`, `ArrayOutwardState`, or Rust state views through
`outward.access` and backend methods. JAX kinematics also returns an
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
  kernels and kinetic-energy derivative operations.
- `Kots`: the stable public facade plus model, motion, semantic state cache,
  targets, and visualization helpers.

`StateCache` holds semantic outward/CMTM state. Rust and inward workspaces are
algorithm-specific numerical storage and must not be inserted into that cache.

### Selected Rust dynamics derivatives

For dynamics requests with motion order at least 3, the Rust derivative adapter
can select link/joint momentum and force in local or world coordinates together
with joint torque. `None` and `"local"` denote local spatial outputs. Mixed
owners, frames, derivative orders, repeated selections, `total_joint` expansion,
and leading batch axes retain the public output ordering.

`robokots/_rust/src/dynamics_outputs.rs` selects results from the same primal
and tangent recurrence used by the torque-series API. World JVPs include both
wrench and moving-transform derivatives. World VJP seeds are accumulated with
local force/momentum/torque seeds before the common dynamics and kinematics
reverse pass. `jacobian_mul()` and `jacobian_transpose_mul()` use direct products;
only `jacobian()` supplies a full input basis to materialize a dense Jacobian.
The existing pure-torque fast paths remain in use.

The Python adapter is in `robokots/api/rust_derivatives.py`, and the batched PyO3
entry points are `dynamics_selected_tangent_batch` and
`dynamics_selected_transpose_batch`. Each call computes its own primal state;
workspaces are reused across samples within the call, not cached across calls.
The supported model set remains the Rust CMTM fixed/revolute rigid-link subset.
Requests outside the selected-output contract retain their existing dispatch.

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
