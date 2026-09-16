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
| `robokots.core.state_spec` | State selection, quantity definitions, orders and dimensions | `robokots.core.state` |
| `robokots.core.outward_protocol` | Shared read-only backend protocol | `robokots.core.outward_data` |
| `robokots.core.state_access` | Direct computational state access | Dictionary helpers in computational code |
| `robokots.state_io.dictionary` | Dictionary export and serialized-state extraction | `robokots.core.state_dict`, `robokots.core.state_dict_utils` |
| `robokots.state_io.jsonl` | JSON Lines serialization | `robokots.core.state_json`, `robokots.core.state_jsonl` |

The old modules have been removed. Update direct imports, dynamic import strings,
and monkeypatch targets to the implementation paths above. Imports from
`robokots.kots` (including `Kots` and `StateType`) remain unchanged.

Polars table helpers are available through
`from robokots.contrib.polars import RobotDF, RobotState`. The compatibility
modules `robokots.core.state_table` and `robokots.core.dataframe`, as well as
the `RobotDF` and `RobotState` exports from `robokots.core`, have been removed.

Pickles containing the removed module paths no longer load by default. Migrate
trusted existing pickles using the compatibility release (commit `32fa548`):
load them and save them again so class/function references use the new paths.
Reading these new pickles with an older RoboKots release is not guaranteed.
Array layouts and JSONL formats are unchanged.

### Computational state and export

Computation reads `OutwardState`, `ArrayOutwardState`, or Rust state views through
`core.state_access` and backend methods. JAX kinematics also returns an
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

Run the fixed Rust comparison used for optimization work:

```bash
uv run --extra developer python -m developer.benchmarks.fixed_rust_compare --profile quick
```
