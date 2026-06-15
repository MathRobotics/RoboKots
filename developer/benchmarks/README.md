# Developer Benchmarks

These scripts are for local performance investigation and are not part of the
normal RoboKots runtime path.

## Runtime Benchmark

```bash
uv run python -m developer.benchmarks.runtime
```

Measures kinematics, dynamics, Jacobian, numerical Jacobian, Jacobian
vector/matrix products, Jacobian-transpose vector/matrix products, and cached
state update runtime on the sample model.

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

```bash
uv run python -m developer.benchmarks.jacobian_compare
uv run python -m developer.benchmarks.jacobian_dof_sweep
uv run python -m developer.benchmarks.jacobian_transpose_matvec_compare
```

These compare analytic, numerical, and JAX autodiff Jacobians, including a DOF
sweep utility for scaling checks. The transpose matvec comparison measures the
direct `jacobian_transpose_mul` API against explicit `jacobian(...).T @ vec`.

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
