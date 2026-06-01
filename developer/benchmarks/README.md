# Developer Benchmarks

These scripts are for local performance investigation and are not part of the
normal RoboKots runtime path.

## Runtime Benchmark

```bash
uv run python -m developer.benchmarks.runtime
```

Measures kinematics, dynamics, Jacobian, numerical Jacobian, and cached state
update runtime on the sample model.

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
direct `jacobian_transpose_matvec` API against explicit `jacobian(...).T @ vec`.

## Pinocchio Comparison

```bash
uv run --extra developer python -m developer.benchmarks.pinocchio_compare
```

Pinocchio is optional and intentionally not listed as a normal project
dependency. If it is not installed, the script prints a skip message and exits.

The Pinocchio comparison measures runtime categories on a generated model with
the same topology. The outputs are not exactly equivalent to RoboKots CMTM
state/Jacobian outputs, so use the numbers as a performance reference rather
than a strict numerical equivalence test.
