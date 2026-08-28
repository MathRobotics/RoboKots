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

`robokots.kots.Kots` remains the public facade. Its implementation is being
split incrementally without adding another user-visible state container.

- `robokots.api.inward`: array-oriented RNEA/ABA and `InwardCache` creation.
- `robokots.api.outward`: kinematics/dynamics orchestration and backend
  validation.
- `robokots.api.state`: semantic state construction, `StateCache`, batch state,
  and lazy `state_dict` materialization.
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
