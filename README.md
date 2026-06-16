# RoboKots

Utilities based on the Kots framework for robotic system modeling, kinematics, and dynamics analysis.

Clone RoboKots
```
git clone https://github.com/MathRobotics/RoboKots
```

Install RoboKots
```
pip install .
```

This builds the Python package and the experimental Rust backend extension.
With uv, RoboKots can also be added directly from Git:
```
uv add "robokots @ git+https://github.com/MathRobotics/RoboKots.git"
```

Install optional table support for Polars-backed state tables:
```
pip install ".[table]"
```

## Examples

See `examples/README.md` for the regular example commands.

* `examples/simple_example/main.py`: End-to-end kinematics and dynamics Jacobian
  checks with a sample robot model.
* `examples/polars_example/main.py`: Demonstrates how to export state to JSONL and
  use Polars to do fast, columnar analytics. This example requires the optional
  `table` extra or a separate `polars` installation.

Run an example from the repository root:
```bash
python -m examples.simple_example.main
```

Developer-only benchmarks live under `developer/benchmarks`; see
`developer/README.md` for setup and commands.

## Batch Shapes and Axis-Aware Data

Motion can be supplied either in the historical flat backend layout or in the
recommended explicit-axis layout:

```python
# Flat owner-major layout, useful for existing code and low-level backends.
kots.import_motions(motion)        # shape: (..., dof * order)
kots.motion()                      # shape: (..., dof * order)

# Explicit computational layout.
kots.import_motion_array(motion)   # shape: (..., dof, order)
kots.motion_array()                # shape: (..., dof, order)
```

The leading `...` dimensions are batch axes. For example, a time trajectory can
be represented as `(time, dof, order)`, and a time/particle batch as
`(time, particle, dof, order)`.

State and Jacobian APIs preserve those batch axes:

```python
kots.state_info(state)             # shape: (..., state_dim)
kots.state_info_list(states)       # shape: (..., total_state_dim)
kots.jacobian(states)              # shape: (..., total_state_dim, motion_dim)
kots.jacobian_mul(states, vec)     # shape: (..., total_state_dim)
kots.jacobian_mul(states, mat)     # shape: (..., total_state_dim, rhs)
kots.jacobian_transpose_mul(states, vec)  # shape: (..., motion_dim)
kots.jacobian_transpose_mul(states, mat)  # shape: (..., motion_dim, rhs)
kots.target_state_info()           # shape: (..., target_state_dim)
```

For `jacobian_mul`, the right-hand side lives on the motion axis and may have
shape `(..., motion_dim)` or `(..., motion_dim, rhs)`. For
`jacobian_transpose_mul`, the right-hand side lives on the state axis and may
have shape `(..., total_state_dim)` or `(..., total_state_dim, rhs)`.

For code that needs named axes, use the tensor adapters:

```python
motion = kots.motion_tensor()      # axes: (..., "dof", "order")
state = kots.state_tensor(states)  # axes: (..., "state")
jac = kots.jacobian_tensor(states) # axes: (..., "state", "motion")
```

`AxisTensor` keeps logical axes separate from physical memory layout. Use
`to_axes(...)` for semantic axis order changes, and `to_layout(...)` or
`materialize()` when an algorithm needs contiguous memory for a specific
backend.

## URDF Input

You can load URDF directly and reuse the same pipeline:
```python
from robokots.kots import Kots

kots = Kots.from_urdf_file("robot.urdf", order=3)
```

## Model JSON

RoboKots model JSON is documented in `docs/model_json.md`. Decoded model data
can be validated with:

```python
from robokots.robot_io import load_json_file, validate_model_data

model_data = load_json_file("robot.json")
validate_model_data(model_data)
```
