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

Install optional table support for Polars-backed state tables:
```
pip install ".[table]"
```

## Examples

* `example/simple_example/main.py`: End-to-end kinematics and dynamics Jacobian
  checks with a sample robot model.
* `example/polars_example/main.py`: Demonstrates how to export state to JSONL and
  use Polars to do fast, columnar analytics. This example requires the optional
  `table` extra or a separate `polars` installation.
* `example/benchmark_example/main.py`: Measures runtime of kinematics/dynamics/
  jacobian/state-update. Benchmark settings are edited in the script.

Run an example from the repository root:
```bash
python -m example.simple_example.main
```

Runtime benchmark example:
```bash
python -m example.benchmark_example.main
```

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
kots.jacobian_matvec(states, vec)  # shape: (..., total_state_dim)
kots.target_state_info()           # shape: (..., target_state_dim)
```

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
