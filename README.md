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

For conventional inverse dynamics with gravity, pass joint position,
velocity, and acceleration arrays to `inverse_dynamics`. Gravity is expressed
in the world frame and defaults to Pinocchio's fixed-base convention:

```python
tau = kots.inverse_dynamics(q, v, a)  # gravity = [0, 0, -9.81]
tau_zero_g = kots.inverse_dynamics(q, v, a, gravity=[0, 0, 0])
```

To avoid building the full Python model before using Rust, specify
`backend="rust"` when calling `Kots.from_urdf_file`, `from_json_file`, or
`from_json_data`. These instances also default to Rust for `kinematics()` and
`dynamics()`. Python model-based operations materialize the Python model only
when needed. URDF XML parsing still uses the Python reader.
See [native-first model input](developer/README.md#native-first-model-input).

For NumPy-only computation, pass `backend="numpy"` to the same constructors.
State/Jacobian calculations, kinetic energy and its products, inverse/forward
dynamics, and inward caches then run without the RoboKots Rust extension.
See [NumPy-only computation](developer/README.md#numpy-only-computation) for
backend selection and performance differences.

The gravity-aware API uses the Rust fixed/revolute/prismatic RNEA backend.
Higher-order force and torque derivatives can include gravity as well. The
`dynamics()` default remains zero gravity for backward compatibility:

```python
kots.dynamics(gravity=[0, 0, -9.81])
kots.dynamics(gravity=[0, 0, 0])  # historical behavior and current default

gravity_torque_jacobian = kots.jacobian(
    StateType("total_joint", "total_joint", "torque")
)

# Optional finite-difference reference for verification.
gravity_torque_jacobian_fd = kots.jacobian(
    StateType("total_joint", "total_joint", "torque"), numerical=True
)
```

Gravity is expressed in the world frame. Its moving-link-frame derivatives are
included in `force_diff*` and `torque_diff*` by both the NumPy and Rust CMTM
backends. For `force`, `torque`, and their higher time derivatives, both paths
differentiate the local gravity-wrench CMVector analytically through the
requested CMTM order, then transport and aggregate that gradient through the
existing generalized-force pipeline. For the ordinary torque Jacobian, gravity
therefore changes only the `coord` columns. Finite differences are never chosen
automatically; they remain available only through `numerical=True` for
comparison and validation. `jacobian_mul()` applies the gravity CMTM variation
directly for vector and matrix right-hand sides, without assembling the dense
gravity Jacobian.

## JAX Automatic Differentiation of Dynamics

`jacobian_autodiff()` computes dynamics Jacobians independently with JAX
forward-mode (default) or reverse-mode automatic differentiation:

```python
import jax
from robokots.kots import Kots, StateType

jax.config.update("jax_enable_x64", True)  # recommended for derivative comparisons
kots = Kots.from_json_file("examples/model/sample_robot.json", order=5)
# Import your motion vector (q and its time derivatives) before evaluation.
kots.dynamics(gravity=[0, 0, -9.81])
state = StateType("total_joint", "total_joint", "torque_diff2")
jac_ad = kots.jacobian_autodiff(state)
jac_reverse = kots.jacobian_autodiff(state, mode="reverse")
jac_analytic = kots.jacobian(state)
```

Supported outputs are link/joint `momentum`, `force`, joint `torque`, and their
`*_diffN` time derivatives. Momentum derivative N needs motion order N+2;
force/torque derivative N needs order N+3. Spatial outputs accept local or
world frames. The API preserves batch axes and accepts `list_output=True`,
using the same row and motion-column ordering as `jacobian()`. Gravity follows
the last `dynamics()` call and defaults to zero.
`mode="forward"` uses `jax.jacfwd`; `mode="reverse"` uses `jax.jacrev`.
Both return NumPy arrays without JIT; neither mode changes the output layout.

The JAX implementation supports rigid links with fixed, revolute and prismatic
joints, including branched trees. Flexible links and spherical/floating joints
raise `NotImplementedError`. This is inverse dynamics, not forward dynamics.
`jacobian()` and `dynamics()` retain their existing backend selection;
`jacobian_autodiff()` explicitly selects this independent JAX calculation.

For JIT or differentiable optimization code, use the pure
array function directly (the `Kots` wrapper returns NumPy arrays):

```python
import jax.numpy as jnp
from robokots.outward.diff.dynamics_jax import dynamics_state_vector_jax

states = [StateType("joint", "joint1", "torque")]
value = lambda x: dynamics_state_vector_jax(
    kots.robot_, x, states, order=3, gravity=[0, 0, -9.81]
)
jac = jax.jit(jax.jacfwd(value))(jnp.asarray(kots.motion(3)))
```

Run `python -m developer.benchmarks.jacobian_compare` to compare the analytic,
finite-difference and automatic derivatives for kinematics and dynamics.

## Model JSON

RoboKots model JSON is documented in `docs/model_json.md`. Decoded model data
can be validated with:

```python
from robokots.robot_io import load_json_file, validate_model_data

model_data = load_json_file("robot.json")
validate_model_data(model_data)
```
