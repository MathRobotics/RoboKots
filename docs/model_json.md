# RoboKots Model JSON

This document describes the canonical RoboKots robot model JSON format.

## Version

Every model must include:

```json
{
  "schema_version": "0.0.2"
}
```

Use `robokots.core.robot.validate_model_data(model_data)` or
`robokots.robot_io.validate_model_data(model_data)` to validate a decoded JSON
object before constructing a `RobotStruct`.

## Top-Level Object

```json
{
  "schema_version": "0.0.2",
  "name": "optional_model_name",
  "links": [],
  "joints": []
}
```

`links` must be a non-empty array. `joints` must be an array.

## IDs And Ordering

`link.id` and `joint.id` must each be unique contiguous integers from `0` to
`N-1`. The array order is not significant; loaders normalize links and joints
by `id` before building the internal model.

This preserves the current internal indexing contract while allowing JSON files
to be organized in a human-friendly order.

## Links

```json
{
  "id": 0,
  "name": "base",
  "type": "rigid",
  "mass": 1.0,
  "cog": [0.0, 0.0, 0.0],
  "inertia": {
    "ixx": 1.0,
    "ixy": 0.0,
    "ixz": 0.0,
    "iyy": 1.0,
    "iyz": 0.0,
    "izz": 1.0
  }
}
```

Supported link types in schema `0.0.2`:

- `rigid`
- `soft`

`cog` is a 3-element vector in the link frame. `inertia` follows URDF naming
and contains the rotational inertia about the link frame origin.

## Joints

```json
{
  "id": 0,
  "name": "joint1",
  "type": "revolute",
  "parent_link_id": 0,
  "child_link_id": 1,
  "axis": [0.0, 0.0, 1.0],
  "origin": {
    "position": [0.0, 0.0, 0.0],
    "orientation": [1.0, 0.0, 0.0, 0.0]
  }
}
```

Supported joint types in schema `0.0.2`:

- `fixed`
- `revolute`
- `prismatic`
- `spherical`
- `floating`

`fix` is not accepted. Use `fixed`.

`parent_link_id` and `child_link_id` reference link IDs. A joint cannot connect a
link to itself.

### Axis

`axis` is required for:

- `revolute`
- `prismatic`

It must be a 3-element, finite, non-zero vector.

`axis` is not required for `fixed`, `spherical`, or `floating`.

For `spherical`, `axis.angular` can optionally specify the 3 angular basis
vectors used by the rotation-vector coordinates:

```json
{
  "type": "spherical",
  "q_representation": "rotation_vector",
  "axis": {
    "angular": [
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0]
    ]
  }
}
```

The `angular` matrix must be 3x3, finite, and full rank. If omitted,
RoboKots uses the identity angular basis.

### Multi-DoF Joints

`spherical` joints use SO(3) rotation-vector coordinates:

```json
{
  "type": "spherical",
  "q_representation": "rotation_vector",
  "dof": 3
}
```

`q` is a 3-element rotation vector. Its direction is the rotation axis, and its
norm is the rotation angle.

`floating` joints use SE(3) exponential-map coordinates:

```json
{
  "type": "floating",
  "q_representation": "expmap",
  "dof": 6
}
```

If `dof` is present, it must match the joint type: `0` for
`fixed`, `1` for `revolute`/`prismatic`, `3` for `spherical`, and `6` for
`floating`.

The Python backend supports these joint types. Rust CMTM calculations,
including kinetic energy and its JVP/VJP, support only `fixed` and `revolute`;
models containing `prismatic` joints are explicitly rejected by these operations.
Rust inverse/forward dynamics (RNEA/ABA) also support `prismatic`.

### Origin

`origin.position` is a 3-element translation vector.

`origin.orientation` is a quaternion in `[w, x, y, z]` order. It must be a
4-element vector with finite components and a finite, non-zero norm.
Both Python and Rust normalize it to unit length when constructing the model.

## Topology

The JSON schema does not make closed links invalid. Closed-loop and non-tree
topologies are reserved for future support.

Current `RobotStruct` construction supports only tree topology. If a model is
valid JSON but uses a topology the current implementation cannot compute, model
construction raises `NotImplementedError` rather than treating the JSON itself
as invalid.

## Reserved Future Joint Types

The following names are reserved for future schema versions and are not
implemented in schema `0.0.2`:

- `planar`
- `custom`

## Native-first input

`Kots.from_json_file(..., backend="rust")` parses and validates JSON in Rust,
without constructing a Python RobotStruct. `from_json_data(..., backend="rust")`
accepts a decoded, JSON-compatible dictionary directly. These instances default
to Rust for kinematics/dynamics; an explicit Python computation or access to
`robot_` materializes the Python model on demand from a retained snapshot.
Calls without the input backend retain the previous behavior.

The supported native model subset is rigid links and fixed/revolute/prismatic
joints. Prismatic models support RNEA/ABA, but not CMTM states/derivatives.
Native input additionally rejects negative/nonfinite mass, nonfinite vectors,
zero quaternions, and zero moving-joint axes. It sorts array entries by ID and
requires joint IDs in parent-before-child order, rooted at link 0. Names are
nonempty and unique within the link/joint groups. Malformed or unsupported
native input raises ValueError; it does not fall back to Python construction.

The input snapshot is independent of later changes to the original dictionary
or file. Construct a new Kots to change the model; mutating the generated Python
model does not synchronize existing native workspaces.
