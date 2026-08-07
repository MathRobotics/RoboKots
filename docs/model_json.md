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

The Python backend supports these joint types. The Rust backend currently
supports only `fixed` and `revolute`.

### Origin

`origin.position` is a 3-element translation vector.

`origin.orientation` is a quaternion in `[w, x, y, z]` order. It must be a
4-element, finite, non-zero vector.

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
