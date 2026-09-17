"""JAX-traceable rigid-body inverse dynamics and time-derivative series.

The robot is static; only the flat, owner-major motion vector is traced.
Series rows are ordinary time derivatives, not factorial-scaled coefficients.
Fixed, revolute and prismatic joints are supported, including branched trees.
"""
from math import comb

import jax.numpy as jnp

from ...core.models.dynamics.base import spatial_inertia
from ...core.state_spec import keys_force, keys_momentum, keys_torque


def _hat(v):
    x, y, z = v
    zero = jnp.zeros_like(x)
    return jnp.array([[zero, -z, y], [z, zero, -x], [-y, x, zero]])


def _ad(v):
    w = _hat(v[:3])
    return jnp.block([[w, jnp.zeros((3, 3), dtype=v.dtype)], [_hat(v[3:]), w]])


def _product(a, b, n):
    """Nth derivative of a matrix product (also accepts vector RHS)."""
    return sum(comb(n, k) * (a[k] @ b[n - k]) for k in range(n + 1))


def _relative_transform(joint, q):
    origin = jnp.asarray(joint.origin.mat())
    rotation, position = origin[:3, :3], origin[:3, 3]
    if joint.type == "revolute":
        # The model stores a normalized, static axis. This expression remains
        # differentiable at q=0, without a norm of traced coordinates.
        axis_hat = _hat(jnp.asarray(joint.axis))
        local = jnp.eye(3) + jnp.sin(q[0]) * axis_hat + (1 - jnp.cos(q[0])) * (axis_hat @ axis_hat)
        rotation = rotation @ local
    elif joint.type == "prismatic":
        position = position + rotation @ (jnp.asarray(joint.axis) * q[0])
    rt = rotation.T
    zero = jnp.zeros((3, 3), dtype=rotation.dtype)
    inverse_adjoint = jnp.block([[rt, zero], [-rt @ _hat(position), rt]])
    wrench_adjoint = jnp.block([[rotation, _hat(position) @ rotation], [zero, rotation]])
    return inverse_adjoint, wrench_adjoint


def dynamics_jax(robot, motions, order=3, gravity=(0.0, 0.0, 0.0)):
    """Return local velocity, momentum, force and torque series as JAX dictionaries.

    ``motions`` has shape ``(robot.dof * order,)`` in ``Kots.motion(order)``
    layout. Velocity/momentum have ``order-1`` rows and force/torque ``order-2`` rows.
    Gravity is expressed in world coordinates. Use ``jax.vmap`` for batches
    and close over ``robot`` and ``order`` when using ``jax.jit``/``jacfwd``.
    This computes inverse dynamics; it does not solve for acceleration.

    Higher time derivatives are evaluated analytically on ordinary derivative
    coefficient series, without building full CMTM block matrices. This is a
    representation of the same high-order algebra, not a different mathematical
    differentiation method. Outer motion Jacobians may then be obtained by AD.
    """
    if order < 2:
        raise ValueError("JAX dynamics requires order >= 2 (>= 3 for force/torque).")
    if any(link.dof for link in robot.links):
        raise NotImplementedError("JAX dynamics supports rigid links only.")
    if any(joint.type not in ("fixed", "revolute", "prismatic") for joint in robot.joints):
        raise NotImplementedError("JAX dynamics supports fixed, revolute and prismatic joints only.")
    motions = jnp.asarray(motions)
    gravity = jnp.asarray(gravity)
    if motions.shape != (robot.dof * order,):
        raise ValueError(f"motions must have shape ({robot.dof * order},), got {motions.shape}.")
    if gravity.shape != (3,):
        raise ValueError(f"gravity must have shape (3,), got {gravity.shape}.")

    count = order - 1
    root = robot.links[0].name
    identity_series = [jnp.eye(6)] + [jnp.zeros((6, 6)) for _ in range(count - 1)]
    velocities = {root: [jnp.zeros(6) for _ in range(count)]}
    world_inverse = {root: identity_series}
    world_wrench = {root: identity_series}
    relative_wrench = {}
    for joint in robot.joints:
        parent = robot.links[joint.parent_link_id].name
        child = robot.links[joint.child_link_id].name
        start = joint.dof_index * order
        blocks = motions[start:start + joint.dof * order].reshape(order, joint.dof)
        subspace = jnp.asarray(joint.select_mat)
        relative_velocity = [subspace @ blocks[n + 1] for n in range(count)]
        x0, y0 = _relative_transform(joint, blocks[0])
        x, y = [x0], [y0]
        for n in range(1, count):
            x.append(-sum(comb(n - 1, k) * (_ad(relative_velocity[k]) @ x[n - 1 - k]) for k in range(n)))
            y.append(-sum(comb(n - 1, k) * (y[k] @ _ad(relative_velocity[n - 1 - k]).T) for k in range(n)))
        relative_wrench[joint.name] = y
        velocities[child] = [_product(x, velocities[parent], n) + relative_velocity[n] for n in range(count)]
        world_inverse[child] = [_product(x, world_inverse[parent], n) for n in range(count)]
        world_wrench[child] = [_product(world_wrench[parent], y, n) for n in range(count)]

    link_momentum, link_force = {}, {}
    gravity_spatial = jnp.concatenate((jnp.zeros(3), gravity))
    for link in robot.links:
        inertia = jnp.asarray(spatial_inertia(link.mass, link.inertia, link.cog))
        v = velocities[link.name]
        momentum = [inertia @ row for row in v]
        link_momentum[link.name] = jnp.stack(momentum)
        force = [
            momentum[n + 1]
            - sum(comb(n, k) * (_ad(v[k]).T @ momentum[n - k]) for k in range(n + 1))
            - inertia @ (world_inverse[link.name][n] @ gravity_spatial)
            for n in range(order - 2)
        ]
        link_force[link.name] = jnp.stack(force) if force else jnp.zeros((0, 6))

    joint_momentum, joint_force, joint_torque = {}, {}, {}
    for joint in reversed(robot.joints):
        child = robot.links[joint.child_link_id]
        momentum = link_momentum[child.name]
        force = link_force[child.name]
        for child_id in child.child_joint_ids:
            descendant = robot.joints[child_id].name
            transform = relative_wrench[descendant]
            momentum = momentum + jnp.stack([_product(transform, joint_momentum[descendant], n) for n in range(count)])
            if order > 2:
                force = force + jnp.stack([_product(transform, joint_force[descendant], n) for n in range(order - 2)])
        joint_momentum[joint.name] = momentum
        joint_force[joint.name] = force
        joint_torque[joint.name] = force @ jnp.asarray(joint.select_mat)

    return {
        "link_velocity": {name: jnp.stack(rows) for name, rows in velocities.items()},
        "link_momentum": link_momentum, "link_force": link_force,
        "joint_momentum": joint_momentum, "joint_force": joint_force,
        "joint_torque": joint_torque,
        "world_wrench": {name: jnp.stack(rows) for name, rows in world_wrench.items()},
    }


def dynamics_state_vector_jax(robot, motions, states, order=3, gravity=(0.0, 0.0, 0.0)):
    """Concatenate selected dynamics states, preserving JAX tracing.

    Accepts expanded link/joint StateTypes for momentum, force, torque and
    their time derivatives. Spatial quantities support local/world frames.
    """
    for state in states:
        if state.owner_type not in ("link", "joint"):
            raise ValueError("JAX dynamics states must have link or joint owners.")
        owner = robot.link(state.owner_name) if state.owner_type == "link" else robot.joint(state.owner_name)
        if owner is None:
            raise ValueError(f"Unknown {state.owner_type}: {state.owner_name}")
        if state.data_type not in keys_momentum + keys_force + keys_torque:
            raise NotImplementedError(f"JAX dynamics does not support {state.data_type}.")
        if state.data_type in keys_torque and state.owner_type != "joint":
            raise ValueError("Torque requires a joint owner.")
        if state.frame_name not in (None, "local", "world"):
            raise ValueError("JAX dynamics supports local/world frames only.")
        if state.time_order > order:
            raise ValueError(f"{state.data_type} requires motion order >= {state.time_order}.")
    result = dynamics_jax(robot, motions, order, gravity)
    parts = []
    for state in states:
        family = state.data_type.split("_diff")[0]
        series = result[f"{state.owner_type}_{family}"][state.owner_name]
        n = state.key_order - 1
        if state.frame_name == "world" and family != "torque":
            link_name = state.owner_name if state.owner_type == "link" else robot.links[robot.joint(state.owner_name).child_link_id].name
            parts.append(_product(result["world_wrench"][link_name], series, n))
        else:
            parts.append(series[n])
    return jnp.concatenate(parts) if parts else jnp.zeros((0,))
