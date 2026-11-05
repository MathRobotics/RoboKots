"""Evaluation-only AD of FK/ID and their total time derivatives.

Inputs are ordinary motion entries (q, qdot, qddot, ...), not a trajectory.
D_t f(x) = JVP(f, x, shift(x)); repeated applications also differentiate
the state-dependent direction shift(x). No high-order dynamics recurrence,
polynomial trajectory, finite difference or custom derivative rule is used.
"""
import jax
import jax.numpy as jnp

from robokots.core.kernels.inertia import spatial_inertia
from robokots.core.state.spec import keys_momentum, keys_force, keys_torque
from robokots.outward.diff.dynamics_jax import _ad, _hat


VELOCITY_TYPES = ("vel", "acc", "jerk", "snap", "crackle")


def motion_direction(motion):
    """Time direction of a (dof, order) motion array.

    The last slot is padded by zero. Callers must supply enough orders that
    this padding cannot affect their requested derivative.
    """
    return jnp.concatenate((motion[:, 1:], jnp.zeros_like(motion[:, :1])), axis=1)


def total_time_derivative(function):
    """Total derivative, including dependence of the direction on motion."""
    def differentiated(motion):
        return jax.jvp(function, (motion,), (motion_direction(motion),))[1]
    return differentiated


def _validate_robot(robot):
    if any(link.dof for link in robot.links):
        raise NotImplementedError("Time AD supports rigid links only")
    if any(j.type not in ("fixed", "revolute", "prismatic") for j in robot.joints):
        raise NotImplementedError("Time AD supports fixed/revolute/prismatic joints only")


def _poses(robot, motion):
    """Ordinary FK only: time derivatives are taken externally by JVP."""
    q = motion[:, 0]
    root = robot.links[0].name
    world = {root: jnp.eye(4, dtype=motion.dtype)}
    relative = {}
    for joint in robot.joints:
        local = jnp.eye(4, dtype=motion.dtype)
        if joint.dof:
            coordinate = q[joint.dof_index]
            axis = jnp.asarray(joint.axis)
            if joint.type == "revolute":
                hat = _hat(axis)
                rotation = jnp.eye(3) + jnp.sin(coordinate) * hat + (1 - jnp.cos(coordinate)) * (hat @ hat)
                local = local.at[:3, :3].set(rotation)
            else:
                local = local.at[:3, 3].set(axis * coordinate)
        relative[joint.name] = jnp.asarray(joint.origin.mat()) @ local
        parent = robot.links[joint.parent_link_id].name
        child = robot.links[joint.child_link_id].name
        world[child] = world[parent] @ relative[joint.name]
    return world, relative


def _wrench_transform(pose):
    r, p = pose[:3, :3], pose[:3, 3]
    z = jnp.zeros((3, 3), dtype=pose.dtype)
    return jnp.block([[r, _hat(p) @ r], [z, r]])


def _flat_function(function, robot, order):
    def evaluate(motion):
        motion = jnp.asarray(motion)
        if motion.shape != (robot.dof * order,):
            raise ValueError(f"motion must have shape ({robot.dof * order},)")
        return function(motion.reshape(robot.dof, order))
    return evaluate


def make_fk_value(robot, link_name, derivative_order=0, order=None):
    """Return FK pose or its ordinary time derivative, shape (4, 4).

    jacfwd/jacrev of this function returns (4, 4, dof*order). This is an
    elementwise pose derivative, NOT the six-dimensional tangent Jacobian
    returned by Kots.jacobian(StateType(..., "frame")).
    """
    _validate_robot(robot)
    if derivative_order < 0:
        raise ValueError("derivative_order must be nonnegative")
    if order is None:
        order = derivative_order + 1
    if order < derivative_order + 1:
        raise ValueError("Insufficient motion order")
    if robot.link(link_name) is None:
        raise ValueError("Unknown link")
    function = lambda motion: _poses(robot, motion)[0][link_name]
    for _ in range(derivative_order):
        function = total_time_derivative(function)
    return _flat_function(function, robot, order)


def make_time_ad_value(robot, state, order, gravity=(0., 0., 0.)):
    """Build an FK-derived spatial quantity or ID output and time derivatives.

    Single expanded StateType. Motion entries are independent, owner-major
    q, qdot, ...; no trajectory is constructed. Inner time AD is forward JVP;
    callers choose forward/reverse outer motion AD and optional JIT.
    """
    _validate_robot(robot)
    is_velocity = state.data_type in VELOCITY_TYPES
    if not is_velocity and state.data_type not in keys_momentum + keys_force + keys_torque:
        raise NotImplementedError(f"Unsupported time AD quantity: {state.data_type}")
    if state.owner_type not in ("link", "joint"):
        raise ValueError("Time AD requires one expanded link/joint state")
    if state.frame_name not in (None, "local", "world"):
        raise ValueError("Time AD supports local/world frames only")
    if is_velocity and (state.owner_type != "link" or state.frame_name == "world"):
        raise NotImplementedError("Time AD velocity supports local link outputs only")
    if state.data_type in keys_torque and state.owner_type != "joint":
        raise ValueError("Torque requires a joint owner")
    owner = robot.link(state.owner_name) if state.owner_type == "link" else robot.joint(state.owner_name)
    if owner is None:
        raise ValueError("Unknown state owner")
    if order < state.time_order:
        raise ValueError("Insufficient motion order")
    gravity = jnp.asarray(gravity)
    if gravity.shape != (3,):
        raise ValueError("gravity must have shape (3,)")
    gravity_spatial = jnp.concatenate((jnp.zeros(3), gravity))
    inertias = {l.name: jnp.asarray(spatial_inertia(l.mass, l.inertia, l.cog)) for l in robot.links}
    family = "velocity" if is_velocity else state.data_type.split("_diff")[0]
    k = VELOCITY_TYPES.index(state.data_type) if is_velocity else state.key_order - 1

    def velocities(motion):
        world, derivative = jax.jvp(lambda m: _poses(robot, m)[0],
                                    (motion,), (motion_direction(motion),))
        result = {}
        for name, pose in world.items():
            r = pose[:3, :3]
            tangent = r.T @ derivative[name][:3, :3]
            omega = jnp.stack((tangent[2, 1], tangent[0, 2], tangent[1, 0]))
            result[name] = jnp.concatenate((omega, r.T @ derivative[name][:3, 3]))
        return result

    def momenta(motion):
        return {name: inertias[name] @ v for name, v in velocities(motion).items()}

    def base(motion):
        if is_velocity:
            return velocities(motion)[state.owner_name]
        world, relative = _poses(robot, motion)
        if family == "momentum":
            values = momenta(motion)
        else:
            momentum, rate = jax.jvp(momenta, (motion,), (motion_direction(motion),))
            velocity = velocities(motion)
            values = {}
            for name, pose in world.items():
                r, p = pose[:3, :3], pose[:3, 3]
                z = jnp.zeros((3, 3), dtype=motion.dtype)
                inverse = jnp.block([[r.T, z], [-r.T @ _hat(p), r.T]])
                values[name] = rate[name] - _ad(velocity[name]).T @ momentum[name] - inertias[name] @ (inverse @ gravity_spatial)
        if state.owner_type == "link":
            output = values[state.owner_name]
            link_name = state.owner_name
        else:
            joint_values = {}
            for joint in reversed(robot.joints):
                child = robot.links[joint.child_link_id]
                value = values[child.name]
                for child_id in child.child_joint_ids:
                    descendant = robot.joints[child_id].name
                    value = value + _wrench_transform(relative[descendant]) @ joint_values[descendant]
                joint_values[joint.name] = value
            output = joint_values[state.owner_name]
            link_name = robot.links[owner.child_link_id].name
            if family == "torque":
                return jnp.asarray(owner.select_mat).T @ output
        if state.frame_name == "world":
            output = _wrench_transform(world[link_name]) @ output
        return output

    derivative = base
    for _ in range(k):
        derivative = total_time_derivative(derivative)
    return _flat_function(derivative, robot, order)
