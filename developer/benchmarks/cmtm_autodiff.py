"""Evaluation-only explicit CMTM block-matrix dynamics followed by outer AD.

Blocks and vectors store Taylor coefficients (ordinary derivatives / n!).
Spatial inverse/coadjoint CMTMs are lower block-Toeplitz matrices. Higher
time derivatives are computed with CMTM algebra, never nested time AD.
This JAX implementation does not call mathrobo's NumPy-cached CMTM methods.
"""
from math import factorial

import jax.numpy as jnp

from robokots.core.kernels.inertia import spatial_inertia
from robokots.core.state.spec import keys_torque
from robokots.outward.diff.dynamics_jax import _ad, _relative_transform
from .time_autodiff import _validate_robot, _flat_function


def block_toeplitz(coefficients):
    """Lower Toeplitz matrix representing multiplication of Taylor series."""
    zero = jnp.zeros_like(coefficients[0])
    count = len(coefficients)
    return jnp.concatenate([
        jnp.concatenate([coefficients[i-j] if i >= j else zero for j in range(count)], axis=1)
        for i in range(count)
    ], axis=0)


def relative_cmtms(joint, q, velocity_coefficients):
    """Inverse-adjoint and wrench CMTMs; coefficients are normalized by n!."""
    x0, y0 = _relative_transform(joint, q)
    x, y = [x0], [y0]
    for n in range(1, len(velocity_coefficients)):
        x.append(-sum(_ad(velocity_coefficients[i]) @ x[n-1-i] for i in range(n)) / n)
        y.append(-sum(y[i] @ _ad(velocity_coefficients[n-1-i]).T for i in range(n)) / n)
    return block_toeplitz(x), block_toeplitz(y)


def make_cmtm_ad_value(robot, state, order, gravity=(0., 0., 0.)):
    """One joint torque/time derivative from explicit spatial CMTM matrices.

    Returns ordinary derivatives. Apply jacfwd/jacrev externally, optionally
    jit. No custom AD rules or sparse matrix storage/graph coloring are used.
    """
    _validate_robot(robot)
    if state.owner_type != "joint" or state.data_type not in keys_torque:
        raise ValueError("CMTM AD requires one joint torque/torque_diffN state")
    if robot.joint(state.owner_name) is None:
        raise ValueError("Unknown joint")
    if state.frame_name not in (None, "local"):
        raise ValueError("CMTM AD torque uses the local joint axis")
    if order < state.time_order:
        raise ValueError("Insufficient motion order")
    gravity = jnp.asarray(gravity)
    if gravity.shape != (3,):
        raise ValueError("gravity must have shape (3,)")
    count = order - 1
    scales = jnp.asarray([factorial(n) for n in range(count)])
    inertias = {link.name: jnp.asarray(spatial_inertia(link.mass, link.inertia, link.cog)) for link in robot.links}
    root = robot.links[0].name
    derivative_order = state.key_order - 1
    gravity_series = jnp.zeros((count, 6)).at[0, 3:].set(gravity).reshape(-1)

    def compute(motion):
        velocities = {root: jnp.zeros((count, 6), dtype=motion.dtype)}
        world_inverse = {root: jnp.eye(6 * count, dtype=motion.dtype)}
        relative_wrench = {}
        for joint in robot.joints:
            blocks = motion[joint.dof_index:joint.dof_index + joint.dof, :]
            relative_velocity = (blocks[:, 1:].T @ jnp.asarray(joint.select_mat).T) / scales[:, None]
            x, y = relative_cmtms(joint, blocks[:, 0], relative_velocity)
            parent = robot.links[joint.parent_link_id].name
            child = robot.links[joint.child_link_id].name
            velocities[child] = (x @ velocities[parent].reshape(-1)).reshape(count, 6) + relative_velocity
            world_inverse[child] = x @ world_inverse[parent]
            relative_wrench[joint.name] = y

        forces = {}
        for link in robot.links:
            velocity = velocities[link.name]
            momentum = velocity @ inertias[link.name].T
            # Differentiation of Taylor coefficients is multiplication by n+1.
            rate = jnp.arange(1, count)[:, None] * momentum[1:]
            ad_dual = block_toeplitz([_ad(v).T for v in velocity[:-1]])
            convective = (ad_dual @ momentum[:-1].reshape(-1)).reshape(count-1, 6)
            local_gravity = (world_inverse[link.name] @ gravity_series).reshape(count, 6)
            forces[link.name] = rate - convective - local_gravity[:-1] @ inertias[link.name].T

        joint_forces = {}
        size = 6 * (count - 1)
        for joint in reversed(robot.joints):
            child = robot.links[joint.child_link_id]
            force = forces[child.name].reshape(-1)
            for child_id in child.child_joint_ids:
                descendant = robot.joints[child_id].name
                force = force + relative_wrench[descendant][:size, :size] @ joint_forces[descendant]
            joint_forces[joint.name] = force
        selected = joint_forces[state.owner_name].reshape(count-1, 6)[derivative_order]
        return factorial(derivative_order) * (jnp.asarray(robot.joint(state.owner_name).select_mat).T @ selected)

    return _flat_function(compute, robot, order)
