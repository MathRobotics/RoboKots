import numpy as np
from mathrobo import CMVector, Factorial

from robokots.core import RobotStruct
from robokots.core.state import dim_to_dof
from robokots.core.state_dict import state_dict_to_cmtm, state_dict_to_cmtm_wrench

from ..cmtm_apply import apply_mat_inv_adj
from ..dynamics.base import spatial_inertia
from ..dynamics.dynamics_matrix import inertia_diag_mat
from .topology_layout import take_joint_child_link_blocks
from .total_kinematics_grad_mat import total_coord_to_link_tan_vel_grad_mat


def state_gravity(state, gravity=None) -> np.ndarray:
    """Return a validated world-frame gravity vector for an outward state."""
    if gravity is None:
        gravity = getattr(state, "gravity", np.zeros(3))
    gravity = np.asarray(gravity, dtype=float)
    if gravity.shape != (3,):
        raise ValueError(f"gravity must have shape (3,), got {gravity.shape}.")
    if not np.all(np.isfinite(gravity)):
        raise ValueError("gravity must contain only finite values.")
    return gravity


def _world_gravity_cmvector(gravity: np.ndarray, force_order: int) -> CMVector:
    world_vecs = np.zeros((force_order, 6), dtype=float)
    world_vecs[0, 3:] = gravity
    return CMVector(world_vecs)


def _validate_gravity_gradient_args(force_order: int, dim: int) -> None:
    if dim != 3:
        raise NotImplementedError("gravity force gradients currently support dim=3 only")
    if force_order < 1:
        raise ValueError("force_order must be at least 1")


def total_link_gravity_force(
    robot: RobotStruct,
    state,
    gravity=None,
    force_order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    """Stack local gravity-wrench CM vectors through ``force_order``."""
    _validate_gravity_gradient_args(force_order, dim)
    gravity = state_gravity(state, gravity)
    world_gravity = _world_gravity_cmvector(gravity, force_order)
    block_size = 6 * force_order
    first_cmtm = state_dict_to_cmtm(
        state, robot.links[0].name, "link", force_order
    )
    batch_shape = np.asarray(first_cmtm.elem_mat()).shape[:-2]
    result = np.zeros(batch_shape + (robot.link_num * block_size,))

    for i, link in enumerate(robot.links):
        link_cmtm = state_dict_to_cmtm(state, link.name, "link", force_order)
        local_gravity = apply_mat_inv_adj(link_cmtm, world_gravity.cm_vec())
        inertia = spatial_inertia(link.mass, link.inertia, link.cog)
        block = slice(i * block_size, (i + 1) * block_size)
        result[..., block] = (
            -inertia_diag_mat(inertia, force_order) @ local_gravity[..., None]
        )[..., 0]
    return result


def total_coord_to_link_gravity_force_grad_mat(
    robot: RobotStruct,
    state,
    gravity=None,
    force_order: int = 1,
    dim: int = 3,
    link_pose_grad=None,
) -> np.ndarray:
    """Differentiate local link gravity wrenches with respect to motion."""
    if link_pose_grad is None:
        link_pose_grad = total_coord_to_link_tan_vel_grad_mat(
            robot,
            state,
            out_order=force_order,
            in_order=force_order + 2,
            dim=dim,
        )
    link_grad, _ = total_link_pose_to_gravity_force_grad_matmul_rhs(
        robot,
        state,
        gravity,
        link_pose_grad,
        force_order=force_order,
        dim=dim,
    )
    return link_grad


def total_coord_to_joint_gravity_force_grad_mat(
    robot: RobotStruct,
    state,
    gravity=None,
    force_order: int = 1,
    dim: int = 3,
    link_pose_grad=None,
) -> np.ndarray:
    """Transport, aggregate, and differentiate link gravity wrenches."""
    if link_pose_grad is None:
        link_pose_grad = total_coord_to_link_tan_vel_grad_mat(
            robot,
            state,
            out_order=force_order,
            in_order=force_order + 2,
            dim=dim,
        )
    _, joint_grad = total_link_pose_to_gravity_force_grad_matmul_rhs(
        robot,
        state,
        gravity,
        link_pose_grad,
        force_order=force_order,
        dim=dim,
    )
    return joint_grad


def _cmtm_var_matmul_rhs(
    cmtm,
    arb_vec,
    rhs: np.ndarray,
    inverse: bool = False,
) -> np.ndarray:
    method_name = (
        "mat_inv_var_x_arb_vec_matmul_rhs"
        if inverse
        else "mat_var_x_arb_vec_matmul_rhs"
    )
    fast = getattr(cmtm, method_name, None)
    if fast is not None:
        return fast(arb_vec, rhs, frame="bframe")
    method = (
        cmtm.mat_inv_var_x_arb_vec_jacob
        if inverse
        else cmtm.mat_var_x_arb_vec_jacob
    )
    return method(arb_vec, frame="bframe") @ rhs


def _factorial_blocks(
    array: np.ndarray,
    count: int,
    order: int,
    inverse: bool = False,
) -> np.ndarray:
    array = np.asarray(array)
    block_size = 6 * order
    factorial_mat = (
        Factorial.mat_inv(order, 6) if inverse else Factorial.mat(order, 6)
    )
    result = np.zeros_like(array)
    for i in range(count):
        block = slice(i * block_size, (i + 1) * block_size)
        result[..., block, :] = factorial_mat @ array[..., block, :]
    return result


def _aggregate_link_blocks(
    robot: RobotStruct,
    array: np.ndarray,
    block_size: int,
) -> np.ndarray:
    array = np.asarray(array)
    result = np.zeros(
        array.shape[:-2] + (robot.joint_num * block_size, array.shape[-1]),
        dtype=array.dtype,
    )
    for i, joint in enumerate(robot.joints):
        link_route = []
        joint_route = []
        robot.route_end_joints(joint, link_route, joint_route)
        out = slice(i * block_size, (i + 1) * block_size)
        for link_id in link_route:
            source = slice(link_id * block_size, (link_id + 1) * block_size)
            result[..., out, :] += array[..., source, :]
    return result


def total_link_pose_to_gravity_force_grad_matmul_rhs(
    robot: RobotStruct,
    state,
    gravity,
    link_pose_rhs: np.ndarray,
    force_order: int = 1,
    dim: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply gravity-force gradients to link-pose directions.

    The input is the link tangent-pose directional derivative. The returned
    pair contains local link-force and local joint-force directional
    derivatives. Scalar, batched, vector, and matrix right-hand sides share
    this implementation.
    """
    _validate_gravity_gradient_args(force_order, dim)
    gravity = state_gravity(state, gravity)
    link_pose_rhs = np.asarray(link_pose_rhs)
    block_size = dim_to_dof(dim) * force_order
    factorial_mat = Factorial.mat(force_order, 6)
    world_gravity = _world_gravity_cmvector(gravity, force_order)
    local_link_rhs = np.zeros_like(link_pose_rhs)

    for i, link in enumerate(robot.links):
        block = slice(i * block_size, (i + 1) * block_size)
        link_cmtm = state_dict_to_cmtm(state, link.name, "link", force_order)
        varied = _cmtm_var_matmul_rhs(
            link_cmtm,
            world_gravity,
            link_pose_rhs[..., block, :],
            inverse=True,
        )
        inertia = spatial_inertia(link.mass, link.inertia, link.cog)
        local_link_rhs[..., block, :] = (
            factorial_mat @ (-inertia_diag_mat(inertia, force_order) @ varied)
        )

    local_link_force = total_link_gravity_force(
        robot, state, gravity, force_order=force_order, dim=dim
    )
    world_link_force = np.zeros_like(local_link_force)
    transformed_link_rhs = np.zeros_like(local_link_rhs)
    varied_link_rhs = np.zeros_like(local_link_rhs)
    inv_local_link_rhs = _factorial_blocks(
        local_link_rhs, robot.link_num, force_order, inverse=True
    )
    local_force_blocks = local_link_force.reshape(
        local_link_force.shape[:-1] + (robot.link_num, block_size)
    )

    for i, link in enumerate(robot.links):
        block = slice(i * block_size, (i + 1) * block_size)
        cmtm = state_dict_to_cmtm_wrench(
            state, link.name, "link", force_order
        )
        world_link_force[..., block] = (
            cmtm.mat_adj() @ local_link_force[..., block, None]
        )[..., 0]
        transformed_link_rhs[..., block, :] = (
            cmtm.mat_adj() @ inv_local_link_rhs[..., block, :]
        )
        arb_vec = CMVector.set_cmvecs(
            local_force_blocks[..., i, :].reshape(
                local_force_blocks.shape[:-2] + (force_order, 6)
            )
        )
        varied_link_rhs[..., block, :] = _cmtm_var_matmul_rhs(
            cmtm, arb_vec, link_pose_rhs[..., block, :]
        )

    world_link_rhs = _factorial_blocks(
        transformed_link_rhs + varied_link_rhs,
        robot.link_num,
        force_order,
    )
    world_joint_force = _aggregate_link_blocks(
        robot, world_link_force[..., None], block_size
    )[..., 0]
    world_joint_rhs = _aggregate_link_blocks(robot, world_link_rhs, block_size)
    inv_world_joint_rhs = _factorial_blocks(
        world_joint_rhs, robot.joint_num, force_order, inverse=True
    )
    local_joint_rhs = np.zeros_like(world_joint_rhs)
    varied_joint_rhs = np.zeros_like(world_joint_rhs)
    child_pose_rhs = take_joint_child_link_blocks(
        link_pose_rhs, robot, block_size, axis=-2
    )
    joint_force_blocks = world_joint_force.reshape(
        world_joint_force.shape[:-1] + (robot.joint_num, block_size)
    )

    for i, joint in enumerate(robot.joints):
        block = slice(i * block_size, (i + 1) * block_size)
        child_link = robot.links[joint.child_link_id]
        cmtm = state_dict_to_cmtm_wrench(
            state, child_link.name, "link", force_order
        )
        local_joint_rhs[..., block, :] = (
            cmtm.mat_inv_adj() @ inv_world_joint_rhs[..., block, :]
        )
        arb_vec = CMVector.set_cmvecs(
            joint_force_blocks[..., i, :].reshape(
                joint_force_blocks.shape[:-2] + (force_order, 6)
            )
        )
        varied_joint_rhs[..., block, :] = _cmtm_var_matmul_rhs(
            cmtm,
            arb_vec,
            child_pose_rhs[..., block, :],
            inverse=True,
        )

    return local_link_rhs, _factorial_blocks(
        local_joint_rhs + varied_joint_rhs,
        robot.joint_num,
        force_order,
    )
