import numpy as np
from mathrobo import CMVector, Factorial

from robokots.core import RobotStruct
from robokots.core.state import StateType, dim_to_dof, data_type_dof, data_type_offset
from robokots.core.state import keys_kinematics, keys_momentum, keys_force, keys_torque
from robokots.core.state_dict import (
    extract_dict_total_link_cmvec,
    state_dict_to_cmtm,
    state_dict_to_cmtm_wrench,
    state_dict_to_cmvec,
    state_dict_to_rel_cmtm,
)
from robokots.core.models.kinematics.kinematics_matrix import joint_select_diag_mat
from robokots.core.models.dynamics.base import spatial_inertia
from robokots.core.models.dynamics.dynamics_matrix import (
    inertia_diag_mat,
    partial_link_sp_vel_to_force_grad_mat,
    partial_momentum_to_force_grad_mat,
)
from robokots.core.models.whole_body.total_dynamics_mat import (
    total_world_link_cmtm_wrench_matvec,
    total_world_link_wrench_to_world_joint_wrench_matvec,
)
from .outward_total_gradient import (
    _batch_selected_coord_to_link_vel_grad_mat,
    _is_batched_kinematics_state,
    outward_jacobian,
)


def _transpose_total_coord_arrange_vec(
    robot: RobotStruct,
    vec: np.ndarray,
    out_order: int = 3,
    in_order: int = 3,
) -> np.ndarray:
    result = np.zeros(robot.joint_dof * in_order)
    for joint in robot.joints:
        if joint.dof == 0:
            continue
        in_start = joint.dof_index * in_order
        out_start = joint.dof_index * out_order
        result[in_start:in_start + joint.dof*out_order] += vec[out_start:out_start + joint.dof*out_order]
    return result


def _transpose_total_coord_to_joint_tan_vel_grad_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    result = np.zeros(robot.joint_dof * order)
    for i, joint in enumerate(robot.joints):
        if joint.dof == 0:
            continue
        joint_cmtm = state_dict_to_cmtm(state, joint.name, "joint", order)
        block = joint_cmtm.tangent_mat() @ joint_select_diag_mat(joint.select_mat, order)
        out_start = joint.dof_index * order
        result[out_start:out_start + joint.dof*order] += block.T @ vec[i*n_:(i+1)*n_]
    return result


def _transpose_total_joint_tan_vel_to_link_tan_vel_grad_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    result = np.zeros(robot.joint_num * n_)
    for i, link in enumerate(robot.links):
        link_route = []
        joint_route = []
        robot.route_target_link(link, link_route, joint_route)
        adj_link = vec[i*n_:(i+1)*n_]
        for j in joint_route:
            joint = robot.joints[j]
            rel_cmtm = state_dict_to_rel_cmtm(state, link.name, robot.links[joint.child_link_id].name, "link", order)
            result[j*n_:(j+1)*n_] += rel_cmtm.mat_adj().T @ adj_link
    return result


def _transpose_total_joint_tan_vel_to_link_vel_grad_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    result = np.zeros(robot.joint_num * n_)
    for i, link in enumerate(robot.links):
        link_route = []
        joint_route = []
        robot.route_target_link(link, link_route, joint_route)
        tangent_mat_inv = state_dict_to_cmtm(state, link.name, "link", order).tangent_mat_inv()
        adj_link = vec[i*n_:(i+1)*n_]
        for j in joint_route:
            joint = robot.joints[j]
            rel_cmtm = state_dict_to_rel_cmtm(state, link.name, robot.links[joint.child_link_id].name, "link", order)
            block = tangent_mat_inv @ rel_cmtm.mat_adj()
            result[j*n_:(j+1)*n_] += block.T @ adj_link
    return result


def _transpose_total_joint_tan_vel_to_link_sp_vel_grad_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    dof = dim_to_dof(dim)
    n_j = dof * order
    n_l = dof * (order - 1)
    result = np.zeros(robot.joint_num * n_j)
    for i, link in enumerate(robot.links):
        link_route = []
        joint_route = []
        robot.route_target_link(link, link_route, joint_route)
        tangent_mat_inv_sp = state_dict_to_cmtm(state, link.name, "link", order).tangent_mat_inv()[dof:]
        adj_link = vec[i*n_l:(i+1)*n_l]
        for j in joint_route:
            joint = robot.joints[j]
            rel_cmtm = state_dict_to_rel_cmtm(state, link.name, robot.links[joint.child_link_id].name, "link", order)
            block = tangent_mat_inv_sp @ rel_cmtm.mat_adj()
            result[j*n_j:(j+1)*n_j] += block.T @ adj_link
    return result


def _transpose_total_coord_to_link_tan_vel_grad_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    out_order: int = 3,
    in_order=None,
    dim: int = 3,
) -> np.ndarray:
    adj_joint_tan = _transpose_total_joint_tan_vel_to_link_tan_vel_grad_matvec(
        robot, state, vec, order=out_order, dim=dim
    )
    adj_coord = _transpose_total_coord_to_joint_tan_vel_grad_matvec(
        robot, state, adj_joint_tan, order=out_order, dim=dim
    )
    if in_order is None:
        return adj_coord
    return _transpose_total_coord_arrange_vec(robot, adj_coord, out_order=out_order, in_order=in_order)


def _transpose_total_coord_to_link_vel_grad_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    adj_joint_tan = _transpose_total_joint_tan_vel_to_link_vel_grad_matvec(
        robot, state, vec, order=order, dim=dim
    )
    return _transpose_total_coord_to_joint_tan_vel_grad_matvec(
        robot, state, adj_joint_tan, order=order, dim=dim
    )


def _transpose_selected_coord_to_link_vel_grad_matvec(
    robot: RobotStruct,
    state: dict,
    link_adjoint_blocks: dict[str, np.ndarray],
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    adj_joint_tan = np.zeros(robot.joint_num * n_)
    for link_name, adj_link in link_adjoint_blocks.items():
        link = robot.link(link_name)
        if link is None:
            raise ValueError(f"Invalid link name: {link_name}")
        link_route = []
        joint_route = []
        robot.route_target_link(link, link_route, joint_route)
        tangent_mat_inv = state_dict_to_cmtm(state, link.name, "link", order).tangent_mat_inv()
        for j in joint_route:
            joint = robot.joints[j]
            rel_cmtm = state_dict_to_rel_cmtm(state, link.name, robot.links[joint.child_link_id].name, "link", order)
            block = tangent_mat_inv @ rel_cmtm.mat_adj()
            adj_joint_tan[j*n_:(j+1)*n_] += block.T @ adj_link
    return _transpose_total_coord_to_joint_tan_vel_grad_matvec(
        robot, state, adj_joint_tan, order=order, dim=dim
    )


def _transpose_total_coord_to_link_sp_vel_grad_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    adj_joint_tan = _transpose_total_joint_tan_vel_to_link_sp_vel_grad_matvec(
        robot, state, vec, order=order, dim=dim
    )
    return _transpose_total_coord_to_joint_tan_vel_grad_matvec(
        robot, state, adj_joint_tan, order=order, dim=dim
    )


def _transpose_total_link_inertia_matvec(
    robot: RobotStruct,
    vec: np.ndarray,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    result = np.zeros(robot.link_num * n_)
    for i, link in enumerate(robot.links):
        start = i * n_
        inertia = inertia_diag_mat(spatial_inertia(link.mass, link.inertia, link.cog), order)
        result[start:start+n_] = inertia.T @ vec[start:start+n_]
    return result


def _transpose_total_coord_to_link_momentum_grad_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    adj_sp_vel = _transpose_total_link_inertia_matvec(robot, vec, order=order-1, dim=dim)
    return _transpose_total_coord_to_link_sp_vel_grad_matvec(
        robot, state, adj_sp_vel, order=order, dim=dim
    )


def _transpose_total_factorial_matvec(num: int, vec: np.ndarray, order: int, submat_dim: int = 6) -> np.ndarray:
    n_ = submat_dim * order
    result = np.zeros(num * n_)
    mat = Factorial.mat(order, submat_dim).T
    for i in range(num):
        start = i * n_
        result[start:start+n_] = mat @ vec[start:start+n_]
    return result


def _transpose_total_factorial_mat_inv_vec(num: int, vec: np.ndarray, order: int, submat_dim: int = 6) -> np.ndarray:
    n_ = submat_dim * order
    result = np.zeros(num * n_)
    mat = Factorial.mat_inv(order, submat_dim).T
    for i in range(num):
        start = i * n_
        result[start:start+n_] = mat @ vec[start:start+n_]
    return result


def _transpose_total_world_link_cmtm_wrench_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    result = np.zeros(robot.link_num * n_)
    for i, link in enumerate(robot.links):
        start = i * n_
        cmtm_wrench = state_dict_to_cmtm_wrench(state, link.name, "link", order)
        result[start:start+n_] = cmtm_wrench.mat_adj().T @ vec[start:start+n_]
    return result


def _transpose_total_world_joint_cmtm_wrench_inv_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    result = np.zeros(robot.joint_num * n_)
    for i, joint in enumerate(robot.joints):
        start = i * n_
        cmtm_wrench = state_dict_to_cmtm_wrench(state, robot.links[joint.child_link_id].name, "link", order)
        result[start:start+n_] = cmtm_wrench.mat_inv_adj().T @ vec[start:start+n_]
    return result


def _transpose_total_world_link_wrench_to_world_joint_wrench_matvec(
    robot: RobotStruct,
    vec: np.ndarray,
    order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    result = np.zeros(robot.link_num * n_)
    for i, joint in enumerate(robot.joints):
        link_route = []
        joint_route = []
        robot.route_end_joints(joint, link_route, joint_route)
        adj_joint = vec[i*n_:(i+1)*n_]
        for j in link_route:
            result[j*n_:(j+1)*n_] += adj_joint
    return result


def _transpose_total_joint_wrench_to_joint_torque_matvec(
    robot: RobotStruct,
    vec: np.ndarray,
    torque_order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * torque_order
    result = np.zeros(robot.joint_num * n_)
    for i, joint in enumerate(robot.joints):
        if joint.dof == 0:
            continue
        in_start = joint.dof_index * torque_order
        in_end = in_start + joint.dof * torque_order
        select = joint_select_diag_mat(joint.select_mat, torque_order)
        result[i*n_:(i+1)*n_] += select @ vec[in_start:in_end]
    return result


def _transpose_total_partial_momentum_to_force_grad_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    force_order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * force_order
    m_ = dim_to_dof(dim) * (force_order + 1)
    result = np.zeros(robot.link_num * m_)
    for i, link in enumerate(robot.links):
        in_start = i * n_
        out_start = i * m_
        cmtm = state_dict_to_cmtm(state, link.name, "link", force_order + 1)
        mat = partial_momentum_to_force_grad_mat(cmtm, force_order=force_order, dim=dim)
        result[out_start:out_start+m_] = mat.T @ vec[in_start:in_start+n_]
    return result


def _transpose_joint_partial_momentum_to_force_grad_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    force_order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * force_order
    m_ = dim_to_dof(dim) * (force_order + 1)
    result = np.zeros(robot.joint_num * m_)
    for i, joint in enumerate(robot.joints):
        in_start = i * n_
        out_start = i * m_
        link = robot.links[joint.child_link_id]
        cmtm = state_dict_to_cmtm(state, link.name, "link", force_order + 1)
        mat = partial_momentum_to_force_grad_mat(cmtm, force_order=force_order, dim=dim)
        result[out_start:out_start+m_] = mat.T @ vec[in_start:in_start+n_]
    return result


def _transpose_total_partial_link_sp_vel_to_link_force_grad_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    force_order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * force_order
    m_ = dim_to_dof(dim) * (force_order + 2)
    result = np.zeros(robot.link_num * m_)
    for i, link in enumerate(robot.links):
        in_start = i * n_
        out_start = i * m_
        link_momentum = state_dict_to_cmvec(state, link.name, "link", "momentum", force_order)
        mat = partial_link_sp_vel_to_force_grad_mat(link_momentum, force_order=force_order, dim=dim)
        result[out_start:out_start+m_] = mat.T @ vec[in_start:in_start+n_]
    return result


def _transpose_total_partial_link_sp_vel_to_joint_force_grad_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    force_order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * force_order
    m_ = dim_to_dof(dim) * (force_order + 2)
    result = np.zeros(robot.joint_num * m_)
    for i, joint in enumerate(robot.joints):
        in_start = i * n_
        out_start = i * m_
        joint_momentum = state_dict_to_cmvec(state, joint.name, "joint", "momentum", force_order)
        mat = partial_link_sp_vel_to_force_grad_mat(joint_momentum, force_order=force_order, dim=dim)
        result[out_start:out_start+m_] = mat.T @ vec[in_start:in_start+n_]
    return result


def _transpose_total_partial_link_momentum_to_world_link_momentum_grad_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    dof = dim_to_dof(dim)
    adj_factor = _transpose_total_factorial_matvec(robot.link_num, vec, order-1, dof)
    adj_cmtm = _transpose_total_world_link_cmtm_wrench_matvec(robot, state, adj_factor, order-1, dim)
    return _transpose_total_factorial_mat_inv_vec(robot.link_num, adj_cmtm, order-1, dof)


def _transpose_total_partial_link_tan_vel_to_world_link_momentum_grad_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    dof = dim_to_dof(dim)
    total_local_link_momentum = extract_dict_total_link_cmvec(state, robot.link_names, "momentum", order-1)
    adj_factor = _transpose_total_factorial_matvec(robot.link_num, vec, order-1, dof)
    n_ = dof * (order - 1)
    result = np.zeros(robot.link_num * n_)
    total_cm_vecs = total_local_link_momentum.reshape(robot.link_num, n_)
    for i, link in enumerate(robot.links):
        start = i * n_
        arb_v = CMVector.set_cmvecs(total_cm_vecs[i].reshape(order-1, -1))
        mat = state_dict_to_cmtm_wrench(state, link.name, "link", order-1).mat_var_x_arb_vec_jacob(arb_v, frame="bframe")
        result[start:start+n_] = mat.T @ adj_factor[start:start+n_]
    return result


def _transpose_total_partial_world_joint_momentum_to_joint_momentum_grad_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    dof = dim_to_dof(dim)
    adj_factor = _transpose_total_factorial_matvec(robot.joint_num, vec, order-1, dof)
    adj_cmtm = _transpose_total_world_joint_cmtm_wrench_inv_matvec(robot, state, adj_factor, order-1, dim)
    return _transpose_total_factorial_mat_inv_vec(robot.joint_num, adj_cmtm, order-1, dof)


def _transpose_total_partial_link_tan_vel_to_joint_momentum_grad_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    dof = dim_to_dof(dim)
    total_local_link_momentum = extract_dict_total_link_cmvec(state, robot.link_names, "momentum", order-1)
    world_link_momentum = total_world_link_cmtm_wrench_matvec(robot, state, total_local_link_momentum, order-1, dim)
    total_world_joint_momentum = total_world_link_wrench_to_world_joint_wrench_matvec(robot, world_link_momentum, order-1, dim)
    adj_factor = _transpose_total_factorial_matvec(robot.joint_num, vec, order-1, dof)
    n_ = dof * (order - 1)
    result = np.zeros(robot.joint_num * n_)
    total_cm_vecs = total_world_joint_momentum.reshape(robot.joint_num, n_)
    for i, joint in enumerate(robot.joints):
        start = i * n_
        arb_v = CMVector.set_cmvecs(total_cm_vecs[i].reshape(order-1, -1))
        c_link = robot.links[joint.child_link_id]
        cmtm_wrench = state_dict_to_cmtm_wrench(state, c_link.name, "link", order-1)
        mat = cmtm_wrench.mat_inv_var_x_arb_vec_jacob(arb_v, frame="bframe")
        result[start:start+n_] = mat.T @ adj_factor[start:start+n_]
    return result


def outward_kinematics_jacobian_transpose_matvec(
    robot: RobotStruct,
    state: dict,
    state_type_list: list[StateType],
    vec: np.ndarray,
    max_time_order=None,
    dim: int = 3,
) -> np.ndarray:
    kine_state_type_list = StateType.filter_list_by_kinematics(state_type_list)
    if max_time_order is None:
        max_time_order = StateType.max_time_order(kine_state_type_list)

    dim_dof = dim_to_dof(dim)
    if _is_batched_kinematics_state(robot, state, kine_state_type_list, max_time_order):
        link_names = StateType.get_owner_names_from_list(kine_state_type_list)
        links = robot.link_list(link_names)
        mat = _batch_selected_coord_to_link_vel_grad_mat(robot, state, links, order=max_time_order, dim=dim)
        adj_link_vel = np.zeros(mat.shape[:-2] + (len(links) * dim_dof * max_time_order,), dtype=np.asarray(vec).dtype)
        link_offsets = {link.name: i * dim_dof * max_time_order for i, link in enumerate(links)}
        offset_in_vec = 0

        for st in kine_state_type_list:
            link = robot.link(st.owner_name)
            if link is None:
                raise ValueError(f"Invalid link name: {st.owner_name}")

            state_dof = data_type_dof(st.data_type, dim=dim)
            output = vec[..., offset_in_vec : offset_in_vec + state_dof]
            offset_in_vec += state_dof

            base = link_offsets[link.name]
            offset = dim_dof*(st.time_order-1) + data_type_offset(st.data_type) * state_dof
            adj_link_vel[..., base + offset : base + offset + state_dof] += output

        return (np.swapaxes(mat, -1, -2) @ adj_link_vel[..., None])[..., 0]

    link_adjoint_blocks = {}
    offset_in_vec = 0

    for st in kine_state_type_list:
        link = robot.link(st.owner_name)
        if link is None:
            raise ValueError(f"Invalid link name: {st.owner_name}")

        state_dof = data_type_dof(st.data_type, dim=dim)
        output = vec[offset_in_vec : offset_in_vec + state_dof]
        offset_in_vec += state_dof

        offset = dim_dof*(st.time_order-1) + data_type_offset(st.data_type) * state_dof
        if link.name not in link_adjoint_blocks:
            link_adjoint_blocks[link.name] = np.zeros(dim_dof * max_time_order)
        link_adjoint_blocks[link.name][offset : offset + state_dof] += output

    return _transpose_selected_coord_to_link_vel_grad_matvec(
        robot, state, link_adjoint_blocks, order=max_time_order, dim=dim
    )


def outward_jacobian_transpose_matvec(
    robot: RobotStruct,
    state: dict,
    state_type_list: list[StateType],
    vec: np.ndarray,
    max_time_order=None,
    dim: int = 3,
) -> np.ndarray:
    if StateType.is_list_all_in_kinematics(state_type_list):
        return outward_kinematics_jacobian_transpose_matvec(
            robot, state, state_type_list, vec, max_time_order, dim=dim
        )

    if max_time_order is None:
        max_time_order = StateType.max_time_order(state_type_list)

    if _is_batched_kinematics_state(
        robot,
        state,
        [StateType("link", robot.links[0].name, "vel")],
        max_time_order,
    ):
        jacob = outward_jacobian(robot, state, state_type_list, max_time_order=max_time_order, dim=dim)
        return (np.swapaxes(jacob, -1, -2) @ vec[..., None])[..., 0]

    dof = dim_to_dof(dim)
    force_order = max_time_order - 2
    motion_dim = robot.dof * max_time_order
    adj_kine = np.zeros(robot.link_num * dof * max_time_order)
    adj_tan_kine = np.zeros(robot.link_num * dof * (max_time_order - 1))
    adj_link_mom = np.zeros(robot.link_num * dof * (max_time_order - 1))
    adj_link_wmom = np.zeros_like(adj_link_mom)
    adj_joint_wmom = np.zeros(robot.joint_num * dof * (max_time_order - 1))
    adj_joint_mom = np.zeros_like(adj_joint_wmom)
    adj_link_force = np.zeros(robot.link_num * dof * max(force_order, 0))
    adj_joint_force = np.zeros(robot.joint_num * dof * max(force_order, 0))
    adj_joint_torque = np.zeros(robot.dof * max(force_order, 0))

    offset_in_vec = 0
    for st in state_type_list:
        if st.owner_type == "link":
            link = robot.link(st.owner_name)
            if link is None:
                raise ValueError(f"Invalid link name: {st.owner_name}")
        elif st.owner_type == "joint":
            joint = robot.joint(st.owner_name)
            if joint is None:
                raise ValueError(f"Invalid joint name: {st.owner_name}")
        else:
            raise ValueError(f"Invalid owner_type: {st.owner_type}")

        order = st.key_order - 1
        if st.data_type in keys_kinematics:
            state_dof = data_type_dof(st.data_type, dim=dim)
            part = vec[offset_in_vec : offset_in_vec + state_dof]
            offset_in_vec += state_dof
            base = link.id * dof * max_time_order
            state_offset = dof*(st.time_order-1) + data_type_offset(st.data_type) * state_dof
            adj_kine[base + state_offset : base + state_offset + state_dof] += part
        elif st.data_type in keys_momentum:
            part = vec[offset_in_vec : offset_in_vec + dof]
            offset_in_vec += dof
            if st.owner_type == "link":
                base = link.id * dof * (max_time_order - 1)
                target = adj_link_wmom if st.frame_name == "world" else adj_link_mom
            else:
                base = joint.id * dof * (max_time_order - 1)
                target = adj_joint_wmom if st.frame_name == "world" else adj_joint_mom
            target[base + dof*order : base + dof*(order+1)] += part
        elif st.data_type in keys_force:
            if max_time_order < 3:
                raise ValueError("force jacobian transpose matvec requires max_time_order >= 3")
            part = vec[offset_in_vec : offset_in_vec + dof]
            offset_in_vec += dof
            if st.owner_type == "link":
                base = link.id * dof * force_order
                adj_link_force[base + dof*order : base + dof*(order+1)] += part
            else:
                base = joint.id * dof * force_order
                adj_joint_force[base + dof*order : base + dof*(order+1)] += part
        elif st.data_type in keys_torque:
            if max_time_order < 3:
                raise ValueError("torque jacobian transpose matvec requires max_time_order >= 3")
            if st.owner_type != "joint":
                raise ValueError("torque can be specified only for joint owner type")
            part = vec[offset_in_vec : offset_in_vec + joint.dof]
            offset_in_vec += joint.dof
            base = joint.dof_index * force_order
            adj_joint_torque[base + joint.dof*order : base + joint.dof*(order+1)] += part
        else:
            raise ValueError(f"Unsupported data_type for jacobian transpose matvec: {st.data_type}")

    if max_time_order >= 3:
        adj_joint_force += _transpose_total_joint_wrench_to_joint_torque_matvec(
            robot, adj_joint_torque, torque_order=force_order, dim=dim
        )

        adj_joint_mom += _transpose_joint_partial_momentum_to_force_grad_matvec(
            robot, state, adj_joint_force, force_order=force_order, dim=dim
        )

        kine_tail = _transpose_total_partial_link_sp_vel_to_joint_force_grad_matvec(
            robot, state, adj_joint_force, force_order=force_order, dim=dim
        )
        adj_kine[max_time_order*dof:] += kine_tail

        adj_link_mom += _transpose_total_partial_momentum_to_force_grad_matvec(
            robot, state, adj_link_force, force_order=force_order, dim=dim
        )
        adj_kine += _transpose_total_partial_link_sp_vel_to_link_force_grad_matvec(
            robot, state, adj_link_force, force_order=force_order, dim=dim
        )

    adj_joint_wmom += _transpose_total_partial_world_joint_momentum_to_joint_momentum_grad_matvec(
        robot, state, adj_joint_mom, order=max_time_order, dim=dim
    )

    tan_tail = _transpose_total_partial_link_tan_vel_to_joint_momentum_grad_matvec(
        robot, state, adj_joint_mom, order=max_time_order, dim=dim
    )
    adj_tan_kine[(max_time_order-1)*dof:] += tan_tail

    adj_link_wmom += _transpose_total_world_link_wrench_to_world_joint_wrench_matvec(
        robot, adj_joint_wmom, order=max_time_order-1, dim=dim
    )

    adj_link_mom += _transpose_total_partial_link_momentum_to_world_link_momentum_grad_matvec(
        robot, state, adj_link_wmom, order=max_time_order, dim=dim
    )

    adj_tan_kine += _transpose_total_partial_link_tan_vel_to_world_link_momentum_grad_matvec(
        robot, state, adj_link_wmom, order=max_time_order, dim=dim
    )

    result = np.zeros(motion_dim)
    result += _transpose_total_coord_to_link_vel_grad_matvec(
        robot, state, adj_kine, order=max_time_order, dim=dim
    )
    result += _transpose_total_coord_to_link_tan_vel_grad_matvec(
        robot, state, adj_tan_kine, out_order=max_time_order-1, in_order=max_time_order, dim=dim
    )
    result += _transpose_total_coord_to_link_momentum_grad_matvec(
        robot, state, adj_link_mom, order=max_time_order, dim=dim
    )
    return result
