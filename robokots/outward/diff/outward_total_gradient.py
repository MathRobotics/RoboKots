#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025. 6.20 Created by T.Ishigaki
# outward computation module from motion and robot_model to state by matrix formulation

import numpy as np
from mathrobo import CMTM, CMVector, Factorial, SE3wrench

from robokots.core import RobotStruct
from robokots.core.state import StateType, dim_to_dof, data_type_dof, data_type_offset
from robokots.core.state import keys_kinematics, keys_momentum, keys_force, keys_torque
from robokots.core.state_dict import extract_dict_total_link_cmvec, state_dict_to_cmtm, state_dict_to_cmtm_wrench, state_dict_to_cmvec, state_dict_to_rel_cmtm
from robokots.core.models.kinematics.kinematics_matrix import joint_select_diag_mat

from robokots.core.models.whole_body.total_kinematics_grad_mat import (
    total_coord_to_joint_tan_vel_grad_mat,
    total_coord_to_link_tan_vel_grad_mat,
    total_coord_to_link_tan_vel_grad_matvec,
    total_coord_to_link_vel_grad_mat,
    total_coord_to_link_vel_grad_matvec,
)
from robokots.core.models.whole_body.total_kinematics_mat import total_coord_arrange
from robokots.core.models.whole_body.total_dynamics_grad_mat import (
    total_coord_to_joint_momentum_grad_mat,
    total_coord_to_link_momentum_grad_mat,
    total_coord_to_link_momentum_grad_matvec,
    total_coord_to_world_joint_momentum_grad_mat,
    total_coord_to_world_link_momentum_grad_mat,
)
from robokots.core.models.whole_body.total_dynamics_grad_mat import total_coord_to_link_force_grad_mat, total_coord_to_joint_force_grad_mat, total_coord_to_joint_torque_grad_mat
from robokots.core.models.whole_body.total_dynamics_mat import (
    total_joint_wrench_to_joint_torque_mat,
    total_joint_wrench_to_joint_torque_matvec,
    total_world_link_wrench_to_world_joint_wrench_mat,
    total_world_link_wrench_to_world_joint_wrench_matvec,
)
from robokots.core.models.whole_body.total_partial_grad_mat import (
    total_partial_link_momentum_to_world_link_momentum_grad_mat,
    total_partial_link_momentum_to_world_link_momentum_grad_matvec,
    total_partial_link_sp_vel_to_joint_force_grad_mat,
    total_partial_link_sp_vel_to_joint_force_grad_matvec,
    total_partial_link_sp_vel_to_link_force_grad_mat,
    total_partial_link_sp_vel_to_link_force_grad_matvec,
    total_partial_link_tan_vel_to_joint_momentum_grad_mat,
    total_partial_link_tan_vel_to_joint_momentum_grad_matvec,
    total_partial_link_tan_vel_to_world_link_momentum_grad_mat,
    total_partial_link_tan_vel_to_world_link_momentum_grad_matvec,
    total_partial_momentum_to_force_grad_mat,
    total_partial_momentum_to_force_grad_matvec,
    total_partial_world_joint_momentum_to_joint_momentum_grad_mat,
    total_partial_world_joint_momentum_to_joint_momentum_grad_matvec,
)
from robokots.core.models.whole_body.topology_layout import (
    take_joint_child_link_blocks,
    take_joint_child_link_matrix_blocks,
)
from robokots.core.models.dynamics.base import spatial_inertia
from robokots.core.models.dynamics.dynamics_matrix import (
    inertia_diag_mat,
)
from robokots.core.models.cmtm_apply import apply_mat_adj, apply_mat_inv_adj


def _selected_coord_to_link_vel_grad_mat(
    robot: RobotStruct,
    state: dict,
    links: list,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    joint_tan_mat = total_coord_to_joint_tan_vel_grad_mat(robot, state, order, dim)
    mat = np.zeros((len(links) * n_, robot.dof * order))

    for i, link in enumerate(links):
        if link is None:
            raise ValueError("link_name_list contains invalid link name")

        link_route = []
        joint_route = []
        robot.route_target_link(link, link_route, joint_route)
        link_tan_inv = state_dict_to_cmtm(state, link.name, "link", order).tangent_mat_inv()

        row = i * n_
        for j in joint_route:
            joint = robot.joints[j]
            rel_cmtm = state_dict_to_rel_cmtm(state, link.name, robot.links[joint.child_link_id].name, "link", order)
            col = j * n_
            mat[row:row+n_, :] += (link_tan_inv @ rel_cmtm.mat_adj()) @ joint_tan_mat[col:col+n_, :]

    return mat


def _state_batch_shape(state: dict, owner_name: str, owner_type: str, order: int) -> tuple:
    cmtm = state_dict_to_cmtm(state, owner_name, owner_type, order)
    return np.asarray(cmtm.elem_mat()).shape[:-2]


def _is_batched_kinematics_state(
    robot: RobotStruct,
    state: dict,
    state_type_list: list[StateType],
    order: int,
) -> bool:
    if not state_type_list:
        return False
    first = state_type_list[0]
    owner_type = first.owner_type
    owner_name = first.owner_name
    if owner_type != "link":
        return False
    if robot.link(owner_name) is None:
        return False
    try:
        return len(_state_batch_shape(state, owner_name, owner_type, order)) > 0
    except Exception:
        return False


def _batched_matvec(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat)
    vec = np.asarray(vec)
    if vec.shape[-1] != mat.shape[-1]:
        raise ValueError(
            "vec last dimension must match mat input dimension: "
            f"{vec.shape[-1]} != {mat.shape[-1]}"
        )
    return (mat @ vec[..., None])[..., 0]


def _cmtm_var_jacob_matvec(cmtm, arb_vec, vec: np.ndarray, inverse: bool = False) -> np.ndarray:
    if inverse:
        fast = getattr(cmtm, "mat_inv_var_x_arb_vec_matvec", None)
        if fast is not None:
            return fast(arb_vec, vec, frame="bframe")
        mat = cmtm.mat_inv_var_x_arb_vec_jacob(arb_vec, frame="bframe")
    else:
        fast = getattr(cmtm, "mat_var_x_arb_vec_matvec", None)
        if fast is not None:
            return fast(arb_vec, vec, frame="bframe")
        mat = cmtm.mat_var_x_arb_vec_jacob(arb_vec, frame="bframe")
    return _batched_matvec(mat, vec)


def _cmtm_var_jacob_matmul_rhs(cmtm, arb_vec, rhs: np.ndarray, inverse: bool = False) -> np.ndarray:
    if inverse:
        fast = getattr(cmtm, "mat_inv_var_x_arb_vec_matmul_rhs", None)
        if fast is not None:
            return fast(arb_vec, rhs, frame="bframe")
        mat = cmtm.mat_inv_var_x_arb_vec_jacob(arb_vec, frame="bframe")
    else:
        fast = getattr(cmtm, "mat_var_x_arb_vec_matmul_rhs", None)
        if fast is not None:
            return fast(arb_vec, rhs, frame="bframe")
        mat = cmtm.mat_var_x_arb_vec_jacob(arb_vec, frame="bframe")
    return _matmul_rhs(mat, rhs)


def _matmul_rhs(mat: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat)
    rhs = np.asarray(rhs)
    if rhs.ndim < 2 or rhs.shape[-2] != mat.shape[-1]:
        raise ValueError(
            "rhs must have shape (..., input_dim, rhs_dim), with input_dim matching mat: "
            f"{rhs.shape} vs {mat.shape}"
        )
    return mat @ rhs


def _batch_total_coord_to_joint_tan_vel_grad_mat(
    robot: RobotStruct,
    state: dict,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    dof = dim_to_dof(dim)
    n_ = dof * order
    batch_shape = None
    dtype = float

    for joint in robot.joints:
        joint_cmtm = state_dict_to_cmtm(state, joint.name, "joint", order)
        tangent_mat = np.asarray(joint_cmtm.tangent_mat())
        batch_shape = tangent_mat.shape[:-2]
        dtype = tangent_mat.dtype
        break

    if batch_shape is None:
        batch_shape = ()

    mat = np.zeros(batch_shape + (robot.joint_num * n_, robot.joint_dof * order), dtype=dtype)
    for i, joint in enumerate(robot.joints):
        if joint.dof == 0:
            continue
        joint_cmtm = state_dict_to_cmtm(state, joint.name, "joint", order)
        block = joint_cmtm.tangent_mat() @ joint_select_diag_mat(joint.select_mat, order)
        col_start = joint.dof_index * order
        col_end = (joint.dof_index + joint.dof) * order
        mat[..., i*n_:(i+1)*n_, col_start:col_end] = block

    return mat


def _batch_total_coord_arrange_vec(
    robot: RobotStruct,
    vec: np.ndarray,
    out_order: int = 3,
    in_order: int = 3,
) -> np.ndarray:
    vec = np.asarray(vec)
    arranged = np.zeros(vec.shape[:-1] + (robot.joint_dof * out_order,), dtype=vec.dtype)
    for joint in robot.joints:
        if joint.dof == 0:
            continue
        in_start = joint.dof_index * in_order
        out_start = joint.dof_index * out_order
        arranged[..., out_start:out_start + joint.dof*out_order] = vec[..., in_start:in_start + joint.dof*out_order]
    return arranged


def _batch_total_coord_arrange_rhs(
    robot: RobotStruct,
    rhs: np.ndarray,
    out_order: int = 3,
    in_order: int = 3,
) -> np.ndarray:
    rhs = np.asarray(rhs)
    arranged = np.zeros(rhs.shape[:-2] + (robot.joint_dof * out_order, rhs.shape[-1]), dtype=rhs.dtype)
    for joint in robot.joints:
        if joint.dof == 0:
            continue
        in_start = joint.dof_index * in_order
        out_start = joint.dof_index * out_order
        arranged[..., out_start:out_start + joint.dof*out_order, :] = rhs[..., in_start:in_start + joint.dof*out_order, :]
    return arranged


def _batch_total_coord_to_joint_tan_vel_grad_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    dof = dim_to_dof(dim)
    n_ = dof * order
    vec = np.asarray(vec)
    result = np.zeros(vec.shape[:-1] + (robot.joint_num * n_,), dtype=vec.dtype)

    for i, joint in enumerate(robot.joints):
        if joint.dof == 0:
            continue
        joint_cmtm = state_dict_to_cmtm(state, joint.name, "joint", order)
        coord_start = joint.dof_index * order
        joint_vec = vec[..., coord_start:coord_start + joint.dof*order]
        block = joint_cmtm.tangent_mat() @ joint_select_diag_mat(joint.select_mat, order)
        result[..., i*n_:(i+1)*n_] = _batched_matvec(block, joint_vec)

    return result


def _batch_total_coord_to_joint_tan_vel_grad_matmul_rhs(
    robot: RobotStruct,
    state: dict,
    rhs: np.ndarray,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    dof = dim_to_dof(dim)
    n_ = dof * order
    rhs = np.asarray(rhs)
    result = np.zeros(rhs.shape[:-2] + (robot.joint_num * n_, rhs.shape[-1]), dtype=rhs.dtype)

    for i, joint in enumerate(robot.joints):
        if joint.dof == 0:
            continue
        joint_cmtm = state_dict_to_cmtm(state, joint.name, "joint", order)
        coord_start = joint.dof_index * order
        joint_rhs = rhs[..., coord_start:coord_start + joint.dof*order, :]
        block = joint_cmtm.tangent_mat() @ joint_select_diag_mat(joint.select_mat, order)
        result[..., i*n_:(i+1)*n_, :] = _matmul_rhs(block, joint_rhs)

    return result


def _batch_total_joint_tan_vel_to_link_tan_vel_grad_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    vec = np.asarray(vec)
    result = np.zeros(vec.shape[:-1] + (robot.link_num * n_,), dtype=vec.dtype)

    for i, link in enumerate(robot.links):
        link_route = []
        joint_route = []
        robot.route_target_link(link, link_route, joint_route)
        out_start = i * n_
        for j in joint_route:
            joint = robot.joints[j]
            rel_cmtm = state_dict_to_rel_cmtm(state, link.name, robot.links[joint.child_link_id].name, "link", order)
            result[..., out_start:out_start+n_] += apply_mat_adj(rel_cmtm, vec[..., j*n_:(j+1)*n_])
    return result


def _batch_total_joint_tan_vel_to_link_tan_vel_grad_matmul_rhs(
    robot: RobotStruct,
    state: dict,
    rhs: np.ndarray,
    order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    rhs = np.asarray(rhs)
    result = np.zeros(rhs.shape[:-2] + (robot.link_num * n_, rhs.shape[-1]), dtype=rhs.dtype)

    for i, link in enumerate(robot.links):
        link_route = []
        joint_route = []
        robot.route_target_link(link, link_route, joint_route)
        out_start = i * n_
        for j in joint_route:
            joint = robot.joints[j]
            rel_cmtm = state_dict_to_rel_cmtm(state, link.name, robot.links[joint.child_link_id].name, "link", order)
            result[..., out_start:out_start+n_, :] += _matmul_rhs(rel_cmtm.mat_adj(), rhs[..., j*n_:(j+1)*n_, :])
    return result


def _batch_total_joint_tan_vel_to_link_vel_grad_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    vec = np.asarray(vec)
    result = np.zeros(vec.shape[:-1] + (robot.link_num * n_,), dtype=vec.dtype)

    for i, link in enumerate(robot.links):
        link_route = []
        joint_route = []
        robot.route_target_link(link, link_route, joint_route)
        tangent_mat_inv = state_dict_to_cmtm(state, link.name, "link", order).tangent_mat_inv()
        out_start = i * n_
        for j in joint_route:
            joint = robot.joints[j]
            rel_cmtm = state_dict_to_rel_cmtm(state, link.name, robot.links[joint.child_link_id].name, "link", order)
            block = tangent_mat_inv @ rel_cmtm.mat_adj()
            result[..., out_start:out_start+n_] += _batched_matvec(block, vec[..., j*n_:(j+1)*n_])
    return result


def _batch_total_joint_tan_vel_to_link_vel_grad_matmul_rhs(
    robot: RobotStruct,
    state: dict,
    rhs: np.ndarray,
    order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    rhs = np.asarray(rhs)
    result = np.zeros(rhs.shape[:-2] + (robot.link_num * n_, rhs.shape[-1]), dtype=rhs.dtype)

    for i, link in enumerate(robot.links):
        link_route = []
        joint_route = []
        robot.route_target_link(link, link_route, joint_route)
        tangent_mat_inv = state_dict_to_cmtm(state, link.name, "link", order).tangent_mat_inv()
        out_start = i * n_
        for j in joint_route:
            joint = robot.joints[j]
            rel_cmtm = state_dict_to_rel_cmtm(state, link.name, robot.links[joint.child_link_id].name, "link", order)
            block = tangent_mat_inv @ rel_cmtm.mat_adj()
            result[..., out_start:out_start+n_, :] += _matmul_rhs(block, rhs[..., j*n_:(j+1)*n_, :])
    return result


def _batch_total_coord_to_link_tan_vel_grad_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    out_order: int = 3,
    in_order: int | None = None,
    dim: int = 3,
) -> np.ndarray:
    coord_vec = vec if in_order is None else _batch_total_coord_arrange_vec(robot, vec, out_order=out_order, in_order=in_order)
    joint_tan_vec = _batch_total_coord_to_joint_tan_vel_grad_matvec(robot, state, coord_vec, out_order, dim)
    return _batch_total_joint_tan_vel_to_link_tan_vel_grad_matvec(robot, state, joint_tan_vec, out_order, dim)


def _batch_total_coord_to_link_tan_vel_grad_matmul_rhs(
    robot: RobotStruct,
    state: dict,
    rhs: np.ndarray,
    out_order: int = 3,
    in_order: int | None = None,
    dim: int = 3,
) -> np.ndarray:
    coord_rhs = rhs if in_order is None else _batch_total_coord_arrange_rhs(robot, rhs, out_order=out_order, in_order=in_order)
    joint_tan_rhs = _batch_total_coord_to_joint_tan_vel_grad_matmul_rhs(robot, state, coord_rhs, out_order, dim)
    return _batch_total_joint_tan_vel_to_link_tan_vel_grad_matmul_rhs(robot, state, joint_tan_rhs, out_order, dim)


def _batch_total_coord_to_link_vel_grad_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    joint_tan_vec = _batch_total_coord_to_joint_tan_vel_grad_matvec(robot, state, vec, order, dim)
    return _batch_total_joint_tan_vel_to_link_vel_grad_matvec(robot, state, joint_tan_vec, order, dim)


def _batch_total_coord_to_link_vel_grad_matmul_rhs(
    robot: RobotStruct,
    state: dict,
    rhs: np.ndarray,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    joint_tan_rhs = _batch_total_coord_to_joint_tan_vel_grad_matmul_rhs(robot, state, rhs, order, dim)
    return _batch_total_joint_tan_vel_to_link_vel_grad_matmul_rhs(robot, state, joint_tan_rhs, order, dim)


def _batch_selected_coord_to_link_tan_vel_grad_mat(
    robot: RobotStruct,
    state: dict,
    links: list,
    out_order: int = 3,
    in_order: int | None = None,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * out_order
    joint_tan_mat = _batch_total_coord_to_joint_tan_vel_grad_mat(robot, state, out_order, dim)
    if in_order is not None:
        joint_tan_mat = joint_tan_mat @ total_coord_arrange(robot, out_order=out_order, in_order=in_order)

    mat = np.zeros(joint_tan_mat.shape[:-2] + (len(links) * n_, joint_tan_mat.shape[-1]), dtype=joint_tan_mat.dtype)

    for i, link in enumerate(links):
        if link is None:
            raise ValueError("link_name_list contains invalid link name")

        link_route = []
        joint_route = []
        robot.route_target_link(link, link_route, joint_route)

        row = i * n_
        for j in joint_route:
            joint = robot.joints[j]
            rel_cmtm = state_dict_to_rel_cmtm(state, link.name, robot.links[joint.child_link_id].name, "link", out_order)
            col = j * n_
            mat[..., row:row+n_, :] += rel_cmtm.mat_adj() @ joint_tan_mat[..., col:col+n_, :]

    return mat


def _batch_selected_coord_to_link_vel_grad_mat(
    robot: RobotStruct,
    state: dict,
    links: list,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    joint_tan_mat = _batch_total_coord_to_joint_tan_vel_grad_mat(robot, state, order, dim)
    mat = np.zeros(joint_tan_mat.shape[:-2] + (len(links) * n_, robot.dof * order), dtype=joint_tan_mat.dtype)

    for i, link in enumerate(links):
        if link is None:
            raise ValueError("link_name_list contains invalid link name")

        link_route = []
        joint_route = []
        robot.route_target_link(link, link_route, joint_route)
        link_tan_inv = state_dict_to_cmtm(state, link.name, "link", order).tangent_mat_inv()

        row = i * n_
        for j in joint_route:
            joint = robot.joints[j]
            rel_cmtm = state_dict_to_rel_cmtm(state, link.name, robot.links[joint.child_link_id].name, "link", order)
            col = j * n_
            block = link_tan_inv @ rel_cmtm.mat_adj()
            mat[..., row:row+n_, :] += block @ joint_tan_mat[..., col:col+n_, :]

    return mat


def _batch_momentum_to_force_grad_mat(link_cmtm, force_order: int = 1, dim: int = 3) -> np.ndarray:
    dof = dim_to_dof(dim)
    momentum_dof = dof * (force_order + 1)
    force_dof = dof * force_order
    cmvecs = link_cmtm.cmvecs().cm_vecs()[..., :force_order + 1, :]
    batch_shape = cmvecs.shape[:-2]

    mat = np.zeros(batch_shape + (force_dof, momentum_dof), dtype=cmvecs.dtype)
    mat[..., :, dof:] = np.diag(np.repeat(np.arange(1, force_order + 1), dof))
    mat[..., :, :-dof] += CMTM.hat_adj(SE3wrench, cmvecs)
    return Factorial.mat(force_order, dof) @ mat @ Factorial.mat_inv(force_order + 1, dof)


def _batch_partial_link_sp_vel_to_force_grad_mat(momentum: CMVector, force_order: int = 1, dim: int = 3) -> np.ndarray:
    dof = dim_to_dof(dim)
    momentum_dof = dof * (force_order + 2)
    force_dof = dof * force_order
    momentum_vecs = momentum.cm_vecs()[..., :force_order + 1, :]
    batch_shape = momentum_vecs.shape[:-2]

    mat = np.zeros(batch_shape + (force_dof, momentum_dof), dtype=momentum_vecs.dtype)
    m = np.zeros(batch_shape + (force_dof, momentum_dof - dof), dtype=momentum_vecs.dtype)
    m[..., :, :-dof] = CMTM.hat_commute_adj(SE3wrench, momentum_vecs)
    mat[..., :, dof:] = Factorial.mat(force_order, dof) @ m @ Factorial.mat_inv(force_order + 1, dof)
    return mat


def _batch_total_partial_momentum_to_force_grad_mat(
    robot: RobotStruct,
    state: dict,
    force_order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * force_order
    m_ = dim_to_dof(dim) * (force_order + 1)
    batch_shape = _state_batch_shape(state, robot.links[0].name, "link", force_order + 1)
    mat = np.zeros(batch_shape + (robot.link_num * n_, robot.link_num * m_))

    for i, link in enumerate(robot.links):
        cmtm = state_dict_to_cmtm(state, link.name, "link", force_order + 1)
        mat[..., i*n_:(i+1)*n_, i*m_:(i+1)*m_] = _batch_momentum_to_force_grad_mat(cmtm, force_order, dim)
    return mat


def _batch_total_partial_link_sp_vel_to_link_force_grad_mat(
    robot: RobotStruct,
    state: dict,
    force_order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * force_order
    m_ = dim_to_dof(dim) * (force_order + 2)
    batch_shape = _state_batch_shape(state, robot.links[0].name, "link", force_order + 2)
    mat = np.zeros(batch_shape + (robot.link_num * n_, robot.link_num * m_))

    for i, link in enumerate(robot.links):
        link_momentum = state_dict_to_cmvec(state, link.name, "link", "momentum", force_order)
        mat[..., i*n_:(i+1)*n_, i*m_:(i+1)*m_] = _batch_partial_link_sp_vel_to_force_grad_mat(link_momentum, force_order, dim)
    return mat


def _batch_total_partial_link_sp_vel_to_joint_force_grad_mat(
    robot: RobotStruct,
    state: dict,
    force_order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * force_order
    m_ = dim_to_dof(dim) * (force_order + 2)
    batch_shape = _state_batch_shape(state, robot.joints[0].name, "joint", force_order + 2)
    mat = np.zeros(batch_shape + (robot.joint_num * n_, robot.joint_num * m_))

    for i, joint in enumerate(robot.joints):
        joint_momentum = state_dict_to_cmvec(state, joint.name, "joint", "momentum", force_order)
        mat[..., i*n_:(i+1)*n_, i*m_:(i+1)*m_] = _batch_partial_link_sp_vel_to_force_grad_mat(joint_momentum, force_order, dim)
    return mat


def _batch_total_link_inertia_matvec(
    robot: RobotStruct,
    vec: np.ndarray,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    vec = np.asarray(vec)
    result = np.zeros(vec.shape[:-1] + (robot.link_num * n_,), dtype=vec.dtype)

    for i, link in enumerate(robot.links):
        start = i * n_
        inertia = inertia_diag_mat(spatial_inertia(link.mass, link.inertia, link.cog), order)
        result[..., start:start+n_] = _batched_matvec(inertia, vec[..., start:start+n_])
    return result


def _batch_total_link_inertia_matmul_rhs(
    robot: RobotStruct,
    rhs: np.ndarray,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    rhs = np.asarray(rhs)
    result = np.zeros(rhs.shape[:-2] + (robot.link_num * n_, rhs.shape[-1]), dtype=rhs.dtype)

    for i, link in enumerate(robot.links):
        start = i * n_
        inertia = inertia_diag_mat(spatial_inertia(link.mass, link.inertia, link.cog), order)
        result[..., start:start+n_, :] = _matmul_rhs(inertia, rhs[..., start:start+n_, :])
    return result


def _batch_total_link_sp_vel_from_link_vel(vec_link_vel: np.ndarray, order: int, dim: int = 3) -> np.ndarray:
    dof = dim_to_dof(dim)
    vec_link_vel = np.asarray(vec_link_vel)
    block = dof * order
    sp_block = dof * (order - 1)
    link_num = vec_link_vel.shape[-1] // block
    result = np.zeros(vec_link_vel.shape[:-1] + (link_num * sp_block,), dtype=vec_link_vel.dtype)
    for i in range(link_num):
        result[..., i*sp_block:(i+1)*sp_block] = vec_link_vel[..., i*block + dof:(i+1)*block]
    return result


def _batch_total_link_sp_vel_from_link_vel_rhs(rhs_link_vel: np.ndarray, order: int, dim: int = 3) -> np.ndarray:
    dof = dim_to_dof(dim)
    rhs_link_vel = np.asarray(rhs_link_vel)
    block = dof * order
    sp_block = dof * (order - 1)
    link_num = rhs_link_vel.shape[-2] // block
    result = np.zeros(rhs_link_vel.shape[:-2] + (link_num * sp_block, rhs_link_vel.shape[-1]), dtype=rhs_link_vel.dtype)
    for i in range(link_num):
        result[..., i*sp_block:(i+1)*sp_block, :] = rhs_link_vel[..., i*block + dof:(i+1)*block, :]
    return result


def _batch_total_coord_to_link_momentum_grad_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    order: int = 3,
    dim: int = 3,
    vec_link_vel: np.ndarray | None = None,
) -> np.ndarray:
    if vec_link_vel is None:
        vec_link_vel = _batch_total_coord_to_link_vel_grad_matvec(robot, state, vec, order, dim)
    vec_link_sp = _batch_total_link_sp_vel_from_link_vel(vec_link_vel, order, dim)
    return _batch_total_link_inertia_matvec(robot, vec_link_sp, order=order-1, dim=dim)


def _batch_total_coord_to_link_momentum_grad_matmul_rhs(
    robot: RobotStruct,
    state: dict,
    rhs: np.ndarray,
    order: int = 3,
    dim: int = 3,
    rhs_link_vel: np.ndarray | None = None,
) -> np.ndarray:
    if rhs_link_vel is None:
        rhs_link_vel = _batch_total_coord_to_link_vel_grad_matmul_rhs(robot, state, rhs, order, dim)
    rhs_link_sp = _batch_total_link_sp_vel_from_link_vel_rhs(rhs_link_vel, order, dim)
    return _batch_total_link_inertia_matmul_rhs(robot, rhs_link_sp, order=order-1, dim=dim)


def _batch_total_factorial_matvec(num: int, vec: np.ndarray, order: int, submat_dim: int = 6) -> np.ndarray:
    n_ = submat_dim * order
    vec = np.asarray(vec)
    result = np.zeros(vec.shape[:-1] + (num * n_,), dtype=vec.dtype)
    mat = Factorial.mat(order, submat_dim)
    for i in range(num):
        start = i * n_
        result[..., start:start+n_] = _batched_matvec(mat, vec[..., start:start+n_])
    return result


def _batch_total_factorial_matmul_rhs(num: int, rhs: np.ndarray, order: int, submat_dim: int = 6) -> np.ndarray:
    n_ = submat_dim * order
    rhs = np.asarray(rhs)
    result = np.zeros(rhs.shape[:-2] + (num * n_, rhs.shape[-1]), dtype=rhs.dtype)
    mat = Factorial.mat(order, submat_dim)
    for i in range(num):
        start = i * n_
        result[..., start:start+n_, :] = _matmul_rhs(mat, rhs[..., start:start+n_, :])
    return result


def _batch_total_factorial_mat_inv_vec(num: int, vec: np.ndarray, order: int, submat_dim: int = 6) -> np.ndarray:
    n_ = submat_dim * order
    vec = np.asarray(vec)
    result = np.zeros(vec.shape[:-1] + (num * n_,), dtype=vec.dtype)
    mat = Factorial.mat_inv(order, submat_dim)
    for i in range(num):
        start = i * n_
        result[..., start:start+n_] = _batched_matvec(mat, vec[..., start:start+n_])
    return result


def _batch_total_factorial_mat_inv_rhs(num: int, rhs: np.ndarray, order: int, submat_dim: int = 6) -> np.ndarray:
    n_ = submat_dim * order
    rhs = np.asarray(rhs)
    result = np.zeros(rhs.shape[:-2] + (num * n_, rhs.shape[-1]), dtype=rhs.dtype)
    mat = Factorial.mat_inv(order, submat_dim)
    for i in range(num):
        start = i * n_
        result[..., start:start+n_, :] = _matmul_rhs(mat, rhs[..., start:start+n_, :])
    return result


def _batch_total_world_link_cmtm_wrench_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    vec = np.asarray(vec)
    result = np.zeros(vec.shape[:-1] + (robot.link_num * n_,), dtype=vec.dtype)

    for i, link in enumerate(robot.links):
        start = i * n_
        cmtm_wrench = state_dict_to_cmtm_wrench(state, link.name, "link", order)
        result[..., start:start+n_] = apply_mat_adj(cmtm_wrench, vec[..., start:start+n_])
    return result


def _batch_total_world_link_cmtm_wrench_matmul_rhs(
    robot: RobotStruct,
    state: dict,
    rhs: np.ndarray,
    order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    rhs = np.asarray(rhs)
    result = np.zeros(rhs.shape[:-2] + (robot.link_num * n_, rhs.shape[-1]), dtype=rhs.dtype)

    for i, link in enumerate(robot.links):
        start = i * n_
        cmtm_wrench = state_dict_to_cmtm_wrench(state, link.name, "link", order)
        result[..., start:start+n_, :] = _matmul_rhs(cmtm_wrench.mat_adj(), rhs[..., start:start+n_, :])
    return result


def _batch_total_world_joint_cmtm_wrench_inv_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    vec = np.asarray(vec)
    result = np.zeros(vec.shape[:-1] + (robot.joint_num * n_,), dtype=vec.dtype)

    for i, joint in enumerate(robot.joints):
        start = i * n_
        cmtm_wrench = state_dict_to_cmtm_wrench(state, robot.links[joint.child_link_id].name, "link", order)
        result[..., start:start+n_] = apply_mat_inv_adj(cmtm_wrench, vec[..., start:start+n_])
    return result


def _batch_total_world_joint_cmtm_wrench_inv_matmul_rhs(
    robot: RobotStruct,
    state: dict,
    rhs: np.ndarray,
    order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    rhs = np.asarray(rhs)
    result = np.zeros(rhs.shape[:-2] + (robot.joint_num * n_, rhs.shape[-1]), dtype=rhs.dtype)

    for i, joint in enumerate(robot.joints):
        start = i * n_
        cmtm_wrench = state_dict_to_cmtm_wrench(state, robot.links[joint.child_link_id].name, "link", order)
        result[..., start:start+n_, :] = _matmul_rhs(cmtm_wrench.mat_inv_adj(), rhs[..., start:start+n_, :])
    return result


def _batch_total_world_link_wrench_to_world_joint_wrench_matvec(
    robot: RobotStruct,
    vec: np.ndarray,
    order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    vec = np.asarray(vec)
    result = np.zeros(vec.shape[:-1] + (robot.joint_num * n_,), dtype=vec.dtype)

    for i, joint in enumerate(robot.joints):
        link_route = []
        joint_route = []
        robot.route_end_joints(joint, link_route, joint_route)
        out_start = i * n_
        for j in link_route:
            result[..., out_start:out_start+n_] += vec[..., j*n_:(j+1)*n_]
    return result


def _batch_total_world_link_wrench_to_world_joint_wrench_matmul_rhs(
    robot: RobotStruct,
    rhs: np.ndarray,
    order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    rhs = np.asarray(rhs)
    result = np.zeros(rhs.shape[:-2] + (robot.joint_num * n_, rhs.shape[-1]), dtype=rhs.dtype)

    for i, joint in enumerate(robot.joints):
        link_route = []
        joint_route = []
        robot.route_end_joints(joint, link_route, joint_route)
        out_start = i * n_
        for j in link_route:
            result[..., out_start:out_start+n_, :] += rhs[..., j*n_:(j+1)*n_, :]
    return result


def _batch_total_partial_link_momentum_to_world_link_momentum_grad_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    dof = dim_to_dof(dim)
    inv_fact_vec = _batch_total_factorial_mat_inv_vec(robot.link_num, vec, order-1, dof)
    cmtm_vec = _batch_total_world_link_cmtm_wrench_matvec(robot, state, inv_fact_vec, order-1, dim)
    return _batch_total_factorial_matvec(robot.link_num, cmtm_vec, order-1, dof)


def _batch_total_partial_link_momentum_to_world_link_momentum_grad_matmul_rhs(
    robot: RobotStruct,
    state: dict,
    rhs: np.ndarray,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    dof = dim_to_dof(dim)
    inv_fact_rhs = _batch_total_factorial_mat_inv_rhs(robot.link_num, rhs, order-1, dof)
    cmtm_rhs = _batch_total_world_link_cmtm_wrench_matmul_rhs(robot, state, inv_fact_rhs, order-1, dim)
    return _batch_total_factorial_matmul_rhs(robot.link_num, cmtm_rhs, order-1, dof)


def _batch_total_link_cmtm_wrench_var_x_arb_vec_matvec(
    robot: RobotStruct,
    state: dict,
    total_cm_vec: np.ndarray,
    vec: np.ndarray,
    order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    vec = np.asarray(vec)
    total_cm_vecs = np.asarray(total_cm_vec).reshape(total_cm_vec.shape[:-1] + (robot.link_num, n_))
    result = np.zeros(vec.shape[:-1] + (robot.link_num * n_,), dtype=vec.dtype)
    for i, link in enumerate(robot.links):
        start = i * n_
        arb_v = CMVector.set_cmvecs(total_cm_vecs[..., i, :].reshape(total_cm_vecs.shape[:-2] + (order, -1)))
        cmtm_wrench = state_dict_to_cmtm_wrench(state, link.name, "link", order)
        result[..., start:start+n_] = _cmtm_var_jacob_matvec(
            cmtm_wrench,
            arb_v,
            vec[..., start:start+n_],
        )
    return result


def _batch_total_link_cmtm_wrench_var_x_arb_vec_matmul_rhs(
    robot: RobotStruct,
    state: dict,
    total_cm_vec: np.ndarray,
    rhs: np.ndarray,
    order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    rhs = np.asarray(rhs)
    total_cm_vecs = np.asarray(total_cm_vec).reshape(total_cm_vec.shape[:-1] + (robot.link_num, n_))
    result = np.zeros(rhs.shape[:-2] + (robot.link_num * n_, rhs.shape[-1]), dtype=rhs.dtype)
    for i, link in enumerate(robot.links):
        start = i * n_
        arb_v = CMVector.set_cmvecs(total_cm_vecs[..., i, :].reshape(total_cm_vecs.shape[:-2] + (order, -1)))
        cmtm_wrench = state_dict_to_cmtm_wrench(state, link.name, "link", order)
        result[..., start:start+n_, :] = _cmtm_var_jacob_matmul_rhs(
            cmtm_wrench,
            arb_v,
            rhs[..., start:start+n_, :],
        )
    return result


def _batch_total_joint_cmtm_wrench_inv_var_x_arb_vec_matvec(
    robot: RobotStruct,
    state: dict,
    total_cm_vec: np.ndarray,
    vec: np.ndarray,
    order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    vec = np.asarray(vec)
    total_cm_vecs = np.asarray(total_cm_vec).reshape(total_cm_vec.shape[:-1] + (robot.joint_num, n_))
    result = np.zeros(vec.shape[:-1] + (robot.joint_num * n_,), dtype=vec.dtype)
    for i, joint in enumerate(robot.joints):
        start = i * n_
        arb_v = CMVector.set_cmvecs(total_cm_vecs[..., i, :].reshape(total_cm_vecs.shape[:-2] + (order, -1)))
        child_link = robot.links[joint.child_link_id]
        cmtm_wrench = state_dict_to_cmtm_wrench(state, child_link.name, "link", order)
        result[..., start:start+n_] = _cmtm_var_jacob_matvec(
            cmtm_wrench,
            arb_v,
            vec[..., start:start+n_],
            inverse=True,
        )
    return result


def _batch_total_joint_cmtm_wrench_inv_var_x_arb_vec_matmul_rhs(
    robot: RobotStruct,
    state: dict,
    total_cm_vec: np.ndarray,
    rhs: np.ndarray,
    order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    rhs = np.asarray(rhs)
    total_cm_vecs = np.asarray(total_cm_vec).reshape(total_cm_vec.shape[:-1] + (robot.joint_num, n_))
    result = np.zeros(rhs.shape[:-2] + (robot.joint_num * n_, rhs.shape[-1]), dtype=rhs.dtype)
    for i, joint in enumerate(robot.joints):
        start = i * n_
        arb_v = CMVector.set_cmvecs(total_cm_vecs[..., i, :].reshape(total_cm_vecs.shape[:-2] + (order, -1)))
        child_link = robot.links[joint.child_link_id]
        cmtm_wrench = state_dict_to_cmtm_wrench(state, child_link.name, "link", order)
        result[..., start:start+n_, :] = _cmtm_var_jacob_matmul_rhs(
            cmtm_wrench,
            arb_v,
            rhs[..., start:start+n_, :],
            inverse=True,
        )
    return result


def _batch_extract_total_link_cmvec(state: dict, link_names: list[str], data_type: str, order: int) -> np.ndarray:
    vecs = [
        state_dict_to_cmvec(state, link_name, "link", data_type, order).cm_vec()
        for link_name in link_names
    ]
    return np.concatenate(vecs, axis=-1)


def _batch_total_partial_link_tan_vel_to_world_link_momentum_grad_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    dof = dim_to_dof(dim)
    total_local_link_momentum = _batch_extract_total_link_cmvec(state, robot.link_names, "momentum", order-1)
    cmtm_vec = _batch_total_link_cmtm_wrench_var_x_arb_vec_matvec(robot, state, total_local_link_momentum, vec, order-1, dim)
    return _batch_total_factorial_matvec(robot.link_num, cmtm_vec, order-1, dof)


def _batch_total_partial_link_tan_vel_to_world_link_momentum_grad_matmul_rhs(
    robot: RobotStruct,
    state: dict,
    rhs: np.ndarray,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    dof = dim_to_dof(dim)
    total_local_link_momentum = _batch_extract_total_link_cmvec(state, robot.link_names, "momentum", order-1)
    cmtm_rhs = _batch_total_link_cmtm_wrench_var_x_arb_vec_matmul_rhs(robot, state, total_local_link_momentum, rhs, order-1, dim)
    return _batch_total_factorial_matmul_rhs(robot.link_num, cmtm_rhs, order-1, dof)


def _batch_total_partial_world_joint_momentum_to_joint_momentum_grad_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    dof = dim_to_dof(dim)
    inv_fact_vec = _batch_total_factorial_mat_inv_vec(robot.joint_num, vec, order-1, dof)
    cmtm_vec = _batch_total_world_joint_cmtm_wrench_inv_matvec(robot, state, inv_fact_vec, order-1, dim)
    return _batch_total_factorial_matvec(robot.joint_num, cmtm_vec, order-1, dof)


def _batch_total_partial_world_joint_momentum_to_joint_momentum_grad_matmul_rhs(
    robot: RobotStruct,
    state: dict,
    rhs: np.ndarray,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    dof = dim_to_dof(dim)
    inv_fact_rhs = _batch_total_factorial_mat_inv_rhs(robot.joint_num, rhs, order-1, dof)
    cmtm_rhs = _batch_total_world_joint_cmtm_wrench_inv_matmul_rhs(robot, state, inv_fact_rhs, order-1, dim)
    return _batch_total_factorial_matmul_rhs(robot.joint_num, cmtm_rhs, order-1, dof)


def _batch_total_partial_link_tan_vel_to_joint_momentum_grad_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    dof = dim_to_dof(dim)
    total_local_link_momentum = _batch_extract_total_link_cmvec(state, robot.link_names, "momentum", order-1)
    world_link_momentum = _batch_total_world_link_cmtm_wrench_matvec(robot, state, total_local_link_momentum, order-1, dim)
    total_world_joint_momentum = _batch_total_world_link_wrench_to_world_joint_wrench_matvec(robot, world_link_momentum, order-1, dim)
    cmtm_vec = _batch_total_joint_cmtm_wrench_inv_var_x_arb_vec_matvec(robot, state, total_world_joint_momentum, vec, order-1, dim)
    return _batch_total_factorial_matvec(robot.joint_num, cmtm_vec, order-1, dof)


def _batch_total_partial_link_tan_vel_to_joint_momentum_grad_matmul_rhs(
    robot: RobotStruct,
    state: dict,
    rhs: np.ndarray,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    dof = dim_to_dof(dim)
    total_local_link_momentum = _batch_extract_total_link_cmvec(state, robot.link_names, "momentum", order-1)
    world_link_momentum = _batch_total_world_link_cmtm_wrench_matvec(robot, state, total_local_link_momentum, order-1, dim)
    total_world_joint_momentum = _batch_total_world_link_wrench_to_world_joint_wrench_matvec(robot, world_link_momentum, order-1, dim)
    cmtm_rhs = _batch_total_joint_cmtm_wrench_inv_var_x_arb_vec_matmul_rhs(robot, state, total_world_joint_momentum, rhs, order-1, dim)
    return _batch_total_factorial_matmul_rhs(robot.joint_num, cmtm_rhs, order-1, dof)


def _batch_total_partial_momentum_to_force_grad_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    force_order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * force_order
    m_ = dim_to_dof(dim) * (force_order + 1)
    vec = np.asarray(vec)
    result = np.zeros(vec.shape[:-1] + (robot.link_num * n_,), dtype=vec.dtype)

    for i, link in enumerate(robot.links):
        out_start = i * n_
        in_start = i * m_
        cmtm = state_dict_to_cmtm(state, link.name, "link", force_order + 1)
        mat = _batch_momentum_to_force_grad_mat(cmtm, force_order=force_order, dim=dim)
        result[..., out_start:out_start+n_] = _batched_matvec(mat, vec[..., in_start:in_start+m_])
    return result


def _batch_total_partial_momentum_to_force_grad_matmul_rhs(
    robot: RobotStruct,
    state: dict,
    rhs: np.ndarray,
    force_order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * force_order
    m_ = dim_to_dof(dim) * (force_order + 1)
    rhs = np.asarray(rhs)
    result = np.zeros(rhs.shape[:-2] + (robot.link_num * n_, rhs.shape[-1]), dtype=rhs.dtype)

    for i, link in enumerate(robot.links):
        out_start = i * n_
        in_start = i * m_
        cmtm = state_dict_to_cmtm(state, link.name, "link", force_order + 1)
        mat = _batch_momentum_to_force_grad_mat(cmtm, force_order=force_order, dim=dim)
        result[..., out_start:out_start+n_, :] = _matmul_rhs(mat, rhs[..., in_start:in_start+m_, :])
    return result


def _batch_total_partial_link_sp_vel_to_link_force_grad_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    force_order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * force_order
    m_ = dim_to_dof(dim) * (force_order + 2)
    vec = np.asarray(vec)
    result = np.zeros(vec.shape[:-1] + (robot.link_num * n_,), dtype=vec.dtype)

    for i, link in enumerate(robot.links):
        out_start = i * n_
        in_start = i * m_
        link_momentum = state_dict_to_cmvec(state, link.name, "link", "momentum", force_order)
        mat = _batch_partial_link_sp_vel_to_force_grad_mat(link_momentum, force_order=force_order, dim=dim)
        result[..., out_start:out_start+n_] = _batched_matvec(mat, vec[..., in_start:in_start+m_])
    return result


def _batch_total_partial_link_sp_vel_to_link_force_grad_matmul_rhs(
    robot: RobotStruct,
    state: dict,
    rhs: np.ndarray,
    force_order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * force_order
    m_ = dim_to_dof(dim) * (force_order + 2)
    rhs = np.asarray(rhs)
    result = np.zeros(rhs.shape[:-2] + (robot.link_num * n_, rhs.shape[-1]), dtype=rhs.dtype)

    for i, link in enumerate(robot.links):
        out_start = i * n_
        in_start = i * m_
        link_momentum = state_dict_to_cmvec(state, link.name, "link", "momentum", force_order)
        mat = _batch_partial_link_sp_vel_to_force_grad_mat(link_momentum, force_order=force_order, dim=dim)
        result[..., out_start:out_start+n_, :] = _matmul_rhs(mat, rhs[..., in_start:in_start+m_, :])
    return result


def _batch_total_partial_link_sp_vel_to_joint_force_grad_matvec(
    robot: RobotStruct,
    state: dict,
    vec: np.ndarray,
    force_order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * force_order
    m_ = dim_to_dof(dim) * (force_order + 2)
    vec = np.asarray(vec)
    result = np.zeros(vec.shape[:-1] + (robot.joint_num * n_,), dtype=vec.dtype)

    for i, joint in enumerate(robot.joints):
        out_start = i * n_
        in_start = i * m_
        joint_momentum = state_dict_to_cmvec(state, joint.name, "joint", "momentum", force_order)
        mat = _batch_partial_link_sp_vel_to_force_grad_mat(joint_momentum, force_order=force_order, dim=dim)
        result[..., out_start:out_start+n_] = _batched_matvec(mat, vec[..., in_start:in_start+m_])
    return result


def _batch_total_partial_link_sp_vel_to_joint_force_grad_matmul_rhs(
    robot: RobotStruct,
    state: dict,
    rhs: np.ndarray,
    force_order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * force_order
    m_ = dim_to_dof(dim) * (force_order + 2)
    rhs = np.asarray(rhs)
    result = np.zeros(rhs.shape[:-2] + (robot.joint_num * n_, rhs.shape[-1]), dtype=rhs.dtype)

    for i, joint in enumerate(robot.joints):
        out_start = i * n_
        in_start = i * m_
        joint_momentum = state_dict_to_cmvec(state, joint.name, "joint", "momentum", force_order)
        mat = _batch_partial_link_sp_vel_to_force_grad_mat(joint_momentum, force_order=force_order, dim=dim)
        result[..., out_start:out_start+n_, :] = _matmul_rhs(mat, rhs[..., in_start:in_start+m_, :])
    return result


def _batch_total_joint_wrench_to_joint_torque_matvec(
    robot: RobotStruct,
    vec: np.ndarray,
    torque_order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * torque_order
    vec = np.asarray(vec)
    result = np.zeros(vec.shape[:-1] + (robot.joint_dof * torque_order,), dtype=vec.dtype)

    for i, joint in enumerate(robot.joints):
        if joint.dof == 0:
            continue
        out_start = joint.dof_index * torque_order
        select = joint_select_diag_mat(joint.select_mat, torque_order).T
        result[..., out_start:out_start + joint.dof*torque_order] = _batched_matvec(select, vec[..., i*n_:(i+1)*n_])
    return result


def _batch_total_joint_wrench_to_joint_torque_matmul_rhs(
    robot: RobotStruct,
    rhs: np.ndarray,
    torque_order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    n_ = dim_to_dof(dim) * torque_order
    rhs = np.asarray(rhs)
    result = np.zeros(rhs.shape[:-2] + (robot.joint_dof * torque_order, rhs.shape[-1]), dtype=rhs.dtype)

    for i, joint in enumerate(robot.joints):
        if joint.dof == 0:
            continue
        out_start = joint.dof_index * torque_order
        select = joint_select_diag_mat(joint.select_mat, torque_order).T
        result[..., out_start:out_start + joint.dof*torque_order, :] = _matmul_rhs(select, rhs[..., i*n_:(i+1)*n_, :])
    return result


def _selected_coord_to_link_tan_vel_grad_mat(
    robot: RobotStruct,
    state: dict,
    links: list,
    out_order: int = 3,
    in_order: int | None = None,
    dim: int = 3,
) -> np.ndarray:
    if links and _is_batched_kinematics_state(robot, state, [StateType("link", links[0].name, "vel")], out_order):
        return _batch_selected_coord_to_link_tan_vel_grad_mat(robot, state, links, out_order, in_order, dim)

    n_ = dim_to_dof(dim) * out_order
    joint_tan_mat = total_coord_to_joint_tan_vel_grad_mat(robot, state, out_order, dim)
    if in_order is not None:
        joint_tan_mat = joint_tan_mat @ total_coord_arrange(robot, out_order=out_order, in_order=in_order)

    mat = np.zeros((len(links) * n_, joint_tan_mat.shape[1]))

    for i, link in enumerate(links):
        if link is None:
            raise ValueError("link_name_list contains invalid link name")

        link_route = []
        joint_route = []
        robot.route_target_link(link, link_route, joint_route)

        row = i * n_
        for j in joint_route:
            joint = robot.joints[j]
            rel_cmtm = state_dict_to_rel_cmtm(state, link.name, robot.links[joint.child_link_id].name, "link", out_order)
            col = j * n_
            mat[row:row+n_, :] += rel_cmtm.mat_adj() @ joint_tan_mat[col:col+n_, :]

    return mat


def _selected_coord_to_link_momentum_grad_mat(
    robot: RobotStruct,
    state: dict,
    links: list,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    if links and _is_batched_kinematics_state(robot, state, [StateType("link", links[0].name, "vel")], order):
        return _batch_selected_coord_to_link_momentum_grad_mat(robot, state, links, order, dim)

    dof = dim_to_dof(dim)
    n_j = dof * order
    n_m = dof * (order - 1)
    joint_tan_mat = total_coord_to_joint_tan_vel_grad_mat(robot, state, order, dim)
    mat = np.zeros((len(links) * n_m, joint_tan_mat.shape[1]))

    for i, link in enumerate(links):
        if link is None:
            raise ValueError("link_name_list contains invalid link name")

        link_route = []
        joint_route = []
        robot.route_target_link(link, link_route, joint_route)
        link_sp_tan_inv = state_dict_to_cmtm(state, link.name, "link", order).tangent_mat_inv()[dof:]

        row = i * n_m
        row_block = np.zeros((n_m, joint_tan_mat.shape[1]))
        for j in joint_route:
            joint = robot.joints[j]
            rel_cmtm = state_dict_to_rel_cmtm(state, link.name, robot.links[joint.child_link_id].name, "link", order)
            col = j * n_j
            row_block += (link_sp_tan_inv @ rel_cmtm.mat_adj()) @ joint_tan_mat[col:col+n_j, :]

        inertia_block = inertia_diag_mat(spatial_inertia(link.mass, link.inertia, link.cog), order - 1)
        mat[row:row+n_m, :] = inertia_block @ row_block

    return mat


def _selected_coord_to_world_link_momentum_grad_mat(
    robot: RobotStruct,
    state: dict,
    links: list,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    if links and _is_batched_kinematics_state(robot, state, [StateType("link", links[0].name, "vel")], order):
        return _batch_selected_coord_to_world_link_momentum_grad_mat(robot, state, links, order, dim)

    dof = dim_to_dof(dim)
    n_m = dof * (order - 1)
    mat_link_mom = _selected_coord_to_link_momentum_grad_mat(robot, state, links, order=order, dim=dim)
    mat_tan_kine = _selected_coord_to_link_tan_vel_grad_mat(robot, state, links, out_order=order-1, in_order=order, dim=dim)
    mat = np.zeros_like(mat_link_mom)

    factorial = Factorial.mat(order - 1, dof)
    factorial_inv = Factorial.mat_inv(order - 1, dof)
    for i, link in enumerate(links):
        if link is None:
            raise ValueError("link_name_list contains invalid link name")

        row = i * n_m
        cmtm_wrench = state_dict_to_cmtm_wrench(state, link.name, "link", order - 1)
        block_mom = factorial @ cmtm_wrench.mat_adj() @ factorial_inv
        link_momentum = state_dict_to_cmvec(state, link.name, "link", "momentum", order - 1)
        block_tan = factorial @ cmtm_wrench.mat_var_x_arb_vec_jacob(link_momentum, frame="bframe")
        mat[row:row+n_m, :] = block_mom @ mat_link_mom[row:row+n_m, :] + block_tan @ mat_tan_kine[row:row+n_m, :]

    return mat


def _selected_coord_to_link_force_grad_mat(
    robot: RobotStruct,
    state: dict,
    links: list,
    force_order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    if links and _is_batched_kinematics_state(robot, state, [StateType("link", links[0].name, "vel")], force_order + 2):
        return _batch_selected_coord_to_link_force_grad_mat(robot, state, links, force_order, dim)

    dof = dim_to_dof(dim)
    n_f = dof * force_order
    full = total_coord_to_link_force_grad_mat(robot, state, force_order=force_order, dim=dim)
    mat = np.zeros((len(links) * n_f, full.shape[1]))

    for i, link in enumerate(links):
        if link is None:
            raise ValueError("link_name_list contains invalid link name")

        src = link.id * n_f
        dst = i * n_f
        mat[dst:dst+n_f, :] = full[src:src+n_f, :]

    return mat


def _batch_selected_coord_to_link_momentum_grad_mat(
    robot: RobotStruct,
    state: dict,
    links: list,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    dof = dim_to_dof(dim)
    n_j = dof * order
    n_m = dof * (order - 1)
    joint_tan_mat = _batch_total_coord_to_joint_tan_vel_grad_mat(robot, state, order, dim)
    mat = np.zeros(joint_tan_mat.shape[:-2] + (len(links) * n_m, joint_tan_mat.shape[-1]), dtype=joint_tan_mat.dtype)

    for i, link in enumerate(links):
        if link is None:
            raise ValueError("link_name_list contains invalid link name")

        link_route = []
        joint_route = []
        robot.route_target_link(link, link_route, joint_route)
        link_sp_tan_inv = state_dict_to_cmtm(state, link.name, "link", order).tangent_mat_inv()[..., dof:, :]

        row = i * n_m
        row_block = np.zeros(joint_tan_mat.shape[:-2] + (n_m, joint_tan_mat.shape[-1]), dtype=joint_tan_mat.dtype)
        for j in joint_route:
            joint = robot.joints[j]
            rel_cmtm = state_dict_to_rel_cmtm(state, link.name, robot.links[joint.child_link_id].name, "link", order)
            col = j * n_j
            row_block += (link_sp_tan_inv @ rel_cmtm.mat_adj()) @ joint_tan_mat[..., col:col+n_j, :]

        inertia_block = inertia_diag_mat(spatial_inertia(link.mass, link.inertia, link.cog), order - 1)
        mat[..., row:row+n_m, :] = inertia_block @ row_block

    return mat


def _batch_selected_coord_to_world_link_momentum_grad_mat(
    robot: RobotStruct,
    state: dict,
    links: list,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    dof = dim_to_dof(dim)
    n_m = dof * (order - 1)
    mat_link_mom = _batch_selected_coord_to_link_momentum_grad_mat(robot, state, links, order=order, dim=dim)
    mat_tan_kine = _batch_selected_coord_to_link_tan_vel_grad_mat(
        robot, state, links, out_order=order - 1, in_order=order, dim=dim
    )
    mat = np.zeros_like(mat_link_mom)

    factorial = Factorial.mat(order - 1, dof)
    factorial_inv = Factorial.mat_inv(order - 1, dof)
    for i, link in enumerate(links):
        if link is None:
            raise ValueError("link_name_list contains invalid link name")

        row = i * n_m
        cmtm_wrench = state_dict_to_cmtm_wrench(state, link.name, "link", order - 1)
        block_mom = factorial @ cmtm_wrench.mat_adj() @ factorial_inv
        link_momentum = state_dict_to_cmvec(state, link.name, "link", "momentum", order - 1)
        block_tan = factorial @ cmtm_wrench.mat_var_x_arb_vec_jacob(link_momentum, frame="bframe")
        mat[..., row:row+n_m, :] = (
            block_mom @ mat_link_mom[..., row:row+n_m, :]
            + block_tan @ mat_tan_kine[..., row:row+n_m, :]
        )

    return mat


def _batch_total_coord_to_link_force_grad_mat(
    robot: RobotStruct,
    state: dict,
    force_order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    partial_mom = _batch_total_partial_momentum_to_force_grad_mat(robot, state, force_order, dim)
    partial_vel = _batch_total_partial_link_sp_vel_to_link_force_grad_mat(robot, state, force_order, dim)
    mat_link_mom = _batch_selected_coord_to_link_momentum_grad_mat(robot, state, robot.links, order=force_order + 2, dim=dim)
    mat_link_vel = _batch_selected_coord_to_link_vel_grad_mat(robot, state, robot.links, order=force_order + 2, dim=dim)
    return partial_mom @ mat_link_mom + partial_vel @ mat_link_vel


def _batch_selected_coord_to_link_force_grad_mat(
    robot: RobotStruct,
    state: dict,
    links: list,
    force_order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    dof = dim_to_dof(dim)
    n_f = dof * force_order
    full = _batch_total_coord_to_link_force_grad_mat(robot, state, force_order=force_order, dim=dim)
    mat = np.zeros(full.shape[:-2] + (len(links) * n_f, full.shape[-1]), dtype=full.dtype)

    for i, link in enumerate(links):
        if link is None:
            raise ValueError("link_name_list contains invalid link name")

        src = link.id * n_f
        dst = i * n_f
        mat[..., dst:dst+n_f, :] = full[..., src:src+n_f, :]

    return mat


def _batch_selected_coord_to_world_joint_momentum_grad_mat(
    robot: RobotStruct,
    state: dict,
    joints: list,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    dof = dim_to_dof(dim)
    n_m = dof * (order - 1)
    links = []
    link_index = {}
    joint_link_routes = []

    for joint in joints:
        if joint is None:
            raise ValueError("joint_name_list contains invalid joint name")

        link_route = []
        joint_route = []
        robot.route_end_joints(joint, link_route, joint_route)
        joint_links = []
        for link_id in link_route:
            joint_links.append(link_id)
            if link_id not in link_index:
                link_index[link_id] = len(links)
                links.append(robot.links[link_id])
        joint_link_routes.append(joint_links)

    mat_link_wmom = _batch_selected_coord_to_world_link_momentum_grad_mat(robot, state, links, order=order, dim=dim)
    mat = np.zeros(mat_link_wmom.shape[:-2] + (len(joints) * n_m, mat_link_wmom.shape[-1]), dtype=mat_link_wmom.dtype)

    for i, joint_links in enumerate(joint_link_routes):
        row = i * n_m
        for link_id in joint_links:
            src = link_index[link_id] * n_m
            mat[..., row:row+n_m, :] += mat_link_wmom[..., src:src+n_m, :]

    return mat


def _batch_selected_coord_to_joint_momentum_grad_mat(
    robot: RobotStruct,
    state: dict,
    joints: list,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    dof = dim_to_dof(dim)
    n_m = dof * (order - 1)
    child_links = []
    for joint in joints:
        if joint is None:
            raise ValueError("joint_name_list contains invalid joint name")
        child_links.append(robot.links[joint.child_link_id])

    mat_world_joint_mom = _batch_selected_coord_to_world_joint_momentum_grad_mat(robot, state, joints, order=order, dim=dim)
    mat_child_link_tan = _batch_selected_coord_to_link_tan_vel_grad_mat(
        robot,
        state,
        child_links,
        out_order=order - 1,
        in_order=order,
        dim=dim,
    )
    mat = np.zeros_like(mat_world_joint_mom)

    factorial = Factorial.mat(order - 1, dof)
    factorial_inv = Factorial.mat_inv(order - 1, dof)
    for i, joint in enumerate(joints):
        row = i * n_m
        child_link = child_links[i]
        cmtm_wrench = state_dict_to_cmtm_wrench(state, child_link.name, "link", order - 1)
        block_world = factorial @ cmtm_wrench.mat_inv_adj() @ factorial_inv
        local_joint_momentum = state_dict_to_cmvec(state, joint.name, "joint", "momentum", order - 1)
        world_vec = _batched_matvec(factorial, apply_mat_adj(cmtm_wrench, local_joint_momentum.cm_vec()))
        world_joint_momentum = CMVector(world_vec.reshape(world_vec.shape[:-1] + (order - 1, -1)))
        block_tan = factorial @ cmtm_wrench.mat_inv_var_x_arb_vec_jacob(world_joint_momentum, frame="bframe")
        mat[..., row:row+n_m, :] = (
            block_world @ mat_world_joint_mom[..., row:row+n_m, :]
            + block_tan @ mat_child_link_tan[..., row:row+n_m, :]
        )

    return mat


def _batch_total_coord_to_joint_force_grad_mat(
    robot: RobotStruct,
    state: dict,
    force_order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    dof = dim_to_dof(dim)
    partial_mom = take_joint_child_link_matrix_blocks(
        _batch_total_partial_momentum_to_force_grad_mat(robot, state, force_order, dim),
        robot,
        dof * force_order,
        dof * (force_order + 1),
    )
    mat_joint_mom = _batch_selected_coord_to_joint_momentum_grad_mat(
        robot, state, robot.joints, order=force_order + 2, dim=dim
    )
    partial_joint_vel = _batch_total_partial_link_sp_vel_to_joint_force_grad_mat(robot, state, force_order, dim)
    mat_link_vel = _batch_selected_coord_to_link_vel_grad_mat(robot, state, robot.links, order=force_order + 2, dim=dim)
    mat_child_link_vel = take_joint_child_link_blocks(
        mat_link_vel, robot, dof * (force_order + 2), axis=-2
    )
    return (
        partial_mom @ mat_joint_mom
        + partial_joint_vel @ mat_child_link_vel
    )


def _batch_selected_coord_to_joint_force_grad_mat(
    robot: RobotStruct,
    state: dict,
    joints: list,
    force_order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    dof = dim_to_dof(dim)
    n_f = dof * force_order
    full = _batch_total_coord_to_joint_force_grad_mat(robot, state, force_order=force_order, dim=dim)
    mat = np.zeros(full.shape[:-2] + (len(joints) * n_f, full.shape[-1]), dtype=full.dtype)

    for i, joint in enumerate(joints):
        if joint is None:
            raise ValueError("joint_name_list contains invalid joint name")

        src = joint.id * n_f
        dst = i * n_f
        mat[..., dst:dst+n_f, :] = full[..., src:src+n_f, :]

    return mat


def _batch_total_coord_to_joint_torque_grad_mat(
    robot: RobotStruct,
    state: dict,
    torque_order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    return total_joint_wrench_to_joint_torque_mat(robot, torque_order, dim=dim) @ _batch_total_coord_to_joint_force_grad_mat(
        robot, state, torque_order, dim=dim
    )


def _batch_selected_coord_to_joint_torque_grad_mat(
    robot: RobotStruct,
    state: dict,
    joints: list,
    torque_order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    full = _batch_total_coord_to_joint_torque_grad_mat(robot, state, torque_order=torque_order, dim=dim)
    mat = np.zeros(full.shape[:-2] + (sum(joint.dof * torque_order for joint in joints), full.shape[-1]), dtype=full.dtype)

    row_torque = 0
    for joint in joints:
        if joint is None:
            raise ValueError("joint_name_list contains invalid joint name")

        rows = joint.dof * torque_order
        src = joint.dof_index * torque_order
        mat[..., row_torque:row_torque+rows, :] = full[..., src:src+rows, :]
        row_torque += rows

    return mat


def _selected_coord_to_world_joint_momentum_grad_mat(
    robot: RobotStruct,
    state: dict,
    joints: list,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    if joints and _is_batched_kinematics_state(
        robot,
        state,
        [StateType("link", robot.links[joints[0].child_link_id].name, "vel")],
        order,
    ):
        return _batch_selected_coord_to_world_joint_momentum_grad_mat(robot, state, joints, order, dim)

    dof = dim_to_dof(dim)
    n_m = dof * (order - 1)
    links = []
    link_index = {}
    joint_link_routes = []

    for joint in joints:
        if joint is None:
            raise ValueError("joint_name_list contains invalid joint name")

        link_route = []
        joint_route = []
        robot.route_end_joints(joint, link_route, joint_route)
        joint_links = []
        for link_id in link_route:
            joint_links.append(link_id)
            if link_id not in link_index:
                link_index[link_id] = len(links)
                links.append(robot.links[link_id])
        joint_link_routes.append(joint_links)

    mat_link_wmom = _selected_coord_to_world_link_momentum_grad_mat(robot, state, links, order=order, dim=dim)
    mat = np.zeros((len(joints) * n_m, mat_link_wmom.shape[1]))

    for i, joint_links in enumerate(joint_link_routes):
        row = i * n_m
        for link_id in joint_links:
            src = link_index[link_id] * n_m
            mat[row:row+n_m, :] += mat_link_wmom[src:src+n_m, :]

    return mat


def _selected_coord_to_joint_momentum_grad_mat(
    robot: RobotStruct,
    state: dict,
    joints: list,
    order: int = 3,
    dim: int = 3,
) -> np.ndarray:
    if joints and _is_batched_kinematics_state(
        robot,
        state,
        [StateType("link", robot.links[joints[0].child_link_id].name, "vel")],
        order,
    ):
        return _batch_selected_coord_to_joint_momentum_grad_mat(robot, state, joints, order, dim)

    dof = dim_to_dof(dim)
    n_m = dof * (order - 1)
    child_links = []
    for joint in joints:
        if joint is None:
            raise ValueError("joint_name_list contains invalid joint name")
        child_links.append(robot.links[joint.child_link_id])

    mat_world_joint_mom = _selected_coord_to_world_joint_momentum_grad_mat(robot, state, joints, order=order, dim=dim)
    mat_child_link_tan = _selected_coord_to_link_tan_vel_grad_mat(
        robot,
        state,
        child_links,
        out_order=order - 1,
        in_order=order,
        dim=dim,
    )
    mat = np.zeros_like(mat_world_joint_mom)

    factorial = Factorial.mat(order - 1, dof)
    factorial_inv = Factorial.mat_inv(order - 1, dof)
    for i, joint in enumerate(joints):
        row = i * n_m
        child_link = child_links[i]
        cmtm_wrench = state_dict_to_cmtm_wrench(state, child_link.name, "link", order - 1)
        block_world = factorial @ cmtm_wrench.mat_inv_adj() @ factorial_inv
        local_joint_momentum = state_dict_to_cmvec(state, joint.name, "joint", "momentum", order - 1)
        world_joint_momentum = CMVector(
            (factorial @ apply_mat_adj(cmtm_wrench, local_joint_momentum.cm_vec())).reshape(order - 1, -1)
        )
        block_tan = factorial @ cmtm_wrench.mat_inv_var_x_arb_vec_jacob(world_joint_momentum, frame="bframe")
        mat[row:row+n_m, :] = (
            block_world @ mat_world_joint_mom[row:row+n_m, :]
            + block_tan @ mat_child_link_tan[row:row+n_m, :]
        )

    return mat


def _selected_coord_to_joint_force_grad_mat(
    robot: RobotStruct,
    state: dict,
    joints: list,
    force_order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    if joints and _is_batched_kinematics_state(
        robot,
        state,
        [StateType("link", robot.links[joints[0].child_link_id].name, "vel")],
        force_order + 2,
    ):
        return _batch_selected_coord_to_joint_force_grad_mat(robot, state, joints, force_order, dim)

    dof = dim_to_dof(dim)
    n_f = dof * force_order
    full = total_coord_to_joint_force_grad_mat(robot, state, force_order=force_order, dim=dim)
    mat = np.zeros((len(joints) * n_f, full.shape[1]))

    for i, joint in enumerate(joints):
        if joint is None:
            raise ValueError("joint_name_list contains invalid joint name")

        src = joint.id * n_f
        dst = i * n_f
        mat[dst:dst+n_f, :] = full[src:src+n_f, :]

    return mat


def _selected_coord_to_joint_torque_grad_mat(
    robot: RobotStruct,
    state: dict,
    joints: list,
    torque_order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    if joints and _is_batched_kinematics_state(
        robot,
        state,
        [StateType("link", robot.links[joints[0].child_link_id].name, "vel")],
        torque_order + 2,
    ):
        return _batch_selected_coord_to_joint_torque_grad_mat(robot, state, joints, torque_order, dim)

    full = total_coord_to_joint_torque_grad_mat(robot, state, torque_order=torque_order, dim=dim)
    mat = np.zeros((sum(joint.dof * torque_order for joint in joints), full.shape[1]))

    row_torque = 0
    for joint in joints:
        if joint is None:
            raise ValueError("joint_name_list contains invalid joint name")

        rows = joint.dof * torque_order
        src = joint.dof_index * torque_order
        mat[row_torque:row_torque+rows, :] = full[src:src+rows, :]
        row_torque += rows

    return mat

def link_jacobian(robot : RobotStruct, state : dict, link_name_list : list[str], order : int = 3, dim : int = 3) -> np.ndarray:
    links = robot.link_list(link_name_list)
    mat = _selected_coord_to_link_vel_grad_mat(robot, state, links, order=order, dim=dim)
    dof = dim_to_dof(dim)
    jacobs = np.zeros((dof * order * len(links), robot.dof * order))
    
    for i, link in enumerate(links):
        jacobs[i*dof*order:(i+1)*dof*order, :] = mat[i*dof*order:(i+1)*dof*order, :]
    return jacobs

def link_momentum_jacobian(robot : RobotStruct, state : dict, link_name_list : list[str], momentum_order : int = 1, dim : int = 3) -> np.ndarray:
    links = robot.link_list(link_name_list)
    mat = _selected_coord_to_link_momentum_grad_mat(robot, state, links, order=momentum_order+1, dim=dim)
    dof = dim_to_dof(dim)
    jacobs = np.zeros((dof * momentum_order * len(links), robot.dof * (momentum_order+1)))

    for i, link in enumerate(links):
        jacobs[i*dof*momentum_order:(i+1)*dof*momentum_order, :] = mat[i*dof*momentum_order:(i+1)*dof*momentum_order, :]
    return jacobs

def world_link_momentum_jacobian(robot : RobotStruct, state : dict, link_name_list : list[str], momentum_order : int = 1, dim : int = 3) -> np.ndarray:
    links = robot.link_list(link_name_list)
    mat = _selected_coord_to_world_link_momentum_grad_mat(robot, state, links, order=momentum_order+1, dim=dim)
    dof = dim_to_dof(dim)
    jacobs = np.zeros((dof * (momentum_order) * len(links), robot.dof * (momentum_order+1)))

    for i, link in enumerate(links):
        jacobs[i*dof*momentum_order:(i+1)*dof*momentum_order, :] = mat[i*dof*momentum_order:(i+1)*dof*momentum_order, :]
    return jacobs

def world_joint_momentum_jacobian(robot : RobotStruct, state : dict, joint_name_list : list[str], momentum_order : int = 1, dim : int = 3) -> np.ndarray:
    joints = robot.joint_list(joint_name_list)
    if any(joint is None for joint in joints):
        raise ValueError("joint_name_list contains invalid joint name")
    return _selected_coord_to_world_joint_momentum_grad_mat(robot, state, joints, order=momentum_order+1, dim=dim)

def joint_momentum_jacobian(robot : RobotStruct, state : dict, joint_name_list : list[str], momentum_order : int = 1, dim : int = 3) -> np.ndarray:
    joints = robot.joint_list(joint_name_list)
    if any(joint is None for joint in joints):
        raise ValueError("joint_name_list contains invalid joint name")
    return _selected_coord_to_joint_momentum_grad_mat(robot, state, joints, order=momentum_order+1, dim=dim)

def link_force_jacobian(robot : RobotStruct, state : dict, link_name_list : list[str], force_order : int = 1, dim : int = 3) -> np.ndarray:
    links = robot.link_list(link_name_list)
    mat = _selected_coord_to_link_force_grad_mat(robot, state, links, force_order=force_order, dim=dim)
    dof = dim_to_dof(dim)
    jacobs = np.zeros((dof * force_order * len(links), robot.dof * (force_order+2)))

    for i, link in enumerate(links):
        jacobs[i*dof*force_order:(i+1)*dof*force_order, :] = mat[i*dof*force_order:(i+1)*dof*force_order, :]
    return jacobs

def joint_force_jacobian(robot : RobotStruct, state : dict, joint_name_list : list[str], force_order : int = 1, dim : int = 3) -> np.ndarray:
    joints = robot.joint_list(joint_name_list)
    if any(joint is None for joint in joints):
        raise ValueError("joint_name_list contains invalid joint name")
    return _selected_coord_to_joint_force_grad_mat(robot, state, joints, force_order=force_order, dim=dim)

def joint_torque_jacobian(robot : RobotStruct, state : dict, joint_name_list : list[str], torque_order : int = 1, dim : int = 3) -> np.ndarray:
    joints = robot.joint_list(joint_name_list)
    if any(joint is None for joint in joints):
        raise ValueError("joint_name_list contains invalid joint name")
    return _selected_coord_to_joint_torque_grad_mat(robot, state, joints, torque_order=torque_order, dim=dim)

def outward_kinematics_jacobian(robot : RobotStruct, state : dict, state_type_list : list[StateType], max_time_order = None, dim : int = 3, list_output : bool = False) -> np.ndarray:
    kine_state_type_list = StateType.filter_list_by_kinematics(state_type_list)
    if max_time_order is None:
        max_time_order = StateType.max_time_order(kine_state_type_list)
    dim_dof = dim_to_dof(dim)
    link_names = StateType.get_owner_names_from_list(kine_state_type_list)
    links = robot.link_list(link_names)
    if _is_batched_kinematics_state(robot, state, kine_state_type_list, max_time_order):
        mat = _batch_selected_coord_to_link_vel_grad_mat(robot, state, links, order=max_time_order, dim=dim)
    else:
        mat = _selected_coord_to_link_vel_grad_mat(robot, state, links, order=max_time_order, dim=dim)
    link_offsets = {link.name: i * dim_dof * max_time_order for i, link in enumerate(links)}

    jacob_list = []
    for st in kine_state_type_list:
        link = robot.link(st.owner_name)
        if link is None:
            raise ValueError(f"Invalid link name: {st.owner_name}")
        base = link_offsets[link.name]
        state_dof = data_type_dof(st.data_type, dim=dim)
        offset = dim_dof*(st.time_order-1) + data_type_offset(st.data_type) * state_dof
        jacob_part = mat[..., base + offset : base + offset + state_dof, :]
        jacob_list.append(jacob_part)

    if list_output:
        return jacob_list
    else:
        return np.concatenate(jacob_list, axis=-2)

def outward_kinematics_jacobian_matvec(robot : RobotStruct, state : dict, state_type_list : list[StateType], vec : np.ndarray, max_time_order = None, dim : int = 3, list_output : bool = False) -> np.ndarray:
    kine_state_type_list = StateType.filter_list_by_kinematics(state_type_list)
    if max_time_order is None:
        max_time_order = StateType.max_time_order(kine_state_type_list)

    dim_dof = dim_to_dof(dim)
    link_offsets = None
    if np.asarray(vec).ndim > 1 or _is_batched_kinematics_state(robot, state, kine_state_type_list, max_time_order):
        link_names = StateType.get_owner_names_from_list(kine_state_type_list)
        links = robot.link_list(link_names)
        mat = _batch_selected_coord_to_link_vel_grad_mat(robot, state, links, order=max_time_order, dim=dim)
        mat_vec = _batched_matvec(mat, vec)
        link_offsets = {link.name: i * dim_dof * max_time_order for i, link in enumerate(links)}
    else:
        mat_vec = total_coord_to_link_vel_grad_matvec(robot, state, vec, order=max_time_order, dim=dim)

    vec_list = []
    for st in kine_state_type_list:
        link = robot.link(st.owner_name)
        if link is None:
            raise ValueError(f"Invalid link name: {st.owner_name}")
        base = link_offsets[link.name] if link_offsets is not None else link.id * dim_dof * max_time_order
        state_dof = data_type_dof(st.data_type, dim=dim)
        offset = dim_dof*(st.time_order-1) + data_type_offset(st.data_type) * state_dof
        vec_part = mat_vec[..., base + offset : base + offset + state_dof]
        vec_list.append(vec_part)

    if list_output:
        return vec_list
    else:
        return np.concatenate(vec_list, axis=-1)


def outward_kinematics_jacobian_matmul_rhs(robot : RobotStruct, state : dict, state_type_list : list[StateType], rhs : np.ndarray, max_time_order = None, dim : int = 3, list_output : bool = False) -> np.ndarray:
    kine_state_type_list = StateType.filter_list_by_kinematics(state_type_list)
    if max_time_order is None:
        max_time_order = StateType.max_time_order(kine_state_type_list)

    dim_dof = dim_to_dof(dim)
    link_names = StateType.get_owner_names_from_list(kine_state_type_list)
    links = robot.link_list(link_names)
    if np.asarray(rhs).ndim > 2 or _is_batched_kinematics_state(robot, state, kine_state_type_list, max_time_order):
        mat = _batch_selected_coord_to_link_vel_grad_mat(robot, state, links, order=max_time_order, dim=dim)
    else:
        mat = _selected_coord_to_link_vel_grad_mat(robot, state, links, order=max_time_order, dim=dim)
    mat_rhs = _matmul_rhs(mat, rhs)
    link_offsets = {link.name: i * dim_dof * max_time_order for i, link in enumerate(links)}

    rhs_list = []
    for st in kine_state_type_list:
        link = robot.link(st.owner_name)
        if link is None:
            raise ValueError(f"Invalid link name: {st.owner_name}")
        base = link_offsets[link.name]
        state_dof = data_type_dof(st.data_type, dim=dim)
        offset = dim_dof*(st.time_order-1) + data_type_offset(st.data_type) * state_dof
        rhs_part = mat_rhs[..., base + offset : base + offset + state_dof, :]
        rhs_list.append(rhs_part)

    if list_output:
        return rhs_list
    else:
        return np.concatenate(rhs_list, axis=-2)


def _batch_outward_dynamics_jacobian_matvec(
    robot: RobotStruct,
    state: dict,
    state_type_list: list[StateType],
    vec: np.ndarray,
    max_time_order: int,
    dim: int = 3,
    list_output: bool = False,
) -> np.ndarray:
    dof = dim_to_dof(dim)
    force_order = max_time_order - 2

    vec_kine = _batch_total_coord_to_link_vel_grad_matvec(robot, state, vec, order=max_time_order, dim=dim)
    vec_tan_kine = _batch_total_coord_to_link_tan_vel_grad_matvec(
        robot, state, vec, out_order=max_time_order - 1, in_order=max_time_order, dim=dim
    )
    vec_link_mom = _batch_total_coord_to_link_momentum_grad_matvec(
        robot, state, vec, order=max_time_order, dim=dim, vec_link_vel=vec_kine
    )

    vec_link_wmom = _batch_total_partial_link_momentum_to_world_link_momentum_grad_matvec(
        robot, state, vec_link_mom, order=max_time_order, dim=dim
    ) + _batch_total_partial_link_tan_vel_to_world_link_momentum_grad_matvec(
        robot, state, vec_tan_kine, order=max_time_order, dim=dim
    )

    vec_joint_wmom = _batch_total_world_link_wrench_to_world_joint_wrench_matvec(
        robot, vec_link_wmom, order=max_time_order - 1, dim=dim
    )

    child_vec_tan_kine = take_joint_child_link_blocks(
        vec_tan_kine, robot, (max_time_order - 1) * dof
    )
    vec_joint_mom = _batch_total_partial_world_joint_momentum_to_joint_momentum_grad_matvec(
        robot, state, vec_joint_wmom, max_time_order, dim
    ) + _batch_total_partial_link_tan_vel_to_joint_momentum_grad_matvec(
        robot, state, child_vec_tan_kine, max_time_order, dim
    )

    if max_time_order >= 3:
        vec_link_force = _batch_total_partial_momentum_to_force_grad_matvec(
            robot, state, vec_link_mom, force_order=force_order, dim=dim
        ) + _batch_total_partial_link_sp_vel_to_link_force_grad_matvec(
            robot, state, vec_kine, force_order=force_order, dim=dim
        )

        partial_mom = take_joint_child_link_matrix_blocks(
            _batch_total_partial_momentum_to_force_grad_mat(robot, state, force_order=force_order, dim=dim),
            robot,
            dof * force_order,
            dof * (force_order + 1),
        )
        child_vec_kine = take_joint_child_link_blocks(
            vec_kine, robot, max_time_order * dof
        )
        vec_joint_force = _batched_matvec(
            partial_mom,
            vec_joint_mom,
        ) + _batch_total_partial_link_sp_vel_to_joint_force_grad_matvec(
            robot, state, child_vec_kine, force_order=force_order, dim=dim
        )

        vec_joint_torque = _batch_total_joint_wrench_to_joint_torque_matvec(
            robot, vec_joint_force, torque_order=force_order, dim=dim
        )

    vec_list = []
    for st in state_type_list:
        if st.owner_type == "link":
            link = robot.link(st.owner_name)
            if link is None:
                raise ValueError(f"Invalid link name: {st.owner_name}")
        elif st.owner_type == "joint":
            joint = robot.joint(st.owner_name)
            if joint is None:
                raise ValueError(f"Invalid joint name: {st.owner_name}")

        order = st.key_order - 1

        if st.data_type in keys_kinematics:
            base = link.id * dof * max_time_order
            vec_part = vec_kine[..., base + dof*order : base + dof*(order+1)]
        elif st.data_type in keys_momentum:
            if st.owner_type == "link":
                base = link.id * dof * (max_time_order - 1)
                source = vec_link_wmom if st.frame_name == "world" else vec_link_mom
            elif st.owner_type == "joint":
                base = joint.id * dof * (max_time_order - 1)
                source = vec_joint_wmom if st.frame_name == "world" else vec_joint_mom
            vec_part = source[..., base + dof*order : base + dof*(order+1)]
        elif st.data_type in keys_force:
            if max_time_order < 3:
                raise ValueError("force jacobian matvec requires max_time_order >= 3")
            if st.owner_type == "link":
                base = link.id * dof * (max_time_order - 2)
                vec_part = vec_link_force[..., base + dof*order : base + dof*(order+1)]
            elif st.owner_type == "joint":
                base = joint.id * dof * (max_time_order - 2)
                vec_part = vec_joint_force[..., base + dof*order : base + dof*(order+1)]
        elif st.data_type in keys_torque:
            if max_time_order < 3:
                raise ValueError("torque jacobian matvec requires max_time_order >= 3")
            if st.owner_type != "joint":
                raise ValueError("torque can be specified only for joint owner type")
            base = joint.dof_index * (max_time_order - 2)
            vec_part = vec_joint_torque[..., base + joint.dof*order : base + joint.dof*(order+1)]
        else:
            raise ValueError(f"Unsupported data_type for jacobian matvec: {st.data_type}")

        vec_list.append(vec_part)

    if list_output:
        return vec_list
    return np.concatenate(vec_list, axis=-1)


def _batch_outward_dynamics_jacobian_matmul_rhs(
    robot: RobotStruct,
    state: dict,
    state_type_list: list[StateType],
    rhs: np.ndarray,
    max_time_order: int,
    dim: int = 3,
    list_output: bool = False,
) -> np.ndarray:
    dof = dim_to_dof(dim)
    force_order = max_time_order - 2

    rhs_kine = _batch_total_coord_to_link_vel_grad_matmul_rhs(robot, state, rhs, order=max_time_order, dim=dim)
    rhs_tan_kine = _batch_total_coord_to_link_tan_vel_grad_matmul_rhs(
        robot, state, rhs, out_order=max_time_order - 1, in_order=max_time_order, dim=dim
    )
    rhs_link_mom = _batch_total_coord_to_link_momentum_grad_matmul_rhs(
        robot, state, rhs, order=max_time_order, dim=dim, rhs_link_vel=rhs_kine
    )

    rhs_link_wmom = _batch_total_partial_link_momentum_to_world_link_momentum_grad_matmul_rhs(
        robot, state, rhs_link_mom, order=max_time_order, dim=dim
    ) + _batch_total_partial_link_tan_vel_to_world_link_momentum_grad_matmul_rhs(
        robot, state, rhs_tan_kine, order=max_time_order, dim=dim
    )

    rhs_joint_wmom = _batch_total_world_link_wrench_to_world_joint_wrench_matmul_rhs(
        robot, rhs_link_wmom, order=max_time_order - 1, dim=dim
    )

    child_rhs_tan_kine = take_joint_child_link_blocks(
        rhs_tan_kine, robot, (max_time_order - 1) * dof, axis=-2
    )
    rhs_joint_mom = _batch_total_partial_world_joint_momentum_to_joint_momentum_grad_matmul_rhs(
        robot, state, rhs_joint_wmom, max_time_order, dim
    ) + _batch_total_partial_link_tan_vel_to_joint_momentum_grad_matmul_rhs(
        robot, state, child_rhs_tan_kine, max_time_order, dim
    )

    if max_time_order >= 3:
        rhs_link_force = _batch_total_partial_momentum_to_force_grad_matmul_rhs(
            robot, state, rhs_link_mom, force_order=force_order, dim=dim
        ) + _batch_total_partial_link_sp_vel_to_link_force_grad_matmul_rhs(
            robot, state, rhs_kine, force_order=force_order, dim=dim
        )

        partial_mom = take_joint_child_link_matrix_blocks(
            _batch_total_partial_momentum_to_force_grad_mat(robot, state, force_order=force_order, dim=dim),
            robot,
            dof * force_order,
            dof * (force_order + 1),
        )
        child_rhs_kine = take_joint_child_link_blocks(
            rhs_kine, robot, max_time_order * dof, axis=-2
        )
        rhs_joint_force = _matmul_rhs(
            partial_mom,
            rhs_joint_mom,
        ) + _batch_total_partial_link_sp_vel_to_joint_force_grad_matmul_rhs(
            robot, state, child_rhs_kine, force_order=force_order, dim=dim
        )

        rhs_joint_torque = _batch_total_joint_wrench_to_joint_torque_matmul_rhs(
            robot, rhs_joint_force, torque_order=force_order, dim=dim
        )

    rhs_list = []
    for st in state_type_list:
        if st.owner_type == "link":
            link = robot.link(st.owner_name)
            if link is None:
                raise ValueError(f"Invalid link name: {st.owner_name}")
        elif st.owner_type == "joint":
            joint = robot.joint(st.owner_name)
            if joint is None:
                raise ValueError(f"Invalid joint name: {st.owner_name}")

        order = st.key_order - 1

        if st.data_type in keys_kinematics:
            base = link.id * dof * max_time_order
            rhs_part = rhs_kine[..., base + dof*order : base + dof*(order+1), :]
        elif st.data_type in keys_momentum:
            if st.owner_type == "link":
                base = link.id * dof * (max_time_order - 1)
                source = rhs_link_wmom if st.frame_name == "world" else rhs_link_mom
            elif st.owner_type == "joint":
                base = joint.id * dof * (max_time_order - 1)
                source = rhs_joint_wmom if st.frame_name == "world" else rhs_joint_mom
            rhs_part = source[..., base + dof*order : base + dof*(order+1), :]
        elif st.data_type in keys_force:
            if max_time_order < 3:
                raise ValueError("force jacobian matmul requires max_time_order >= 3")
            if st.owner_type == "link":
                base = link.id * dof * (max_time_order - 2)
                rhs_part = rhs_link_force[..., base + dof*order : base + dof*(order+1), :]
            elif st.owner_type == "joint":
                base = joint.id * dof * (max_time_order - 2)
                rhs_part = rhs_joint_force[..., base + dof*order : base + dof*(order+1), :]
        elif st.data_type in keys_torque:
            if max_time_order < 3:
                raise ValueError("torque jacobian matmul requires max_time_order >= 3")
            if st.owner_type != "joint":
                raise ValueError("torque can be specified only for joint owner type")
            base = joint.dof_index * (max_time_order - 2)
            rhs_part = rhs_joint_torque[..., base + joint.dof*order : base + joint.dof*(order+1), :]
        else:
            raise ValueError(f"Unsupported data_type for jacobian matmul: {st.data_type}")

        rhs_list.append(rhs_part)

    if list_output:
        return rhs_list
    return np.concatenate(rhs_list, axis=-2)


def _outward_link_only_jacobian(
    robot: RobotStruct,
    state: dict,
    state_type_list: list[StateType],
    max_time_order: int,
    dim: int = 3,
    list_output: bool = False,
) -> np.ndarray:
    dof = dim_to_dof(dim)
    link_names = StateType.get_owner_names_from_list(state_type_list)
    links = robot.link_list(link_names)
    link_index = {}
    for i, link in enumerate(links):
        if link is None:
            raise ValueError("state_type_list contains invalid link name")
        link_index[link.name] = i

    cache = {}

    def get_mat_kine() -> np.ndarray:
        if "mat_kine" not in cache:
            cache["mat_kine"] = _selected_coord_to_link_vel_grad_mat(robot, state, links, order=max_time_order, dim=dim)
        return cache["mat_kine"]

    def get_mat_link_mom() -> np.ndarray:
        if "mat_link_mom" not in cache:
            cache["mat_link_mom"] = _selected_coord_to_link_momentum_grad_mat(robot, state, links, order=max_time_order, dim=dim)
        return cache["mat_link_mom"]

    def get_mat_link_wmom() -> np.ndarray:
        if "mat_link_wmom" not in cache:
            cache["mat_link_wmom"] = _selected_coord_to_world_link_momentum_grad_mat(robot, state, links, order=max_time_order, dim=dim)
        return cache["mat_link_wmom"]

    def get_mat_link_force() -> np.ndarray:
        if max_time_order < 3:
            raise ValueError("force jacobian requires max_time_order >= 3")
        if "mat_link_force" not in cache:
            cache["mat_link_force"] = _selected_coord_to_link_force_grad_mat(robot, state, links, force_order=max_time_order-2, dim=dim)
        return cache["mat_link_force"]

    jacob_list = []
    for st in state_type_list:
        link = robot.link(st.owner_name)
        if link is None:
            raise ValueError(f"Invalid link name: {st.owner_name}")

        link_id = link_index[link.name]
        order = st.key_order - 1

        if st.data_type in keys_kinematics:
            base = link_id * dof * max_time_order
            state_dof = data_type_dof(st.data_type, dim=dim)
            offset = dof*(st.time_order-1) + data_type_offset(st.data_type) * state_dof
            jacob_part = get_mat_kine()[..., base + offset : base + offset + state_dof, :]
        elif st.data_type in keys_momentum:
            base = link_id * dof * (max_time_order-1)
            if st.frame_name == "world":
                jacob_part = get_mat_link_wmom()[..., base + dof*order : base + dof*(order+1), :]
            else:
                jacob_part = get_mat_link_mom()[..., base + dof*order : base + dof*(order+1), :]
        elif st.data_type in keys_force:
            base = link_id * dof * (max_time_order-2)
            jacob_part = get_mat_link_force()[..., base + dof*order : base + dof*(order+1), :]
        else:
            raise ValueError("link-only fast path supports only kinematics, momentum, and force states")

        jacob_list.append(jacob_part)

    if list_output:
        return jacob_list
    return np.concatenate(jacob_list, axis=-2)


def _outward_joint_only_jacobian(
    robot: RobotStruct,
    state: dict,
    state_type_list: list[StateType],
    max_time_order: int,
    dim: int = 3,
    list_output: bool = False,
) -> np.ndarray:
    dof = dim_to_dof(dim)
    joint_names = StateType.get_owner_names_from_list(state_type_list)
    joints = robot.joint_list(joint_names)
    joint_index = {}
    torque_offset = {}
    running_torque_offset = 0

    for i, joint in enumerate(joints):
        if joint is None:
            raise ValueError("state_type_list contains invalid joint name")
        joint_index[joint.name] = i
        torque_offset[joint.name] = running_torque_offset
        running_torque_offset += joint.dof * max(max_time_order - 2, 0)

    cache = {}

    def get_mat_joint_wmom() -> np.ndarray:
        if "mat_joint_wmom" not in cache:
            cache["mat_joint_wmom"] = _selected_coord_to_world_joint_momentum_grad_mat(
                robot, state, joints, order=max_time_order, dim=dim
            )
        return cache["mat_joint_wmom"]

    def get_mat_joint_mom() -> np.ndarray:
        if "mat_joint_mom" not in cache:
            cache["mat_joint_mom"] = _selected_coord_to_joint_momentum_grad_mat(
                robot, state, joints, order=max_time_order, dim=dim
            )
        return cache["mat_joint_mom"]

    def get_mat_joint_force() -> np.ndarray:
        if max_time_order < 3:
            raise ValueError("force jacobian requires max_time_order >= 3")
        if "mat_joint_force" not in cache:
            cache["mat_joint_force"] = _selected_coord_to_joint_force_grad_mat(
                robot, state, joints, force_order=max_time_order - 2, dim=dim
            )
        return cache["mat_joint_force"]

    def get_mat_joint_torque() -> np.ndarray:
        if max_time_order < 3:
            raise ValueError("torque jacobian requires max_time_order >= 3")
        if "mat_joint_torque" not in cache:
            cache["mat_joint_torque"] = _selected_coord_to_joint_torque_grad_mat(
                robot, state, joints, torque_order=max_time_order - 2, dim=dim
            )
        return cache["mat_joint_torque"]

    jacob_list = []
    for st in state_type_list:
        joint = robot.joint(st.owner_name)
        if joint is None:
            raise ValueError(f"Invalid joint name: {st.owner_name}")

        joint_id = joint_index[joint.name]
        order = st.key_order - 1

        if st.data_type in keys_momentum:
            base = joint_id * dof * (max_time_order - 1)
            if st.frame_name == "world":
                jacob_part = get_mat_joint_wmom()[..., base + dof*order : base + dof*(order+1), :]
            else:
                jacob_part = get_mat_joint_mom()[..., base + dof*order : base + dof*(order+1), :]
        elif st.data_type in keys_force:
            base = joint_id * dof * (max_time_order - 2)
            jacob_part = get_mat_joint_force()[..., base + dof*order : base + dof*(order+1), :]
        elif st.data_type in keys_torque:
            base = torque_offset[joint.name]
            jacob_part = get_mat_joint_torque()[..., base + joint.dof*order : base + joint.dof*(order+1), :]
        else:
            raise ValueError("joint-only fast path supports only momentum, force, and torque states")

        jacob_list.append(jacob_part)

    if list_output:
        return jacob_list
    return np.concatenate(jacob_list, axis=-2)

def outward_jacobian(robot : RobotStruct, state : dict, state_type_list : list[StateType], max_time_order = None, dim : int = 3, list_output : bool = False) -> np.ndarray:
    if StateType.is_list_all_in_kinematics(state_type_list):
        return outward_kinematics_jacobian(robot, state, state_type_list, max_time_order, dim=dim, list_output=list_output)
    
    if max_time_order is None:
        max_time_order = StateType.max_time_order(state_type_list)

    if all(st.owner_type == "link" for st in state_type_list):
        return _outward_link_only_jacobian(robot, state, state_type_list, max_time_order, dim=dim, list_output=list_output)
    if all(st.owner_type == "joint" for st in state_type_list):
        return _outward_joint_only_jacobian(robot, state, state_type_list, max_time_order, dim=dim, list_output=list_output)

    dof = dim_to_dof(dim)
    force_order = max_time_order - 2
    is_batched = _is_batched_kinematics_state(
        robot,
        state,
        [StateType("link", robot.links[0].name, "vel")],
        max_time_order,
    )
    cache = {}

    def get_mat_kine() -> np.ndarray:
        if "mat_kine" not in cache:
            if is_batched:
                cache["mat_kine"] = _batch_selected_coord_to_link_vel_grad_mat(robot, state, robot.links, order=max_time_order, dim=dim)
            else:
                cache["mat_kine"] = total_coord_to_link_vel_grad_mat(robot, state, order=max_time_order, dim=dim)
        return cache["mat_kine"]

    def get_mat_tan_kine() -> np.ndarray:
        if "mat_tan_kine" not in cache:
            if is_batched:
                cache["mat_tan_kine"] = _batch_selected_coord_to_link_tan_vel_grad_mat(
                    robot, state, robot.links, out_order=max_time_order-1, in_order=max_time_order, dim=dim
                )
            else:
                cache["mat_tan_kine"] = total_coord_to_link_tan_vel_grad_mat(
                    robot, state, out_order=max_time_order-1, in_order=max_time_order, dim=dim
                )
        return cache["mat_tan_kine"]

    def get_mat_link_mom() -> np.ndarray:
        if "mat_link_mom" not in cache:
            if is_batched:
                cache["mat_link_mom"] = _batch_selected_coord_to_link_momentum_grad_mat(
                    robot, state, robot.links, order=max_time_order, dim=dim
                )
            else:
                cache["mat_link_mom"] = total_coord_to_link_momentum_grad_mat(robot, state, order=max_time_order, dim=dim)
        return cache["mat_link_mom"]

    def get_mat_link_wmom() -> np.ndarray:
        if "mat_link_wmom" not in cache:
            if is_batched:
                cache["mat_link_wmom"] = _batch_selected_coord_to_world_link_momentum_grad_mat(
                    robot, state, robot.links, order=max_time_order, dim=dim
                )
            else:
                cache["mat_link_wmom"] = (
                    total_partial_link_momentum_to_world_link_momentum_grad_mat(robot, state, order=max_time_order, dim=dim) @ get_mat_link_mom()
                    + total_partial_link_tan_vel_to_world_link_momentum_grad_mat(robot, state, order=max_time_order, dim=dim) @ get_mat_tan_kine()
                )
        return cache["mat_link_wmom"]

    def get_mat_joint_wmom() -> np.ndarray:
        if "mat_joint_wmom" not in cache:
            cache["mat_joint_wmom"] = total_world_link_wrench_to_world_joint_wrench_mat(
                robot, order=max_time_order-1, dim=dim
            ) @ get_mat_link_wmom()
        return cache["mat_joint_wmom"]

    def get_mat_joint_mom() -> np.ndarray:
        if "mat_joint_mom" not in cache:
            if is_batched:
                cache["mat_joint_mom"] = _batch_selected_coord_to_joint_momentum_grad_mat(
                    robot, state, robot.joints, order=max_time_order, dim=dim
                )
            else:
                child_link_tan = take_joint_child_link_blocks(
                    get_mat_tan_kine(),
                    robot,
                    dof * (max_time_order - 1),
                    axis=-2,
                )
                cache["mat_joint_mom"] = (
                    total_partial_world_joint_momentum_to_joint_momentum_grad_mat(robot, state, max_time_order, dim) @ get_mat_joint_wmom()
                    + total_partial_link_tan_vel_to_joint_momentum_grad_mat(robot, state, max_time_order, dim)
                    @ child_link_tan
                )
        return cache["mat_joint_mom"]

    def get_partial_mom_to_force() -> np.ndarray:
        if "partial_mom_to_force" not in cache:
            if is_batched:
                cache["partial_mom_to_force"] = _batch_total_partial_momentum_to_force_grad_mat(
                    robot, state, force_order=force_order, dim=dim
                )
            else:
                cache["partial_mom_to_force"] = total_partial_momentum_to_force_grad_mat(
                    robot, state, force_order=force_order, dim=dim
                )
        return cache["partial_mom_to_force"]

    def get_mat_link_force() -> np.ndarray:
        if max_time_order < 3:
            raise ValueError("force jacobian requires max_time_order >= 3")
        if "mat_link_force" not in cache:
            if is_batched:
                cache["mat_link_force"] = _batch_total_coord_to_link_force_grad_mat(
                    robot, state, force_order=force_order, dim=dim
                )
            else:
                cache["mat_link_force"] = (
                    get_partial_mom_to_force() @ get_mat_link_mom()
                    + total_partial_link_sp_vel_to_link_force_grad_mat(robot, state, force_order=force_order, dim=dim) @ get_mat_kine()
                )
        return cache["mat_link_force"]

    def get_mat_joint_force() -> np.ndarray:
        if max_time_order < 3:
            raise ValueError("force jacobian requires max_time_order >= 3")
        if "mat_joint_force" not in cache:
            if is_batched:
                cache["mat_joint_force"] = _batch_total_coord_to_joint_force_grad_mat(
                    robot, state, force_order=force_order, dim=dim
                )
            else:
                child_partial_momentum = take_joint_child_link_matrix_blocks(
                    get_partial_mom_to_force(),
                    robot,
                    dof * (max_time_order - 2),
                    dof * (max_time_order - 1),
                )
                child_link_kine = take_joint_child_link_blocks(
                    get_mat_kine(),
                    robot,
                    dof * max_time_order,
                    axis=-2,
                )
                cache["mat_joint_force"] = (
                    child_partial_momentum @ get_mat_joint_mom()
                    + total_partial_link_sp_vel_to_joint_force_grad_mat(robot, state, force_order=force_order, dim=dim)
                    @ child_link_kine
                )
        return cache["mat_joint_force"]

    def get_mat_joint_torque() -> np.ndarray:
        if max_time_order < 3:
            raise ValueError("torque jacobian requires max_time_order >= 3")
        if "mat_joint_torque" not in cache:
            cache["mat_joint_torque"] = total_joint_wrench_to_joint_torque_mat(
                robot, torque_order=force_order, dim=dim
            ) @ get_mat_joint_force()
        return cache["mat_joint_torque"]

    jacob_list = []
    for st in state_type_list:
        if st.owner_type == "link":
            link = robot.link(st.owner_name)
            if link is None:
                raise ValueError(f"Invalid link name: {st.owner_name}")
        elif st.owner_type == "joint":
            joint = robot.joint(st.owner_name)
            if joint is None:
                raise ValueError(f"Invalid joint name: {st.owner_name}")
            
        order = st.key_order -1

        if st.data_type in keys_kinematics:
            base = link.id * dof * max_time_order
            jacob_part = get_mat_kine()[..., base + dof*order : base + dof*(order+1), :]
            jacob_list.append(jacob_part)
        elif st.data_type in keys_momentum:
            if st.owner_type == "link":
                if st.frame_name == "world":
                    base = link.id * dof * (max_time_order-1)
                    jacob_part = get_mat_link_wmom()[..., base + dof*(order) : base + dof*(order+1), :]
                else:
                    base = link.id * dof * (max_time_order-1)
                    jacob_part = get_mat_link_mom()[..., base + dof*(order) : base + dof*(order+1), :]
            elif st.owner_type == "joint":
                if st.frame_name == "world":
                    base = joint.id * dof * (max_time_order-1)
                    jacob_part = get_mat_joint_wmom()[..., base + dof*(order) : base + dof*(order+1), :]
                else:
                    base = joint.id * dof * (max_time_order-1)
                    jacob_part = get_mat_joint_mom()[..., base + dof*(order) : base + dof*(order+1), :]
            jacob_list.append(jacob_part)
        elif st.data_type in keys_force:
            if st.owner_type == "link":
                base = link.id * dof * (max_time_order-2)
                jacob_part = get_mat_link_force()[..., base + dof*order : base + dof*(order+1), :]
            elif st.owner_type == "joint":
                base = joint.id * dof * (max_time_order-2)
                jacob_part = get_mat_joint_force()[..., base + dof*order : base + dof*(order+1), :]
            jacob_list.append(jacob_part)
        elif st.data_type in keys_torque:
            if st.owner_type == "joint":
                base = joint.dof_index * (max_time_order-2)
                jacob_part = get_mat_joint_torque()[..., base + joint.dof*(order) : base + joint.dof*(order+1), :]
            else:
                raise ValueError("torque can be specified only for joint owner type")
            jacob_list.append(jacob_part)

    if list_output:
        return jacob_list
    else:
        return np.concatenate(jacob_list, axis=-2)


def outward_jacobian_matvec(robot : RobotStruct, state : dict, state_type_list : list[StateType], vec : np.ndarray, max_time_order = None, dim : int = 3, list_output : bool = False) -> np.ndarray:
    if StateType.is_list_all_in_kinematics(state_type_list):
        return outward_kinematics_jacobian_matvec(robot, state, state_type_list, vec, max_time_order, dim=dim, list_output=list_output)

    if max_time_order is None:
        max_time_order = StateType.max_time_order(state_type_list)

    dof = dim_to_dof(dim)
    if np.asarray(vec).ndim > 1 or _is_batched_kinematics_state(
        robot,
        state,
        [StateType("link", robot.links[0].name, "vel")],
        max_time_order,
    ):
        return _batch_outward_dynamics_jacobian_matvec(
            robot,
            state,
            state_type_list,
            vec,
            max_time_order=max_time_order,
            dim=dim,
            list_output=list_output,
        )

    vec_kine = total_coord_to_link_vel_grad_matvec(robot, state, vec, order=max_time_order, dim=dim)
    vec_tan_kine = total_coord_to_link_tan_vel_grad_matvec(robot, state, vec, out_order=max_time_order-1, in_order=max_time_order, dim=dim)
    vec_link_mom = total_coord_to_link_momentum_grad_matvec(robot, state, vec, order=max_time_order, dim=dim)

    vec_link_wmom = total_partial_link_momentum_to_world_link_momentum_grad_matvec(
        robot, state, vec_link_mom, order=max_time_order, dim=dim
    ) + total_partial_link_tan_vel_to_world_link_momentum_grad_matvec(
        robot, state, vec_tan_kine, order=max_time_order, dim=dim
    )

    vec_joint_wmom = total_world_link_wrench_to_world_joint_wrench_matvec(
        robot, vec_link_wmom, order=max_time_order-1, dim=dim
    )

    child_vec_tan_kine = take_joint_child_link_blocks(
        vec_tan_kine, robot, (max_time_order - 1) * dof
    )
    vec_joint_mom = total_partial_world_joint_momentum_to_joint_momentum_grad_matvec(
        robot, state, vec_joint_wmom, max_time_order, dim
    ) + total_partial_link_tan_vel_to_joint_momentum_grad_matvec(
        robot, state, child_vec_tan_kine, max_time_order, dim
    )

    if max_time_order >= 3:
        mat_mom_to_force = total_partial_momentum_to_force_grad_mat(robot, state, force_order=max_time_order-2, dim=dim)
        vec_link_force = total_partial_momentum_to_force_grad_matvec(
            robot, state, vec_link_mom, force_order=max_time_order-2, dim=dim
        ) + total_partial_link_sp_vel_to_link_force_grad_matvec(
            robot, state, vec_kine, force_order=max_time_order-2, dim=dim
        )

        mat_joint_mom_to_force = take_joint_child_link_matrix_blocks(
            mat_mom_to_force,
            robot,
            dof * (max_time_order - 2),
            dof * (max_time_order - 1),
        )
        child_vec_kine = take_joint_child_link_blocks(
            vec_kine, robot, max_time_order * dof
        )
        vec_joint_force = mat_joint_mom_to_force @ vec_joint_mom \
                    + total_partial_link_sp_vel_to_joint_force_grad_matvec(
                        robot, state, child_vec_kine, force_order=max_time_order-2, dim=dim
                    )

        vec_joint_torque = total_joint_wrench_to_joint_torque_matvec(
            robot, vec_joint_force, torque_order=max_time_order-2, dim=dim
        )

    vec_list = []
    for st in state_type_list:
        if st.owner_type == "link":
            link = robot.link(st.owner_name)
            if link is None:
                raise ValueError(f"Invalid link name: {st.owner_name}")
        elif st.owner_type == "joint":
            joint = robot.joint(st.owner_name)
            if joint is None:
                raise ValueError(f"Invalid joint name: {st.owner_name}")

        order = st.key_order -1

        if st.data_type in keys_kinematics:
            base = link.id * dof * max_time_order
            vec_part = vec_kine[base + dof*order : base + dof*(order+1)]
            vec_list.append(vec_part)
        elif st.data_type in keys_momentum:
            if st.owner_type == "link":
                base = link.id * dof * (max_time_order-1)
                if st.frame_name == "world":
                    vec_part = vec_link_wmom[base + dof*(order) : base + dof*(order+1)]
                else:
                    vec_part = vec_link_mom[base + dof*(order) : base + dof*(order+1)]
            elif st.owner_type == "joint":
                base = joint.id * dof * (max_time_order-1)
                if st.frame_name == "world":
                    vec_part = vec_joint_wmom[base + dof*(order) : base + dof*(order+1)]
                else:
                    vec_part = vec_joint_mom[base + dof*(order) : base + dof*(order+1)]
            vec_list.append(vec_part)
        elif st.data_type in keys_force:
            if st.owner_type == "link":
                base = link.id * dof * (max_time_order-2)
                vec_part = vec_link_force[base + dof*order : base + dof*(order+1)]
            elif st.owner_type == "joint":
                base = joint.id * dof * (max_time_order-2)
                vec_part = vec_joint_force[base + dof*order : base + dof*(order+1)]
            vec_list.append(vec_part)
        elif st.data_type in keys_torque:
            if st.owner_type == "joint":
                base = joint.dof_index * (max_time_order-2)
                vec_part = vec_joint_torque[base + joint.dof*(order) : base + joint.dof*(order+1)]
            else:
                raise ValueError("torque can be specified only for joint owner type")
            vec_list.append(vec_part)

    if list_output:
        return vec_list
    else:
        return np.concatenate(vec_list)


def outward_jacobian_matmul_rhs(robot : RobotStruct, state : dict, state_type_list : list[StateType], rhs : np.ndarray, max_time_order = None, dim : int = 3, list_output : bool = False) -> np.ndarray:
    if StateType.is_list_all_in_kinematics(state_type_list):
        return outward_kinematics_jacobian_matmul_rhs(robot, state, state_type_list, rhs, max_time_order, dim=dim, list_output=list_output)

    if max_time_order is None:
        max_time_order = StateType.max_time_order(state_type_list)

    return _batch_outward_dynamics_jacobian_matmul_rhs(
        robot,
        state,
        state_type_list,
        rhs,
        max_time_order,
        dim=dim,
        list_output=list_output,
    )
