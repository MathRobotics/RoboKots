#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025. 6.20 Created by T.Ishigaki
# outward computation module from motion and robot_model to state by matrix formulation

import numpy as np
from mathrobo import CMVector, Factorial

from robokots.core import RobotStruct
from robokots.core.state import StateType, dim_to_dof, data_type_dof, data_type_offset
from robokots.core.state import keys_kinematics, keys_momentum, keys_force, keys_torque
from robokots.core.state_dict import state_dict_to_cmtm, state_dict_to_cmtm_wrench, state_dict_to_cmvec, state_dict_to_rel_cmtm

from robokots.core.models.whole_body.total_kinematics_grad_mat import total_coord_to_joint_tan_vel_grad_mat, total_coord_to_link_vel_grad_mat
from robokots.core.models.whole_body.total_kinematics_mat import total_coord_arrange
from robokots.core.models.whole_body.total_dynamics_grad_mat import total_coord_to_link_momentum_grad_mat, total_coord_to_joint_momentum_grad_mat
from robokots.core.models.whole_body.total_dynamics_grad_mat import total_coord_to_world_link_momentum_grad_mat, total_coord_to_world_joint_momentum_grad_mat
from robokots.core.models.whole_body.total_dynamics_grad_mat import total_coord_to_link_force_grad_mat, total_coord_to_joint_force_grad_mat, total_coord_to_joint_torque_grad_mat
from robokots.core.models.dynamics.base import spatial_inertia
from robokots.core.models.dynamics.dynamics_matrix import (
    inertia_diag_mat,
    partial_link_sp_vel_to_force_grad_mat,
    partial_momentum_to_force_grad_mat,
)
from robokots.core.models.kinematics.kinematics_matrix import joint_select_diag_mat


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


def _selected_coord_to_link_tan_vel_grad_mat(
    robot: RobotStruct,
    state: dict,
    links: list,
    out_order: int = 3,
    in_order: int | None = None,
    dim: int = 3,
) -> np.ndarray:
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
    dof = dim_to_dof(dim)
    order = force_order + 2
    n_v = dof * order
    n_m = dof * (order - 1)
    n_f = dof * force_order
    mat_link_mom = _selected_coord_to_link_momentum_grad_mat(robot, state, links, order=order, dim=dim)
    mat_kine = _selected_coord_to_link_vel_grad_mat(robot, state, links, order=order, dim=dim)
    mat = np.zeros((len(links) * n_f, mat_kine.shape[1]))

    for i, link in enumerate(links):
        if link is None:
            raise ValueError("link_name_list contains invalid link name")

        row_f = i * n_f
        row_m = i * n_m
        row_v = i * n_v
        cmtm = state_dict_to_cmtm(state, link.name, "link", force_order + 1)
        p_mom = partial_momentum_to_force_grad_mat(cmtm, force_order=force_order, dim=dim)
        link_momentum = state_dict_to_cmvec(state, link.name, "link", "momentum", force_order)
        p_vel = partial_link_sp_vel_to_force_grad_mat(link_momentum, force_order=force_order, dim=dim)
        mat[row_f:row_f+n_f, :] = p_mom @ mat_link_mom[row_m:row_m+n_m, :] + p_vel @ mat_kine[row_v:row_v+n_v, :]

    return mat


def _selected_coord_to_world_joint_momentum_grad_mat(
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
            (factorial @ cmtm_wrench.mat_adj() @ local_joint_momentum.cm_vec()).reshape(order - 1, -1)
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
    dof = dim_to_dof(dim)
    order = force_order + 2
    n_v = dof * order
    n_m = dof * (order - 1)
    n_f = dof * force_order
    child_links = []
    for joint in joints:
        if joint is None:
            raise ValueError("joint_name_list contains invalid joint name")
        child_links.append(robot.links[joint.child_link_id])

    mat_joint_mom = _selected_coord_to_joint_momentum_grad_mat(robot, state, joints, order=order, dim=dim)
    mat_child_link_vel = _selected_coord_to_link_vel_grad_mat(robot, state, child_links, order=order, dim=dim)
    mat = np.zeros((len(joints) * n_f, mat_child_link_vel.shape[1]))

    for i, joint in enumerate(joints):
        row_f = i * n_f
        row_m = i * n_m
        row_v = i * n_v
        child_link = child_links[i]
        cmtm = state_dict_to_cmtm(state, child_link.name, "link", force_order + 1)
        p_mom = partial_momentum_to_force_grad_mat(cmtm, force_order=force_order, dim=dim)
        joint_momentum = state_dict_to_cmvec(state, joint.name, "joint", "momentum", force_order)
        p_vel = partial_link_sp_vel_to_force_grad_mat(joint_momentum, force_order=force_order, dim=dim)
        mat[row_f:row_f+n_f, :] = (
            p_mom @ mat_joint_mom[row_m:row_m+n_m, :]
            + p_vel @ mat_child_link_vel[row_v:row_v+n_v, :]
        )

    return mat


def _selected_coord_to_joint_torque_grad_mat(
    robot: RobotStruct,
    state: dict,
    joints: list,
    torque_order: int = 1,
    dim: int = 3,
) -> np.ndarray:
    dof = dim_to_dof(dim)
    n_f = dof * torque_order
    mat_joint_force = _selected_coord_to_joint_force_grad_mat(
        robot, state, joints, force_order=torque_order, dim=dim
    )
    mat = np.zeros((sum(joint.dof * torque_order for joint in joints), mat_joint_force.shape[1]))

    row_torque = 0
    for i, joint in enumerate(joints):
        if joint is None:
            raise ValueError("joint_name_list contains invalid joint name")

        row_force = i * n_f
        rows = joint.dof * torque_order
        mat[row_torque:row_torque+rows, :] = (
            joint_select_diag_mat(joint.select_mat, torque_order).T
            @ mat_joint_force[row_force:row_force+n_f, :]
        )
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
        jacob_part = mat[base + offset : base + offset + state_dof, :]
        jacob_list.append(jacob_part)

    if list_output:
        return jacob_list
    else:
        return np.vstack(jacob_list)


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
            jacob_part = get_mat_kine()[base + offset : base + offset + state_dof, :]
        elif st.data_type in keys_momentum:
            base = link_id * dof * (max_time_order-1)
            if st.frame_name == "world":
                jacob_part = get_mat_link_wmom()[base + dof*order : base + dof*(order+1), :]
            else:
                jacob_part = get_mat_link_mom()[base + dof*order : base + dof*(order+1), :]
        elif st.data_type in keys_force:
            base = link_id * dof * (max_time_order-2)
            jacob_part = get_mat_link_force()[base + dof*order : base + dof*(order+1), :]
        else:
            raise ValueError("link-only fast path supports only kinematics, momentum, and force states")

        jacob_list.append(jacob_part)

    if list_output:
        return jacob_list
    return np.vstack(jacob_list)


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
                jacob_part = get_mat_joint_wmom()[base + dof*order : base + dof*(order+1), :]
            else:
                jacob_part = get_mat_joint_mom()[base + dof*order : base + dof*(order+1), :]
        elif st.data_type in keys_force:
            base = joint_id * dof * (max_time_order - 2)
            jacob_part = get_mat_joint_force()[base + dof*order : base + dof*(order+1), :]
        elif st.data_type in keys_torque:
            base = torque_offset[joint.name]
            jacob_part = get_mat_joint_torque()[base + joint.dof*order : base + joint.dof*(order+1), :]
        else:
            raise ValueError("joint-only fast path supports only momentum, force, and torque states")

        jacob_list.append(jacob_part)

    if list_output:
        return jacob_list
    return np.vstack(jacob_list)

from robokots.core.models.whole_body.total_kinematics_grad_mat import total_coord_to_link_tan_vel_grad_mat
from robokots.core.models.whole_body.total_partial_grad_mat import *

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
    cache = {}

    def get_mat_kine() -> np.ndarray:
        if "mat_kine" not in cache:
            cache["mat_kine"] = total_coord_to_link_vel_grad_mat(robot, state, order=max_time_order, dim=dim)
        return cache["mat_kine"]

    def get_mat_tan_kine() -> np.ndarray:
        if "mat_tan_kine" not in cache:
            cache["mat_tan_kine"] = total_coord_to_link_tan_vel_grad_mat(
                robot, state, out_order=max_time_order-1, in_order=max_time_order, dim=dim
            )
        return cache["mat_tan_kine"]

    def get_mat_link_mom() -> np.ndarray:
        if "mat_link_mom" not in cache:
            cache["mat_link_mom"] = total_coord_to_link_momentum_grad_mat(robot, state, order=max_time_order, dim=dim)
        return cache["mat_link_mom"]

    def get_mat_link_wmom() -> np.ndarray:
        if "mat_link_wmom" not in cache:
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
            cache["mat_joint_mom"] = (
                total_partial_world_joint_momentum_to_joint_momentum_grad_mat(robot, state, max_time_order, dim) @ get_mat_joint_wmom()
                + total_partial_link_tan_vel_to_joint_momentum_grad_mat(robot, state, max_time_order, dim)
                @ get_mat_tan_kine()[(max_time_order-1)*dof:]
            )
        return cache["mat_joint_mom"]

    def get_partial_mom_to_force() -> np.ndarray:
        if "partial_mom_to_force" not in cache:
            cache["partial_mom_to_force"] = total_partial_momentum_to_force_grad_mat(
                robot, state, force_order=force_order, dim=dim
            )
        return cache["partial_mom_to_force"]

    def get_mat_link_force() -> np.ndarray:
        if max_time_order < 3:
            raise ValueError("force jacobian requires max_time_order >= 3")
        if "mat_link_force" not in cache:
            cache["mat_link_force"] = (
                get_partial_mom_to_force() @ get_mat_link_mom()
                + total_partial_link_sp_vel_to_link_force_grad_mat(robot, state, force_order=force_order, dim=dim) @ get_mat_kine()
            )
        return cache["mat_link_force"]

    def get_mat_joint_force() -> np.ndarray:
        if max_time_order < 3:
            raise ValueError("force jacobian requires max_time_order >= 3")
        if "mat_joint_force" not in cache:
            cache["mat_joint_force"] = (
                get_partial_mom_to_force()[(max_time_order-2)*dof:,(max_time_order-1)*dof:] @ get_mat_joint_mom()
                + total_partial_link_sp_vel_to_joint_force_grad_mat(robot, state, force_order=force_order, dim=dim)
                @ get_mat_kine()[(max_time_order)*dof:]
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
            jacob_part = get_mat_kine()[base + dof*order : base + dof*(order+1), :]
            jacob_list.append(jacob_part)
        elif st.data_type in keys_momentum:
            if st.owner_type == "link":
                if st.frame_name == "world":
                    base = link.id * dof * (max_time_order-1)
                    jacob_part = get_mat_link_wmom()[base + dof*(order) : base + dof*(order+1), :]
                else:
                    base = link.id * dof * (max_time_order-1)
                    jacob_part = get_mat_link_mom()[base + dof*(order) : base + dof*(order+1), :]
            elif st.owner_type == "joint":
                if st.frame_name == "world":
                    base = joint.id * dof * (max_time_order-1)
                    jacob_part = get_mat_joint_wmom()[base + dof*(order) : base + dof*(order+1), :]
                else:
                    base = joint.id * dof * (max_time_order-1)
                    jacob_part = get_mat_joint_mom()[base + dof*(order) : base + dof*(order+1), :]
            jacob_list.append(jacob_part)
        elif st.data_type in keys_force:
            if st.owner_type == "link":
                base = link.id * dof * (max_time_order-2)
                jacob_part = get_mat_link_force()[base + dof*order : base + dof*(order+1), :]
            elif st.owner_type == "joint":
                base = joint.id * dof * (max_time_order-2)
                jacob_part = get_mat_joint_force()[base + dof*order : base + dof*(order+1), :]
            jacob_list.append(jacob_part)
        elif st.data_type in keys_torque:
            if st.owner_type == "joint":
                base = joint.dof_index * (max_time_order-2)
                jacob_part = get_mat_joint_torque()[base + joint.dof*(order) : base + joint.dof*(order+1), :]
            else:
                raise ValueError("torque can be specified only for joint owner type")
            jacob_list.append(jacob_part)

    if list_output:
        return jacob_list
    else:
        return np.vstack(jacob_list)
