#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025. 6.20 Created by T.Ishigaki
# outward computation module from motion and robot_model to state by matrix formulation

import numpy as np

from robokots.core import RobotStruct
from robokots.core.state import StateType, dim_to_dof
from robokots.core.state import keys_kinematics, keys_momentum, keys_force, keys_torque

from robokots.core.models.whole_body.total_kinematics_grad_mat import total_coord_to_link_vel_grad_mat
from robokots.core.models.whole_body.total_dynamics_grad_mat import total_coord_to_link_momentum_grad_mat, total_coord_to_joint_momentum_grad_mat
from robokots.core.models.whole_body.total_dynamics_grad_mat import total_coord_to_world_link_momentum_grad_mat, total_coord_to_world_joint_momentum_grad_mat
from robokots.core.models.whole_body.total_dynamics_grad_mat import total_coord_to_link_force_grad_mat, total_coord_to_joint_force_grad_mat, total_coord_to_joint_torque_grad_mat

def link_jacobian(robot : RobotStruct, state : dict, link_name_list : list[str], order : int = 3, dim : int = 3) -> np.ndarray:
    links = robot.link_list(link_name_list)
    mat = total_coord_to_link_vel_grad_mat(robot, state, order=order, dim=dim)
    dof = dim_to_dof(dim)
    jacobs = np.zeros((dof * order * len(links), robot.dof * order))
    
    for i, link in enumerate(links):
        jacobs[i*dof*order:(i+1)*dof*order, :] = mat[link.id*dof*order:(link.id+1)*dof*order, :]
    return jacobs

def link_momentum_jacobian(robot : RobotStruct, state : dict, link_name_list : list[str], momentum_order : int = 1, dim : int = 3) -> np.ndarray:
    links = robot.link_list(link_name_list)
    mat = total_coord_to_link_momentum_grad_mat(robot, state, order=momentum_order+1, dim=dim)
    dof = dim_to_dof(dim)
    jacobs = np.zeros((dof * momentum_order * len(links), robot.dof * (momentum_order+1)))

    for i, link in enumerate(links):
        jacobs[i*dof*momentum_order:(i+1)*dof*momentum_order, :] = mat[link.id*dof*momentum_order:(link.id+1)*dof*momentum_order, :]
    return jacobs

def world_link_momentum_jacobian(robot : RobotStruct, state : dict, link_name_list : list[str], momentum_order : int = 1, dim : int = 3) -> np.ndarray:
    links = robot.link_list(link_name_list)
    mat = total_coord_to_world_link_momentum_grad_mat(robot, state, order=momentum_order+1, dim=dim)
    dof = dim_to_dof(dim)
    jacobs = np.zeros((dof * (momentum_order) * len(links), robot.dof * (momentum_order+1)))

    for i, link in enumerate(links):
        jacobs[i*dof*momentum_order:(i+1)*dof*momentum_order, :] = mat[link.id*dof*momentum_order:(link.id+1)*dof*momentum_order, :]
    return jacobs

def world_joint_momentum_jacobian(robot : RobotStruct, state : dict, joint_name_list : list[str], momentum_order : int = 1, dim : int = 3) -> np.ndarray:
    joints = robot.joint_list(joint_name_list)
    if joints == [None]:
        raise ValueError("joint_name_list contains invalid joint name")
    mat = total_coord_to_world_joint_momentum_grad_mat(robot, state, order=momentum_order+1, dim=dim)
    dof = dim_to_dof(dim)
    jacobs = np.zeros((dof * (momentum_order) * len(joints), robot.dof * (momentum_order+1)))

    for i, joint in enumerate(joints):
        jacobs[i*dof*(momentum_order):(i+1)*dof*(momentum_order), :] = mat[joint.id*dof*(momentum_order):(joint.id+1)*dof*(momentum_order), :]
    return jacobs

def joint_momentum_jacobian(robot : RobotStruct, state : dict, joint_name_list : list[str], momentum_order : int = 1, dim : int = 3) -> np.ndarray:
    joints = robot.joint_list(joint_name_list)
    if joints == [None]:
        raise ValueError("joint_name_list contains invalid joint name")
    mat = total_coord_to_joint_momentum_grad_mat(robot, state, order=momentum_order+1, dim=dim)
    dof = dim_to_dof(dim)
    jacobs = np.zeros((dof * (momentum_order) * len(joints), robot.dof * (momentum_order+1)))

    for i, joint in enumerate(joints):
        jacobs[i*dof*(momentum_order):(i+1)*dof*(momentum_order), :] = mat[joint.id*dof*(momentum_order):(joint.id+1)*dof*(momentum_order), :]
    return jacobs

def link_force_jacobian(robot : RobotStruct, state : dict, link_name_list : list[str], force_order : int = 1, dim : int = 3) -> np.ndarray:
    links = robot.link_list(link_name_list)
    mat = total_coord_to_link_force_grad_mat(robot, state, force_order=force_order, dim=dim)
    dof = dim_to_dof(dim)
    jacobs = np.zeros((dof * force_order * len(links), robot.dof * (force_order+2)))

    for i, link in enumerate(links):
        jacobs[i*dof*force_order:(i+1)*dof*force_order, :] = mat[link.id*dof*force_order:(link.id+1)*dof*force_order, :]
    return jacobs

def joint_force_jacobian(robot : RobotStruct, state : dict, joint_name_list : list[str], force_order : int = 1, dim : int = 3) -> np.ndarray:
    joints = robot.joint_list(joint_name_list)
    mat = total_coord_to_joint_force_grad_mat(robot, state, force_order=force_order, dim=dim)
    dof = dim_to_dof(dim)
    jacobs = np.zeros((dof * force_order * len(joints), robot.dof * (force_order+2)))

    for i, joint in enumerate(joints):
        jacobs[i*dof*force_order:(i+1)*dof*force_order, :] = mat[joint.id*dof*force_order:(joint.id+1)*dof*force_order, :]
    return jacobs

def joint_torque_jacobian(robot : RobotStruct, state : dict, joint_name_list : list[str], torque_order : int = 1, dim : int = 3) -> np.ndarray:
    joints = robot.joint_list(joint_name_list)
    mat = total_coord_to_joint_torque_grad_mat(robot, state, torque_order=torque_order, dim=dim)
    jacobs = np.empty((0, robot.dof * (torque_order+2)))
    
    for joint in joints:
        jacobs = np.vstack((jacobs, mat[joint.dof_index*torque_order:(joint.dof_index+joint.dof)*torque_order, :]))

    return jacobs

def outward_kinematics_jacobian(robot : RobotStruct, state : dict, state_type_list : list[StateType], max_time_order = None, dim : int = 3, list_output : bool = False) -> np.ndarray:
    kine_state_type_list = StateType.filter_list_by_kinematics(state_type_list)
    if max_time_order is None:
        max_time_order = StateType.max_time_order(kine_state_type_list)
    mat = total_coord_to_link_vel_grad_mat(robot, state, order=max_time_order, dim=dim)
    dof = dim_to_dof(dim)

    jacob_list = []
    for st in kine_state_type_list:
        link = robot.link(st.owner_name)
        if link is None:
            raise ValueError(f"Invalid link name: {st.owner_name}")
        base = link.id * dof * max_time_order
        jacob_part = mat[base + dof*(st.time_order-1) : base + dof*st.time_order, :]
        jacob_list.append(jacob_part)

    if list_output:
        return jacob_list
    else:
        return np.vstack(jacob_list)

from robokots.core.models.whole_body.total_kinematics_grad_mat import total_coord_to_link_tan_vel_grad_mat
from robokots.core.models.whole_body.total_partial_grad_mat import *

def outward_jacobian(robot : RobotStruct, state : dict, state_type_list : list[StateType], max_time_order = None, dim : int = 3, list_output : bool = False) -> np.ndarray:
    if StateType.is_list_all_in_kinematics(state_type_list):
        return outward_kinematics_jacobian(robot, state, state_type_list, max_time_order, dim=dim, list_output=list_output)
    
    if max_time_order is None:
        max_time_order = StateType.max_time_order(state_type_list)

    dof = dim_to_dof(dim)

    mat_kine = total_coord_to_link_vel_grad_mat(robot, state, order=max_time_order, dim=dim)
    mat_tan_kine = total_coord_to_link_tan_vel_grad_mat(robot, state, out_order=max_time_order-1, in_order=max_time_order, dim=dim)
    
    mat_link_mom = total_coord_to_link_momentum_grad_mat(robot, state, order=max_time_order, dim=dim)
    
    mat_link_wmom = total_partial_link_momentum_to_world_link_momentum_grad_mat(robot, state, order=max_time_order, dim=dim) @ mat_link_mom \
        + total_partial_link_tan_vel_to_world_link_momentum_grad_mat(robot, state, order=max_time_order, dim=dim) @ mat_tan_kine
    
    mat_joint_wmom = total_world_link_wrench_to_world_joint_wrench_mat(robot, order=max_time_order-1, dim=dim) @ mat_link_wmom

    mat_joint_mom = total_partial_world_joint_momentum_to_joint_momentum_grad_mat(robot, state, max_time_order, dim) @ mat_joint_wmom \
                + total_partial_link_tan_vel_to_joint_momentum_grad_mat(robot, state, max_time_order, dim) @ mat_tan_kine[(max_time_order-1)*dof:]    

    if max_time_order >= 3:
        mat_link_force = total_partial_momentum_to_force_grad_mat(robot, state, force_order=max_time_order-2, dim=dim) @ mat_link_mom \
                + total_partial_link_sp_vel_to_link_force_grad_mat(robot, state, force_order=max_time_order-2, dim=dim) @ mat_kine

        mat_joint_force = total_partial_momentum_to_force_grad_mat(robot, state, force_order=max_time_order-2, dim=dim)[(max_time_order-2)*dof:,(max_time_order-1)*dof:] @ mat_joint_mom \
                    + total_partial_link_sp_vel_to_joint_force_grad_mat(robot, state, force_order=max_time_order-2, dim=dim) @ mat_kine[(max_time_order)*dof:]

        mat_joint_torque = total_joint_wrench_to_joint_torque_mat(robot, torque_order=max_time_order-2, dim=dim) @ mat_joint_force

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
            jacob_part = mat_kine[base + dof*order : base + dof*(order+1), :]
            jacob_list.append(jacob_part)
        elif st.data_type in keys_momentum:
            if st.owner_type == "link":
                if st.frame_name == "world":
                    base = link.id * dof * (max_time_order-1)
                    jacob_part = mat_link_wmom[base + dof*(order) : base + dof*(order+1), :]
                else:
                    base = link.id * dof * (max_time_order-1)
                    jacob_part = mat_link_mom[base + dof*(order) : base + dof*(order+1), :]
            elif st.owner_type == "joint":
                if st.frame_name == "world":
                    base = joint.id * dof * (max_time_order-1)
                    jacob_part = mat_joint_wmom[base + dof*(order) : base + dof*(order+1), :]
                else:
                    base = joint.id * dof * (max_time_order-1)
                    jacob_part = mat_joint_mom[base + dof*(order) : base + dof*(order+1), :]
            jacob_list.append(jacob_part)
        elif st.data_type in keys_force:
            if st.owner_type == "link":
                base = link.id * dof * (max_time_order-2)
                jacob_part = mat_link_force[base + dof*order : base + dof*(order+1), :]
            elif st.owner_type == "joint":
                base = joint.id * dof * (max_time_order-2)
                jacob_part = mat_joint_force[base + dof*order : base + dof*(order+1), :]
            jacob_list.append(jacob_part)
        elif st.data_type in keys_torque:
            if st.owner_type == "joint":
                base = joint.dof_index * (max_time_order-2)
                jacob_part = mat_joint_torque[base + joint.dof*(order) : base + joint.dof*(order+1), :]
            else:
                raise ValueError("torque can be specified only for joint owner type")
            jacob_list.append(jacob_part)

    if list_output:
        return jacob_list
    else:
        return np.vstack(jacob_list)