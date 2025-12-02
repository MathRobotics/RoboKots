#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025. 6.20 Created by T.Ishigaki
# outward computation module from motion and robot_model to state by matrix formulation

import numpy as np

from mathrobo import numerical_grad

from ..basic.robot import RobotStruct
from ..basic.state import keys_order, data_type_dof, keys_time_order, StateType, dim_to_dof
from ..basic.state import keys_kinematics, keys_momentum, keys_force, keys_torque
from ..basic.state_dict import extract_dict_link_info
from ..basic.motion import RobotMotions

from ..total.total_kinematics_grad_mat import total_coord_to_link_vel_grad_mat
from ..total.total_dynamics_grad_mat import total_coord_to_link_momentum_grad_mat, total_coord_to_joint_momentum_grad_mat
from ..total.total_dynamics_grad_mat import total_coord_to_world_link_momentum_grad_mat, total_coord_to_world_joint_momentum_grad_mat
from ..total.total_dynamics_grad_mat import total_coord_to_link_force_grad_mat, total_coord_to_joint_force_grad_mat, total_coord_to_joint_torque_grad_mat

from .outward import dynamics_cmtm as outward_dynamics
from .outward_state import outward_state, outward_state_dof

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

def outward_kinematics_jacobian(robot : RobotStruct, state : dict, state_type_list : list[StateType], max_time_order = None, dim : int = 3) -> np.ndarray:
    kine_state_type_list = StateType.filter_list_by_kinematics(state_type_list)
    if max_time_order is None:
        max_time_order = StateType.max_time_order(kine_state_type_list)
    mat = total_coord_to_link_vel_grad_mat(robot, state, order=max_time_order, dim=dim)
    jacobs = np.empty((0, robot.dof * max_time_order))
    dof = dim_to_dof(dim)
    for st in kine_state_type_list:
        link = robot.link(st.owner_name)
        if link is None:
            raise ValueError(f"Invalid link name: {st.owner_name}")
        base = link.id * dof * max_time_order
        jacob_part = mat[base + dof*(st.time_order-1) : base + dof*st.time_order, :]
        jacobs =  np.vstack((jacobs, jacob_part))

    return jacobs

def outward_jacobian(robot : RobotStruct, state : dict, state_type_list : list[StateType], max_time_order = None, dim : int = 3) -> np.ndarray:
    if StateType.is_list_all_in_kinematics(state_type_list):
        return outward_kinematics_jacobian(robot, state, state_type_list, max_time_order, dim=dim)
    
    if max_time_order is None:
        max_time_order = StateType.max_time_order(state_type_list)

    mat_kine = total_coord_to_link_vel_grad_mat(robot, state, order=max_time_order, dim=dim)
    mat_link_mom = total_coord_to_link_momentum_grad_mat(robot, state, order=max_time_order, dim=dim)
    mat_joint_mom = total_coord_to_joint_momentum_grad_mat(robot, state, order=max_time_order, dim=dim)
    if max_time_order >=3:
        mat_link_force = total_coord_to_link_force_grad_mat(robot, state, force_order=max_time_order-2, dim=dim)
        mat_joint_force = total_coord_to_joint_force_grad_mat(robot, state, force_order=max_time_order-2, dim=dim)
        mat_joint_torque = total_coord_to_joint_torque_grad_mat(robot, state, torque_order=max_time_order-2, dim=dim)
    dof = dim_to_dof(dim)

    jacobs = np.empty((0, robot.dof * max_time_order))
    for st in state_type_list:
        if st.owner_type == "link":
            link = robot.link(st.owner_name)
            if link is None:
                raise ValueError(f"Invalid link name: {st.owner_name}")
        elif st.owner_type == "joint":
            joint = robot.joint(st.owner_name)
            if joint is None:
                raise ValueError(f"Invalid joint name: {st.owner_name}")
            
        order = st.key_order

        if st.data_type in keys_kinematics:
            base = link.id * dof * max_time_order
            jacob_part = mat_kine[base + dof*order : base + dof*(order+1), :]
            jacobs = np.vstack((jacobs, jacob_part))
        elif st.data_type in keys_momentum:
            if st.owner_type == "link":
                base = link.id * dof * (max_time_order-1)
                jacob_part = mat_link_mom[base + dof*(order) : base + dof*(order+1), :]
            elif st.owner_type == "joint":
                base = joint.id * dof * (max_time_order-1)
                jacob_part = mat_joint_mom[base + dof*(order) : base + dof*(order+1), :]
            jacobs = np.vstack((jacobs, jacob_part))
        elif st.data_type in keys_force:
            if st.owner_type == "link":
                base = link.id * dof * (max_time_order-2)
                jacob_part = mat_link_force[base + dof*order : base + dof*(order+1), :]
            elif st.owner_type == "joint":
                base = joint.id * dof * (max_time_order-2)
                jacob_part = mat_joint_force[base + dof*order : base + dof*(order+1), :]
            jacobs = np.vstack((jacobs, jacob_part))
        elif st.data_type in keys_torque:
            if st.owner_type == "joint":
                base = joint.dof_index * (max_time_order-2)
                jacob_part = mat_joint_torque[base + joint.dof*(order) : base + joint.dof*(order+1), :]
            else:
                raise ValueError("torque can be specified only for joint owner type")
            jacobs = np.vstack((jacobs, jacob_part))
    return jacobs

def dynamics_jacobian_numerical(robot : RobotStruct, motions : RobotMotions, link_name_list : list[str], data_type, owner_type, frame_name : str = None, output_order_ : int = 1) -> np.ndarray:
    order = keys_time_order[data_type] + output_order_ - 1
    dynamics_time_order = keys_time_order[data_type] + output_order_ - keys_time_order["force"]

    if dynamics_time_order < 1:
        dynamics_time_order = 0
    d = outward_state_dof(robot, StateType(owner_type, link_name_list[0], data_type, frame_name), dim = 3)
    dof = d * output_order_
    
    jacobs = np.zeros((dof*len(link_name_list),robot.dof*order))
    motion = np.zeros(robot.dof * order)

    for joint in robot.joints:
        m = motions.joint_motions(joint.dof, joint.dof_index, order)
        motion[joint.dof_index*order:joint.dof_index*order+joint.dof*order] = m.flatten()
    
    for link in robot.links:
        m = motions.link_motions(link.dof, link.dof_index, order)
        motion[link.dof_index*order:link.dof_index*order+link.dof*order] = m.flatten()

    for i in range(len(link_name_list)):
        def dynamics_func(x):
            state = outward_dynamics(robot, x, dynamics_order=dynamics_time_order)
            y = np.zeros(dof)
            for j in range(output_order_):
                if j == 0:
                    y[d*j:d*(j+1)] = outward_state(robot, state, StateType(owner_type, link_name_list[i], data_type, frame_name))
                else:
                    y[d*j:d*(j+1)] = outward_state(robot, state, StateType(owner_type, link_name_list[i], data_type+"_diff"+str(j), frame_name))
            return y

        jacobs[dof*i:dof*(i+1)] = numerical_grad(motion, dynamics_func)

    return jacobs