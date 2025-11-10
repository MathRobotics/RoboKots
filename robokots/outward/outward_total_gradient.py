#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025. 6.20 Created by T.Ishigaki
# outward computation module from motion and robot_model to state by matrix formulation

import numpy as np

from mathrobo import numerical_grad

from ..basic.robot import RobotStruct
from ..basic.state import keys_order, data_type_dof, keys_time_order
from ..basic.state_dict import extract_dict_link_info
from ..basic.motion import RobotMotions

from ..total.total_kinematics_grad_mat import total_coord_to_link_vel_grad_mat
from ..total.total_dynamics_grad_mat import total_coord_to_link_momentum_grad_mat, total_coord_to_joint_momentum_grad_mat, total_coord_to_link_force_grad_mat

from .outward import dynamics_cmtm as outward_dynamics

def link_jacobian(robot : RobotStruct, state : dict, link_name_list : list[str], order : int = 3, dim : int = 6) -> np.ndarray:
    links = robot.link_list(link_name_list)
    mat = total_coord_to_link_vel_grad_mat(robot, state, order=order, dim=dim)
    jacobs = np.zeros((dim * order * len(links), robot.dof * order))
    
    for i, link in enumerate(links):
        jacobs[i*dim*order:(i+1)*dim*order, :] = mat[link.id*dim*order:(link.id+1)*dim*order, :]
    return jacobs

def link_momentum_jacobian(robot : RobotStruct, state : dict, link_name_list : list[str], momentum_order : int = 1, dim : int = 6) -> np.ndarray:
    links = robot.link_list(link_name_list)
    mat = total_coord_to_link_momentum_grad_mat(robot, state, order=momentum_order, dim=dim)
    jacobs = np.zeros((dim * (momentum_order-1) * len(links), robot.dof * (momentum_order)))

    for i, link in enumerate(links):
        jacobs[i*dim*(momentum_order-1):(i+1)*dim*(momentum_order-1), :] = mat[link.id*dim*(momentum_order-1):(link.id+1)*dim*(momentum_order-1), :]
    return jacobs

def joint_momentum_jacobian(robot : RobotStruct, state : dict, joint_name_list : list[str], momentum_order : int = 1, dim : int = 6) -> np.ndarray:
    joints = robot.joint_list(joint_name_list)
    if joints == [None]:
        raise ValueError("joint_name_list contains invalid joint name")
    mat = total_coord_to_joint_momentum_grad_mat(robot, state, order=momentum_order-1, dim=dim)
    jacobs = np.zeros((dim * (momentum_order-1) * len(joints), robot.dof * momentum_order))

    for i, joint in enumerate(joints):
        jacobs[i*dim*(momentum_order-1):(i+1)*dim*(momentum_order-1), :] = mat[joint.id*dim*(momentum_order-1):(joint.id+1)*dim*(momentum_order-1), :]
    return jacobs

def link_force_jacobian(robot : RobotStruct, state : dict, link_name_list : list[str], force_order : int = 1, dim : int = 6) -> np.ndarray:
    links = robot.link_list(link_name_list)
    mat = total_coord_to_link_force_grad_mat(robot, state, force_order=force_order, dim=dim)
    jacobs = np.zeros((dim * force_order * len(links), robot.dof * (force_order+2)))

    for i, link in enumerate(links):
        jacobs[i*dim*force_order:(i+1)*dim*force_order, :] = mat[link.id*dim*force_order:(link.id+1)*dim*force_order, :]
    return jacobs

def joint_force_jacobian(robot : RobotStruct, state : dict, joint_name_list : list[str], force_order : int = 1, dim : int = 6) -> np.ndarray:
    pass

def link_dynamics_jacobian_numerical(robot : RobotStruct, motions : RobotMotions, link_name_list : list[str], data_type, output_order_ : int = 1) -> np.ndarray:
    order = keys_time_order[data_type] + output_order_ - 1
    dynamics_order = keys_order[data_type]
    if dynamics_order < 1:
        dynamics_order = 1
    dof = data_type_dof(data_type, dim = 3) * output_order_
    
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
            state = outward_dynamics(robot, x, dynamics_order)
            y = np.zeros(dof)
            for j in range(output_order_):
                if j == 0:
                    y[6*j:6*(j+1)] = extract_dict_link_info(state, data_type, link_name_list[i])
                else:
                    y[6*j:6*(j+1)] = extract_dict_link_info(state, data_type+"_diff"+str(j), link_name_list[i])
            return y

        jacobs[dof*i:dof*(i+1)] = numerical_grad(motion, dynamics_func)

    return jacobs