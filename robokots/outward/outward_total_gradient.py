#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025. 6.20 Created by T.Ishigaki
# outward computation module from motion and robot_model to state by matrix formulation

import numpy as np

from mathrobo import CMTM, SE3wrench, numerical_grad, Factorial

from ..basic.robot import RobotStruct
from ..basic.state_dict import state_dict_to_rel_cmtm, state_dict_to_cmtm
from ..basic.state import keys_order, data_type_dof, keys_time_order
from ..basic.state_dict import extract_dict_link_info
from ..basic.state_dict import extract_dict_total_link_cmvec, extract_dict_total_link
from ..basic.motion import RobotMotions

from ..dynamics.base import spatial_inertia
from ..dynamics.dynamics_matrix import inertia_diag_mat, momentum_to_force_mat, link_to_force_tan_map_mat
from ..kinematics.kinematics_matrix import joint_select_diag_mat

from ..total import total_link_inertia_mat, total_factorial_mat, total_factorial_mat_inv

from .outward import dynamics_cmtm as outward_dynamics

def total_joint_tan_vel_to_link_vel_grad_mat(r : RobotStruct, state : dict, order : int = 1, dim : int = 6) -> np.ndarray:
    n_ = dim * order
    mat = np.zeros((r.link_num * n_, r.joint_num * n_))

    for i, link in enumerate(r.links):
        link_route = []
        joint_route = []
        r.route_target_link(link, link_route, joint_route)
        for j in joint_route:
            joint = r.joints[j]
            rel_cmtm = state_dict_to_rel_cmtm(state, link.name, r.links[joint.child_link_id].name, order)
            link_cmtm = state_dict_to_cmtm(state, link.name, order)
            mat[i*n_:(i+1)*n_, j*n_:(j+1)*n_] = link_cmtm.tangent_mat_inv() @ rel_cmtm.mat_adj()
    return mat

def total_joint_tan_vel_to_link_sp_vel_grad_mat(r : RobotStruct, state : dict, order : int = 1, dim : int = 6) -> np.ndarray:
    n_j = dim * order
    n_l = dim * (order-1)
    mat = np.zeros((r.link_num * n_l, r.joint_num * n_j))

    for i, link in enumerate(r.links):
        link_route = []
        joint_route = []
        r.route_target_link(link, link_route, joint_route)
        for j in joint_route:
            joint = r.joints[j]
            rel_cmtm = state_dict_to_rel_cmtm(state, link.name, r.links[joint.child_link_id].name, order)
            link_cmtm = state_dict_to_cmtm(state, link.name, order)
            mat[i*n_l:(i+1)*n_l, j*n_j:(j+1)*n_j] = link_cmtm.tangent_mat_inv()[dim:] @ rel_cmtm.mat_adj()
    return mat

def total_coord_to_joint_tan_vel_grad_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 6) -> np.ndarray:
    n_ = dim * order
    mat = np.zeros((r.joint_num * n_, r.joint_dof * order))

    for i, joint in enumerate(r.joints):
        if joint.dof == 0:  # Joint with no degree of freedom
            continue
        joint_cmtm = state_dict_to_cmtm(state, joint.name, order)
        mat[i*n_:(i+1)*n_, joint.dof_index*order:(joint.dof_index+joint.dof)*order] = \
            joint_cmtm.tangent_mat() @ joint_select_diag_mat(joint.select_mat, order)

    return mat

def total_coord_to_joint_tan_vel_grad_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 6) -> np.ndarray:
    n_ = dim * (order-1)
    mat = np.zeros((r.joint_num * n_, r.joint_dof * order))

    for i, joint in enumerate(r.joints):
        if joint.dof == 0:  # Joint with no degree of freedom
            continue
        joint_cmtm = state_dict_to_cmtm(state, joint.name, order)
        mat[i*n_:(i+1)*n_, joint.dof_index*order:(joint.dof_index+joint.dof)*order] = \
            joint_cmtm.tangent_mat()[dim:] @ joint_select_diag_mat(joint.select_mat, order)

    return mat

def total_joint_wrench_to_force_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 6) -> np.ndarray:
    n_ = dim * order
    mat = np.zeros((r.joint_dof * order, r.joint_num * n_))

    for i, joint in enumerate(r.joints):
        if joint.dof == 0:  # Joint with no degree of freedom
            continue
        joint_cmtm = state_dict_to_cmtm(state, joint.name, order)
        mat[joint.dof_index:joint.dof_index+joint.dof, i*n_:(i+1)*n_] = \
            joint.select_mat.T @ joint_cmtm.tangent_mat_inv()

    return mat

def total_link_sp_vel_to_link_force_grad_mat(r : RobotStruct, state : dict, force_order : int = 1, dim : int = 6) -> np.ndarray:
    n_ = dim * force_order
    m_ = dim * (force_order+2)
    mat = np.zeros((r.link_num * n_, r.link_num * m_))

    for i, link in enumerate(r.links):
        cmtm = state_dict_to_cmtm(state, link.name, force_order+1)
        mat[i*n_:(i+1)*n_, i*m_:(i+1)*m_] = link_to_force_tan_map_mat(cmtm, spatial_inertia(link.mass, link.inertia, link.cog), force_order=force_order, dim=dim)
    return mat

def total_coord_to_link_vel_grad_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 6) -> np.ndarray:
    return total_joint_tan_vel_to_link_vel_grad_mat(r, state, order, dim) @ total_coord_to_joint_tan_vel_grad_mat(r, state, order, dim)

def total_coord_to_link_momentum_grad_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 6) -> np.ndarray:
    return total_link_inertia_mat(r, order=order-1, dim=dim) @ total_joint_tan_vel_to_link_sp_vel_grad_mat(r, state, order, dim) @ total_coord_to_joint_tan_vel_grad_mat(r, state, order, dim)

from ..total import total_cmtm_hat_commute
from ..total import total_link_to_joint_wrench_mat, total_world_link_to_joint_wrench_mat
from ..total import total_world_joint_cmtm_wrench_inv, total_world_link_cmtm_wrench
def total_coord_to_joint_momentum_grad_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 6) -> np.ndarray:
    link_momentum = extract_dict_total_link_cmvec(state, r.link_names, "link_momentum", order)
    t_l2j_moment = total_world_link_to_joint_wrench_mat(r, order, dim)
    t_w_l_wrench_inv = total_world_joint_cmtm_wrench_inv(r, state, order, dim)
    t_w_j_wrench = total_world_link_cmtm_wrench(r, state, order, dim)
    world_joint_momentum = t_l2j_moment @ t_w_j_wrench @ link_momentum
    tf_j_m = total_factorial_mat(r.joint_num, order, dim)
    tf_j_m_inv = total_factorial_mat_inv(r.joint_num, order+1, dim)
    tf_l_m_inv = total_factorial_mat_inv(r.link_num, order, dim)

    j1 = tf_j_m @ total_link_to_joint_wrench_mat(r, state, order, dim) @ tf_l_m_inv \
         @ total_coord_to_link_momentum_grad_mat(r, state, order+1, dim)
    j2 = tf_j_m @ t_w_l_wrench_inv @ t_l2j_moment @ t_w_j_wrench @ total_cmtm_hat_commute(link_momentum, SE3wrench, num=r.link_num, order=order, dim=dim) \
         @ total_joint_tan_vel_to_link_sp_vel_grad_mat(r, state, order+1, dim) @ tf_j_m_inv @ total_coord_to_joint_tan_vel_grad_mat(r, state, order+1, dim)
    tf_j_m_inv = total_factorial_mat_inv(r.joint_num, order, dim)
    j3 = tf_j_m @ -t_w_l_wrench_inv @ total_cmtm_hat_commute(world_joint_momentum, SE3wrench, num=r.joint_num, order=order, dim=dim) \
         @ tf_j_m_inv @ total_coord_to_joint_tan_vel_grad_mat(r, state, order+1, dim)

    return j1 + j2 + j3

def total_coord_to_link_force_grad_mat(r : RobotStruct, state : dict, force_order : int = 1, dim : int = 6) -> np.ndarray:
    return total_link_sp_vel_to_link_force_grad_mat(r, state, force_order=force_order, dim=dim) @ total_coord_to_link_vel_grad_mat(r, state, force_order+2, dim)

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