#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025. 6.20 Created by T.Ishigaki
# outward computation module from motion and robot_model to state by matrix formulation

import numpy as np

from mathrobo import CMTM, SE3wrench, numerical_grad

from ..basic.robot import RobotStruct
from ..basic.state_dict import state_dict_to_rel_cmtm, state_dict_to_cmtm
from ..basic.state import keys_order, data_type_dof
from ..basic.state_dict import extract_dict_link_info
from ..basic.state_dict import extract_dict_total_link_cmvec
from ..basic.motion import RobotMotions

from ..dynamics.base import spatial_inertia
from ..dynamics.dynamics_matrix import inertia_diag_mat, momentum_to_force_mat, link_to_force_tan_map_mat
from ..kinematics.kinematics_matrix import joint_select_diag_mat

from ..total import total_link_inertia_mat

from .outward import dynamics_cmtm as outward_dynamics

def total_joint_to_link_vel_grad_mat(r : RobotStruct, state : dict, order : int = 1, dim : int = 6) -> np.ndarray:
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

def total_coord_to_joint_grad_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 6) -> np.ndarray:
    n_ = dim * order
    mat = np.zeros((r.joint_num * n_, r.joint_dof * order))

    for i, joint in enumerate(r.joints):
        if joint.dof == 0:  # Joint with no degree of freedom
            continue
        joint_cmtm = state_dict_to_cmtm(state, joint.name, order)
        mat[i*n_:(i+1)*n_, joint.dof_index*order:(joint.dof_index+joint.dof)*order] = \
            joint_cmtm.tangent_mat() @ joint_select_diag_mat(joint.select_mat, order)

    return mat

def total_coord_to_joint_grad_mat_inv(r : RobotStruct, state : dict, order : int = 3, dim : int = 6) -> np.ndarray:
    n_ = dim * order
    mat = np.zeros((r.joint_dof * order, r.joint_num * n_))

    for i, joint in enumerate(r.joints):
        if joint.dof == 0:  # Joint with no degree of freedom
            continue
        joint_cmtm = state_dict_to_cmtm(state, joint.name, order)
        mat[joint.dof_index:joint.dof_index+joint.dof, i*n_:(i+1)*n_] = \
            joint.select_mat.T @ joint_cmtm.tangent_mat_inv()

    return mat

def total_link_to_force_grad_mat(r : RobotStruct, state : dict, force_order : int = 1, dim : int = 6) -> np.ndarray:
    n_ = dim * force_order
    m_ = dim * (force_order+2)
    mat = np.zeros((r.link_num * n_, r.link_num * m_))

    for i, link in enumerate(r.links):
        cmtm = state_dict_to_cmtm(state, link.name, force_order+1)
        mat[i*n_:(i+1)*n_, i*m_:(i+1)*m_] = link_to_force_tan_map_mat(cmtm, spatial_inertia(link.mass, link.inertia, link.cog), force_order=force_order, dim=dim)
    return mat

def total_coord_to_link_grad_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 6) -> np.ndarray:
    return total_joint_to_link_vel_grad_mat(r, state, order, dim) @ total_coord_to_joint_grad_mat(r, state, order, dim)

def total_coord_to_link_momentum_grad_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 6) -> np.ndarray:
    return total_link_inertia_mat(r, order=order, dim=dim) @ total_coord_to_link_grad_mat(r, state, order, dim)

from ..total import total_link_to_joint_wrench_mat, total_world_link_to_joint_wrench_mat
from ..total import total_world_joint_cmtm_wrench_inv, total_world_link_cmtm_wrench
def total_coord_to_joint_momentum_grad_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 6) -> np.ndarray:
    link_momentum = extract_dict_total_link_cmvec(state, r.link_names, "link_momentum")
    t_l2j_moment = total_world_link_to_joint_wrench_mat(r, state, order)
    world_joint_momentum = t_l2j_moment @ total_world_link_cmtm_wrench(r, state, order, dim) @ link_momentum
    a1 = - total_world_joint_cmtm_wrench_inv(r, state, order, dim) @ CMTM.hat_adj
    a2 = total_world_link_cmtm_wrench(r, state, order, dim) @ t_l2j_moment
    j1 = (a1 + a2) @ total_joint_to_link_vel_grad_mat(r, state, order, dim)
    j2 = total_link_to_joint_wrench_mat(r, state, order, dim) @ total_coord_to_link_momentum_grad_mat(r, state, order, dim)
    return j2

def total_coord_to_link_force_grad_mat(r : RobotStruct, state : dict, force_order : int = 1, dim : int = 6) -> np.ndarray:
    return total_link_to_force_grad_mat(r, state, force_order=force_order, dim=dim) @ total_coord_to_link_grad_mat(r, state, force_order+2, dim)

def link_jacobian(robot : RobotStruct, state : dict, link_name_list : list[str], order : int = 3, dim : int = 6) -> np.ndarray:
    links = robot.link_list(link_name_list)
    mat = total_coord_to_link_grad_mat(robot, state, order=order, dim=dim)
    jacobs = np.zeros((dim * order * len(links), robot.dof * order))
    
    for i, link in enumerate(links):
        jacobs[i*dim*order:(i+1)*dim*order, :] = mat[link.id*dim*order:(link.id+1)*dim*order, :]
    return jacobs

def link_momentum_jacobian(robot : RobotStruct, state : dict, link_name_list : list[str], momentum_order : int = 1, dim : int = 6) -> np.ndarray:
    links = robot.link_list(link_name_list)
    mat = total_coord_to_link_momentum_grad_mat(robot, state, order=momentum_order, dim=dim)
    jacobs = np.zeros((dim * momentum_order * len(links), robot.dof * (momentum_order)))

    for i, link in enumerate(links):
        jacobs[i*dim*momentum_order:(i+1)*dim*momentum_order, :] = mat[link.id*dim*momentum_order:(link.id+1)*dim*momentum_order, :]
    return jacobs

def joint_momentum_jacobian(robot : RobotStruct, state : dict, joint_name_list : list[str], momentum_order : int = 1, dim : int = 6) -> np.ndarray:
    joints = robot.joint_list(joint_name_list)
    if joints == [None]:
        raise ValueError("joint_name_list contains invalid joint name")
    mat = total_coord_to_joint_momentum_grad_mat(robot, state, order=momentum_order, dim=dim)
    jacobs = np.zeros((dim * momentum_order * len(joints), robot.dof * momentum_order))

    for i, joint in enumerate(joints):
        jacobs[i*dim*momentum_order:(i+1)*dim*momentum_order, :] = mat[joint.id*dim*momentum_order:(joint.id+1)*dim*momentum_order, :]
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

def link_force_jacobian_numerical(robot : RobotStruct, motions : RobotMotions, link_name_list : list[str], force_order_ : int = 1) -> np.ndarray:
    order = force_order_ + 2
    dof = data_type_dof("force", dim = 3) * force_order_

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
            state = outward_dynamics(robot, x, force_order_)
            y = np.zeros(dof)
            for j in range(force_order_):
                if j == 0:
                    y[6*j:6*(j+1)] = extract_dict_link_info(state, "force", link_name_list[i])
                else:
                    y[6*j:6*(j+1)] = extract_dict_link_info(state, "force_diff"+str(j), link_name_list[i])
            return y

        jacobs[dof*i:dof*(i+1)] = numerical_grad(motion, dynamics_func)

    return jacobs