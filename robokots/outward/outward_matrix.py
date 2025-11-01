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
from ..basic.motion import RobotMotions

from ..dynamics.base import spatial_inertia
from ..dynamics.dynamics_matrix import inertia_diag_mat, momentum_to_force_mat, link_to_force_tan_map_mat
from ..kinematics.kinematics_matrix import joint_select_diag_mat

from .outward import dynamics_cmtm as outward_dynamics

def total_link_to_joint_vel_mat(r : RobotStruct, state : dict, order : int = 1, dim : int = 6) -> np.ndarray:
    n_ = dim * order
    mat = np.zeros((r.joint_num * n_, r.link_num * n_))

    for i, joint in enumerate(r.joints):
        p_id = joint.parent_link_id
        c_id = joint.child_link_id
        rel_cmtm = state_dict_to_rel_cmtm(state, r.links[c_id].name, r.links[p_id].name, order)
        mat[i*n_:(i+1)*n_, p_id*n_:(p_id+1)*n_] = - rel_cmtm.mat_adj()
        mat[i*n_:(i+1)*n_, c_id*n_:(c_id+1)*n_] = np.eye(n_)
    return mat

def total_joint_to_link_vel_mat(r : RobotStruct, state : dict, order : int = 1, dim : int = 6) -> np.ndarray:
    n_ = dim * order
    mat = np.zeros((r.link_num * n_, r.joint_num * n_))

    for i, link in enumerate(r.links):
        link_route = []
        joint_route = []
        r.route_target_link(link, link_route, joint_route)
        for j in joint_route:
            joint = r.joints[j]
            rel_cmtm = state_dict_to_rel_cmtm(state, link.name, r.links[joint.child_link_id].name, order)
            mat[i*n_:(i+1)*n_, j*n_:(j+1)*n_] = rel_cmtm.mat_adj()
    return mat

def total_joint_to_link_momentum_mat(r : RobotStruct, state : dict, order : int = 1, dim : int = 6) -> np.ndarray:
    n_ = dim * order
    mat = np.zeros((r.link_num * n_, r.joint_num * n_))

    for i, joint in enumerate(r.joints):
        p_id = joint.parent_link_id
        c_id = joint.child_link_id
        rel_cmtm_wrench = state_dict_to_rel_cmtm_wrench(state, r.links[c_id].name, r.links[p_id].name, order)
        mat[i*n_:(i+1)*n_, p_id*n_:(p_id+1)*n_] = np.eye(n_)
        mat[i*n_:(i+1)*n_, c_id*n_:(c_id+1)*n_] = - rel_cmtm_wrench.mat_adj()
    return mat

def total_link_to_joint_momentum_mat(r : RobotStruct, state : dict, order : int = 1, dim : int = 6) -> np.ndarray:
    n_ = dim * order
    mat = np.zeros((r.joint_num * n_, r.link_num * n_))
    for i, joint in enumerate(r.joints):
        link_route = []
        joint_route = []
        r.route_end_joints(joint, link_route, joint_route)
        for j in link_route:
            link = r.links[j]
            rel_cmtm_wrench = state_dict_to_rel_cmtm_wrench(state, r.links[joint.child_link_id].name, link.name, order)
            mat[i*n_:(i+1)*n_, j*n_:(j+1)*n_] = rel_cmtm_wrench.mat_adj()
    return mat

def total_world_link_cmtm(r : RobotStruct, state : dict, order : int = 1, dim : int = 6) -> np.ndarray:
    n_ = dim * order
    mat = np.zeros((r.link_num * n_, r.link_num * n_))

    for i, link in enumerate(r.links):
        cmtm = state_dict_to_cmtm(state, link.name, order)
        mat[i*n_:(i+1)*n_, i*n_:(i+1)*n_] = cmtm.mat_adj()
    return mat

def total_world_link_cmtm_inv(r : RobotStruct, state : dict, order : int = 1, dim : int = 6) -> np.ndarray:
    n_ = dim * order
    mat = np.zeros((r.link_num * n_, r.link_num * n_))

    for i, link in enumerate(r.links):
        cmtm = state_dict_to_cmtm(state, link.name, order)
        mat[i*n_:(i+1)*n_, i*n_:(i+1)*n_] = cmtm.mat_inv_adj()
    return mat

def total_world_link_cmtm_wrench(r : RobotStruct, state : dict, order : int = 1, dim : int = 6) -> np.ndarray:
    n_ = dim * order
    mat = np.zeros((r.link_num * n_, r.link_num * n_))

    for i, link in enumerate(r.links):
        cmtm_wrench = state_dict_to_cmtm_wrench(state, link.name, order)
        mat[i*n_:(i+1)*n_, i*n_:(i+1)*n_] = cmtm_wrench.mat_adj()
    return mat

def total_world_link_cmtm_wrench_inv(r : RobotStruct, state : dict, order : int = 1, dim : int = 6) -> np.ndarray:
    n_ = dim * order
    mat = np.zeros((r.link_num * n_, r.link_num * n_))

    for i, link in enumerate(r.links):
        cmtm_wrench = state_dict_to_cmtm_wrench(state, link.name, order)
        mat[i*n_:(i+1)*n_, i*n_:(i+1)*n_] = cmtm_wrench.mat_inv_adj()
    return mat

def total_world_joint_cmtm(r : RobotStruct, state : dict, order : int = 1, dim : int = 6) -> np.ndarray:
    n_ = dim * order
    mat = np.zeros((r.joint_num * n_, r.joint_num * n_))

    for i, joint in enumerate(r.joints):
        cmtm = state_dict_to_cmtm(state, r.links[joint.child_link_id].name, order)
        mat[i*n_:(i+1)*n_, i*n_:(i+1)*n_] = cmtm.mat_adj()
    return mat

def total_world_joint_cmtm_inv(r : RobotStruct, state : dict, order : int = 1, dim : int = 6) -> np.ndarray:
    n_ = dim * order
    mat = np.zeros((r.joint_num * n_, r.joint_num * n_))

    for i, joint in enumerate(r.joints):
        cmtm = state_dict_to_cmtm(state, r.links[joint.child_link_id].name, order)
        mat[i*n_:(i+1)*n_, i*n_:(i+1)*n_] = cmtm.mat_inv_adj()
    return mat

def total_world_joint_cmtm_wrench(r : RobotStruct, state : dict, order : int = 1, dim : int = 6) -> np.ndarray:
    n_ = dim * order
    mat = np.zeros((r.joint_num * n_, r.joint_num * n_))

    for i, joint in enumerate(r.joints):
        cmtm_wrench = state_dict_to_cmtm_wrench(state, r.links[joint.child_link_id].name, order)
        mat[i*n_:(i+1)*n_, i*n_:(i+1)*n_] = cmtm_wrench.mat_adj()
    return mat

def total_world_joint_cmtm_wrench_inv(r : RobotStruct, state : dict, order : int = 1, dim : int = 6) -> np.ndarray:
    n_ = dim * order
    mat = np.zeros((r.joint_num * n_, r.joint_num * n_))

    for i, joint in enumerate(r.joints):
        cmtm_wrench = state_dict_to_cmtm_wrench(state, r.links[joint.child_link_id].name, order)
        mat[i*n_:(i+1)*n_, i*n_:(i+1)*n_] = cmtm_wrench.mat_inv_adj()
    return mat

def total_world_joint_to_link_momentum_mat(r : RobotStruct, order : int = 1, dim : int = 6) -> np.ndarray:
    n_ = dim * order
    mat = np.zeros((r.link_num * n_, r.joint_num * n_))

    for i, joint in enumerate(r.joints):
        p_id = joint.parent_link_id
        c_id = joint.child_link_id
        mat[i*n_:(i+1)*n_, p_id*n_:(p_id+1)*n_] = np.eye(n_)
        mat[i*n_:(i+1)*n_, c_id*n_:(c_id+1)*n_] = - np.eye(n_)
    return mat

def total_world_link_to_joint_momentum_mat(r : RobotStruct, order : int = 1, dim : int = 6) -> np.ndarray:
    n_ = dim * order
    mat = np.zeros((r.joint_num * n_, r.link_num * n_))
    for i, joint in enumerate(r.joints):
        link_route = []
        joint_route = []
        r.route_end_joints(joint, link_route, joint_route)
        for j in link_route:
            mat[i*n_:(i+1)*n_, j*n_:(j+1)*n_] = np.eye(n_)
    return mat

def total_link_inertia_mat(r : RobotStruct, order : int = 3, dim : int = 6) -> np.ndarray:
    n_ = dim * order
    mat = np.zeros((r.link_num * n_, r.link_num * n_))

    for i, link in enumerate(r.links):
        mat[i*n_:(i+1)*n_, i*n_:(i+1)*n_] = inertia_diag_mat(spatial_inertia(link.mass, link.inertia, link.cog), order)

    return mat

def total_momentum_to_force_mat(r : RobotStruct, state : dict, force_order : int = 1, dim : int = 6) -> np.ndarray:
    n_ = dim * force_order
    m_ = dim * (force_order+1)
    mat = np.zeros((r.link_num * n_, r.link_num * m_))

    for i, link in enumerate(r.links):
        cmtm = state_dict_to_cmtm(state, link.name, force_order+1)
        mat[i*n_:(i+1)*n_, i*m_:(i+1)*m_] = momentum_to_force_mat(cmtm, force_order=force_order, dim=dim)
    return mat

def total_coord_to_joint_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 6) -> np.ndarray:
    n_ = dim * order
    mat = np.zeros((r.joint_num * n_, r.joint_dof * order))

    for i, joint in enumerate(r.joints):
        if joint.dof == 0:  # Joint with no degree of freedom
            continue
        mat[i*n_:(i+1)*n_, joint.dof_index*order:(joint.dof_index+joint.dof)*order] = \
            joint_select_diag_mat(joint.select_mat, order)

    return mat

def total_coord_to_joint_mat_inv(r : RobotStruct, state : dict, order : int = 3, dim : int = 6) -> np.ndarray:
    n_ = dim * order
    mat = np.zeros((r.joint_dof * order, r.joint_num * n_))

    for i, joint in enumerate(r.joints):
        if joint.dof == 0:  # Joint with no degree of freedom
            continue
        mat[joint.dof_index:joint.dof_index+joint.dof, i*n_:(i+1)*n_] = \
            joint.select_mat.T

    return mat

def total_coord_to_link_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 6) -> np.ndarray:
    return total_joint_to_link_vel_mat(r, state, order, dim) @ total_coord_to_joint_mat(r, state,order, dim)

def total_coord_to_link_momentum_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 6) -> np.ndarray:
    return total_link_inertia_mat(r, order=order, dim=dim) @ total_coord_to_link_mat(r, state, order, dim)

def total_coord_to_joint_momentum_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 6) -> np.ndarray:
    return total_link_to_joint_momentum_mat(r, state, order, dim) @ total_coord_to_link_momentum_mat(r, state, order, dim)

def total_coord_to_link_force_mat(r : RobotStruct, state : dict, force_order : int = 1, dim : int = 6) -> np.ndarray:
    return total_momentum_to_force_mat(r, state, force_order, dim) @ total_coord_to_link_momentum_mat(r, state, force_order+1, dim)

def total_coord_to_joint_force_mat(r : RobotStruct, state : dict, order : int = 1, dim : int = 6) -> np.ndarray:
    pass