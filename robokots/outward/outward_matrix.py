#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025. 6.20 Created by T.Ishigaki
# outward computation module from motion and robot_model to state by matrix formulation

import numpy as np

from ..basic.robot import RobotStruct
from ..basic.state_dict import state_dict_to_rel_cmtm

from ..dynamics.dynamics_matrix import inertia_diag_mat
from ..kinematics.kinematics_matrix import joint_select_diag_mat


def link_to_joint_mat(r : RobotStruct, state : dict, order : int = 1, dim : int = 6) -> np.ndarray:
    n_ = dim * order
    mat = np.zeros((r.joint_num * n_, r.link_num * n_))

    for i, joint in enumerate(r.joints):
        p_id = joint.parent_link_id
        c_id = joint.child_link_id
        rel_cmtm = state_dict_to_rel_cmtm(state, r.links[c_id].name, r.links[p_id].name, order)
        mat[i*n_:(i+1)*n_, p_id*n_:(p_id+1)*n_] = - rel_cmtm.mat_adj()
        mat[i*n_:(i+1)*n_, c_id*n_:(c_id+1)*n_] = np.eye(n_)
    return mat

def joint_to_link_mat(r : RobotStruct, state : dict, order : int = 1, dim : int = 6) -> np.ndarray:
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

def link_inertia_mat(r : RobotStruct, order : int = 3, dim : int = 6) -> np.ndarray:
    n_ = dim * order
    mat = np.zeros((r.link_num * n_, r.link_num * n_))

    for i, link in enumerate(r.links):
        mat[i*n_:(i+1)*n_, i*n_:(i+1)*n_] = inertia_diag_mat(link.inertia, order)

    return mat

def coord_to_joint_mat(r : RobotStruct, order : int = 3, dim : int = 6) -> np.ndarray:
    n_ = dim * order
    mat = np.zeros((r.joint_num * n_, r.joint_dof * order))

    for i, joint in enumerate(r.joints):
        if joint.dof == 0:  # Joint with no degree of freedom
            continue
        mat[i*n_:(i+1)*n_, joint.dof_index*order:(joint.dof_index+joint.dof)*order] = joint_select_diag_mat(joint, order)

    return mat

def coord_to_joint_mat_inv(r : RobotStruct, order : int = 3, dim : int = 6) -> np.ndarray:
    n_ = dim * order
    mat = np.zeros((r.joint_dof * order, r.joint_num * n_))

    for i, joint in enumerate(r.joints):
        mat[joint.dof_index:joint.dof_index+joint.dof, i*n_:(i+1)*n_] = joint.select_mat.T

    return mat

def coord_to_link_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 6) -> np.ndarray:
    return joint_to_link_mat(r, state, order, dim) @ coord_to_joint_mat(r, order, dim)

def coord_to_link_momentum_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 6) -> np.ndarray:
    return link_inertia_mat(r, order, dim) @ joint_to_link_mat(r, state, order, dim) @ coord_to_joint_mat(r, order, dim)

def coord_to_joint_momentum_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 6) -> np.ndarray:
    return link_to_joint_mat(r, state, order, dim) @ link_inertia_mat(r, order, dim) @ \
            joint_to_link_mat(r, state, order, dim) @ coord_to_joint_mat(r, order, dim)

def link_jacobian(robot : RobotStruct, state : dict, link_name_list : list[str], order : int = 3, dim : int = 6) -> np.ndarray:
    links = robot.link_list(link_name_list)
    mat = coord_to_link_mat(robot, state, order=order, dim=dim)
    jacobs = np.zeros((dim * order * len(links), robot.dof * order))
    
    for i, link in enumerate(links):
        jacobs[i*dim*order:(i+1)*dim*order, :] = mat[link.id*dim*order:(link.id+1)*dim*order, :]
    return jacobs

def link_jacobian_momentum(robot : RobotStruct, state : dict, link_name_list : list[str], order : int = 3, dim : int = 6) -> np.ndarray:
    links = robot.link_list(link_name_list)
    mat = coord_to_link_momentum_mat(robot, state, order=order, dim=dim)
    jacobs = np.zeros((dim * order * len(links), robot.dof * order))

    for i, link in enumerate(links):
        jacobs[i*dim*order:(i+1)*dim*order, :] = mat[link.id*dim*order:(link.id+1)*dim*order, :]
    return jacobs

def joint_jacobian_momentum(robot : RobotStruct, state : dict, joint_name_list : list[str], order : int = 3, dim : int = 6) -> np.ndarray:
    joints = robot.joint_list(joint_name_list)
    mat = coord_to_joint_momentum_mat(robot, state, order=order, dim=dim)
    jacobs = np.zeros((dim * order * len(joints), robot.dof * order))

    for i, joint in enumerate(joints):
        jacobs[i*dim*order:(i+1)*dim*order, :] = mat[joint.id*dim*order:(joint.id+1)*dim*order, :]
    return jacobs

def link_jacob_force(robot : RobotStruct, state : dict, link_name_list : list[str], order : int = 3, dim : int = 6) -> np.ndarray:
    links = robot.link_list(link_name_list)
    mat = coord_to_link_momentum_mat(robot, state, order=order, dim=dim)
    jacobs = np.zeros((dim * order * len(links), robot.dof * order))

    for i, link in enumerate(links):
        jacobs[i*dim*order:(i+1)*dim*order, :] = mat[link.id*dim*order:(link.id+1)*dim*order, :]
    return jacobs

def joint_jacob_force(robot : RobotStruct, state : dict, joint_name_list : list[str], order : int = 3, dim : int = 6) -> np.ndarray:
    joints = robot.joint_list(joint_name_list)
    mat = coord_to_joint_momentum_mat(robot, state, order=order, dim=dim)
    jacobs = np.zeros((dim * order * len(joints), robot.dof * order))

    for i, joint in enumerate(joints):
        jacobs[i*dim*order:(i+1)*dim*order, :] = mat[joint.id*dim*order:(joint.id+1)*dim*order, :]
    return jacobs