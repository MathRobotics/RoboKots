#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025. 6.20 Created by T.Ishigaki
# outward computation module from motion and robot_model to state by matrix formulation

import numpy as np

from ..basic.robot import RobotStruct
from ..basic.state_dict import *

from ..dynamics.dynamics_matrix import inertia_diag_mat


def link_to_joint_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 6) -> np.ndarray:
    n_ = dim * order
    mat = np.zeros((r.joint_num * dim * order, r.link_num * dim * order))

    for i, joint in enumerate(r.joints):
        p_id = joint.parent_link_id
        c_id = joint.child_link_id
        rel_cmtm = state_dict_to_rel_cmtm(state, r.links[c_id].name, r.links[c_id].name, order)
        mat[i*n_:(i+1)*n_, p_id*n_:(p_id+1)*n_] = np.eye(n_)
        mat[i*n_:(i+1)*n_, c_id*n_:(c_id+1)*n_] = - rel_cmtm.mat()
    return mat

def link_to_joint_mat_inv(r : RobotStruct, state : dict, order : int = 1, dim : int = 6) -> np.ndarray:
    n_ = dim * order
    mat = np.zeros((r.joint_num * n_, r.link_num * n_))

    for i, joint in enumerate(r.joints):
        c_id = joint.child_link_id
        mat[c_id*n_:(c_id+1)*n_, i*n_:(i+1)*n_] = np.eye(n_)
        for j, link in enumerate(r.links):
            rel_cmtm = state_dict_to_rel_cmtm(state, r.links[c_id].name, link.name, order)
            mat[j*n_:(j+1)*n_, i*n_:(i+1)*n_] = rel_cmtm.mat()
    return mat

def joint_to_link_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 6) -> np.ndarray:
    return link_to_joint_mat_inv(r, state, order, dim)

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
        mat[i*n_:(i+1)*n_, joint.dof_index:joint.dof_index+joint.dof] = joint.select_mat

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