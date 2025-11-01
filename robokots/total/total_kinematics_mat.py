import numpy as np

from ..basic.robot import RobotStruct
from ..basic.state_dict import state_dict_to_cmtm, state_dict_to_rel_cmtm

from ..kinematics.kinematics_matrix import joint_select_diag_mat

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
