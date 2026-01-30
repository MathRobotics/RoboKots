import numpy as np
from mathrobo import CMTM

from robokots.core import RobotStruct
from robokots.core.state import dim_to_dof
from robokots.core.state_dict import state_dict_to_cmtm, state_dict_to_rel_cmtm

from ..kinematics.kinematics_matrix import joint_select_diag_mat

def total_coord_arrange(r : RobotStruct, out_order : int = 3, in_order : int = 3) -> np.ndarray:
    mat = np.zeros((r.joint_dof * out_order, r.joint_dof * in_order))
    row = 0
    col = 0
    for i, joint in enumerate(r.joints):
        if joint.dof == 0:  # Joint with no degree of freedom
            continue
        mat[row:row+joint.dof*out_order, col:col+joint.dof*out_order] = np.eye(joint.dof*out_order)
        row += joint.dof * out_order
        col += joint.dof * in_order
    return mat

def total_cmtm_hat(vec : np.ndarray, mat_type, num : int, order : int, vec_dim : int = 6) -> np.ndarray:
    '''
     Create a block diagonal matrix where each block is the hat matrix of the given vector.
     Args:
        vec (np.ndarray): Input vector of shape (num * dim * order, ).
        mat_type: Type of the matrix to be created using CMTM.hat.
        num (int): Number of blocks.
        order (int): Order of the CMTM.
        dim (int, optional): Dimension of the space. Defaults to 6.
    '''
    n_ = vec_dim * order
    mat = np.zeros((num * n_, num * n_))

    for i in range(num):
        start = i * n_
        mat[start:start+n_, start:start+n_] = CMTM.hat_adj(mat_type, vec[start:start+n_].reshape(order, vec_dim))
    return mat

def total_cmtm_hat_commute(vec : np.ndarray, mat_type, num : int, order : int, vec_dim : int = 6) -> np.ndarray:
    '''
    Create a block diagonal matrix where each block is the commute matrix of the given type.
    Args:
        vec (np.ndarray): Input vector of shape (num * dim * order, ).  
        mat_type: Type of the matrix to be created using CMTM.commute.
        num (int): Number of blocks.
        order (int): Order of the CMTM.
        vec_dim (int, optional): Dimension of the space. Defaults to 6.
    '''
    if vec.shape[0] != num * vec_dim * order:
        raise ValueError("Input vector has incorrect shape.")
    n_ = vec_dim * order
    mat = np.zeros((num * n_, num * n_))
    for i in range(num):
        start = i * n_
        mat[start:start+n_, start:start+n_] = CMTM.hat_commute_adj(mat_type, vec[start:start+n_].reshape(order, vec_dim))
    return mat

def total_world_link_cmtm(r : RobotStruct, state : dict, order : int = 1, dim : int = 3) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    mat = np.zeros((r.link_num * n_, r.link_num * n_))

    for i, link in enumerate(r.links):
        cmtm = state_dict_to_cmtm(state, link.name, "link", order)
        mat[i*n_:(i+1)*n_, i*n_:(i+1)*n_] = cmtm.mat_adj()
    return mat

def total_world_link_cmtm_inv(r : RobotStruct, state : dict, order : int = 1, dim : int = 3) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    mat = np.zeros((r.link_num * n_, r.link_num * n_))

    for i, link in enumerate(r.links):
        cmtm = state_dict_to_cmtm(state, link.name, "link", order)
        mat[i*n_:(i+1)*n_, i*n_:(i+1)*n_] = cmtm.mat_inv_adj()
    return mat

def total_world_joint_cmtm(r : RobotStruct, state : dict, order : int = 1, dim : int = 3) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    mat = np.zeros((r.joint_num * n_, r.joint_num * n_))

    for i, joint in enumerate(r.joints):
        cmtm = state_dict_to_cmtm(state, r.links[joint.child_linkz_id].name, "link", order)
        mat[i*n_:(i+1)*n_, i*n_:(i+1)*n_] = cmtm.mat_adj()
    return mat

def total_world_joint_cmtm_inv(r : RobotStruct, state : dict, order : int = 1, dim : int = 3) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    mat = np.zeros((r.joint_num * n_, r.joint_num * n_))

    for i, joint in enumerate(r.joints):
        cmtm = state_dict_to_cmtm(state, r.links[joint.child_link_id].name, "link", order)
        mat[i*n_:(i+1)*n_, i*n_:(i+1)*n_] = cmtm.mat_inv_adj()
    return mat

def total_link_vel_to_joint_vel_mat(r : RobotStruct, state : dict, order : int = 1, dim : int = 3) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    mat = np.zeros((r.joint_num * n_, r.link_num * n_))

    for i, joint in enumerate(r.joints):
        p_id = joint.parent_link_id
        c_id = joint.child_link_id
        rel_cmtm = state_dict_to_rel_cmtm(state, r.links[c_id].name, r.links[p_id].name, "link", order)
        mat[i*n_:(i+1)*n_, p_id*n_:(p_id+1)*n_] = - rel_cmtm.mat_adj()
        mat[i*n_:(i+1)*n_, c_id*n_:(c_id+1)*n_] = np.eye(n_)
    return mat

def total_joint_vel_to_link_vel_mat(r : RobotStruct, state : dict, order : int = 1, dim : int = 3) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    mat = np.zeros((r.link_num * n_, r.joint_num * n_))

    for i, link in enumerate(r.links):
        link_route = []
        joint_route = []
        r.route_target_link(link, link_route, joint_route)
        for j in joint_route:
            joint = r.joints[j]
            rel_cmtm = state_dict_to_rel_cmtm(state, link.name, r.links[joint.child_link_id].name, "link", order)
            mat[i*n_:(i+1)*n_, j*n_:(j+1)*n_] = rel_cmtm.mat_adj()
    return mat

def total_coord_to_joint_vel_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 3) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    mat = np.zeros((r.joint_num * n_, r.joint_dof * order))

    for i, joint in enumerate(r.joints):
        if joint.dof == 0:  # Joint with no degree of freedom
            continue
        mat[i*n_:(i+1)*n_, joint.dof_index*order:(joint.dof_index+joint.dof)*order] = \
            joint_select_diag_mat(joint.select_mat, order)

    return mat

def total_coord_to_link_vel_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 3) -> np.ndarray:
    return total_joint_vel_to_link_vel_mat(r, state, order, dim) @ total_coord_to_joint_vel_mat(r, state, order, dim)
