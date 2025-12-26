import numpy as np
from mathrobo import Factorial, CMVector
from robokots.core import RobotStruct
from robokots.core.state import dim_to_dof
from robokots.core.state_dict import state_dict_to_cmtm, state_dict_to_cmtm_wrench

def total_factorial_mat(num : int, order : int, submat_dim : int = 6) -> np.ndarray:
    '''
    Create a block diagonal matrix where each block is the factorial matrix.
    Args:
        order (int): Order of the CMTM.
        dim (int, optional): Dimension of the space. Defaults to 6.
    '''
    n = submat_dim * order
    mat = np.zeros((num * n, num * n))

    for i in range(num):
        start = i * n
        mat[start:start+n, start:start+n] = Factorial.mat(order, submat_dim)
    return mat

def total_factorial_mat_inv(num : int, order : int, submat_dim : int = 6) -> np.ndarray:
    '''
    Create a block diagonal matrix where each block is the factorial matrix.
    Args:
        order (int): Order of the CMTM.
        dim (int, optional): Dimension of the space. Defaults to 6.
    '''
    n = submat_dim * order
    mat = np.zeros((num * n, num * n))

    for i in range(num):
        start = i * n
        mat[start:start+n, start:start+n] = Factorial.mat_inv(order, submat_dim)
    return mat

def total_link_cmtm_var_x_arb_vec(r : RobotStruct, state : dict, total_cm_vec : np.ndarray, order : int = 1, dim : int = 3) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    mat = np.zeros((r.link_num * n_, r.link_num * n_))
    total_cm_vecs = total_cm_vec.reshape(r.link_num, n_)
    for i, link in enumerate(r.links):
        arb_v = CMVector.set_cmvecs(total_cm_vecs[i].reshape(order, -1))
        m = state_dict_to_cmtm(state, link.name, "link", order).mat_var_x_arb_vec_jacob(arb_v, frame='bframe')
        mat[i*n_:(i+1)*n_, i*n_:(i+1)*n_] = m
    return mat

def total_joint_cmtm_var_x_arb_vec(r : RobotStruct, state : dict, total_cm_vec : np.ndarray, order : int = 1, dim : int = 3) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    mat = np.zeros((r.joint_num * n_, r.joint_num * n_))
    total_cm_vecs = total_cm_vec.reshape(r.joint_num, n_)
    for i, joint in enumerate(r.joints):
        arb_v = CMVector.set_cmvecs(total_cm_vecs[i].reshape(order, -1))
        c_link = r.links[joint.child_link_id]
        m = state_dict_to_cmtm(state, c_link.name, "link", order).mat_var_x_arb_vec_jacob(arb_v, frame='bframe')
        mat[i*n_:(i+1)*n_, i*n_:(i+1)*n_] = m
    return mat

def total_link_cmtm_wrench_var_x_arb_vec(r : RobotStruct, state : dict, total_cm_vec : np.ndarray, order : int = 1, dim : int = 3) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    mat = np.zeros((r.link_num * n_, r.link_num * n_))
    total_cm_vecs = total_cm_vec.reshape(r.link_num, n_)
    for i, link in enumerate(r.links):
        arb_v = CMVector.set_cmvecs(total_cm_vecs[i].reshape(order, -1))
        m = state_dict_to_cmtm_wrench(state, link.name, "link", order).mat_var_x_arb_vec_jacob(arb_v, frame='bframe')
        mat[i*n_:(i+1)*n_, i*n_:(i+1)*n_] = m
    return mat

def total_joint_cmtm_wrench_var_x_arb_vec(r : RobotStruct, state : dict, total_cm_vec : np.ndarray, order : int = 1, dim : int = 3) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    mat = np.zeros((r.joint_num * n_, r.joint_num * n_))
    total_cm_vecs = total_cm_vec.reshape(r.joint_num, n_)
    for i, joint in enumerate(r.joints):
        arb_v = CMVector.set_cmvecs(total_cm_vecs[i].reshape(order, -1))
        c_link = r.links[joint.child_link_id]
        m = state_dict_to_cmtm_wrench(state, c_link.name, "link", order).mat_var_x_arb_vec_jacob(arb_v, frame='bframe')
        mat[i*n_:(i+1)*n_, i*n_:(i+1)*n_] = m
    return mat

def total_joint_cmtm_wrench_inv_var_x_arb_vec(r : RobotStruct, state : dict, total_cm_vec : np.ndarray, order : int = 1, dim : int = 3) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    mat = np.zeros((r.joint_num * n_, r.joint_num * n_))
    total_cm_vecs = total_cm_vec.reshape(r.joint_num, n_)
    for i, joint in enumerate(r.joints):
        arb_v = CMVector.set_cmvecs(total_cm_vecs[i].reshape(order, -1))
        c_link = r.links[joint.child_link_id]
        cmtm_wrench = state_dict_to_cmtm_wrench(state, c_link.name, "link", order)
        m = cmtm_wrench.mat_inv_var_x_arb_vec_jacob(arb_v, frame='bframe')
        mat[i*n_:(i+1)*n_, i*n_:(i+1)*n_] = m
    return mat