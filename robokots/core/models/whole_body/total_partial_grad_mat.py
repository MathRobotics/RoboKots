import numpy as np

from robokots.core import RobotStruct
from robokots.core.state import dim_to_dof
from robokots.core.state_dict import extract_dict_total_link_cmvec, state_dict_to_cmtm, state_dict_to_cmvec
from ..dynamics.base import spatial_inertia
from ..dynamics.dynamics_matrix import link_sp_vel_to_link_force_grad_mat, partial_link_sp_vel_to_force_grad_mat, partial_momentum_to_force_grad_mat
from .basic import total_factorial_mat, total_factorial_mat_inv
from .basic import total_factorial_matvec, total_factorial_mat_inv_vec
from .basic import total_link_cmtm_wrench_var_x_arb_vec, total_joint_cmtm_wrench_inv_var_x_arb_vec
from .basic import total_link_cmtm_wrench_var_x_arb_vec_matvec, total_joint_cmtm_wrench_inv_var_x_arb_vec_matvec
from .total_dynamics_mat import total_world_link_cmtm_wrench, total_world_joint_cmtm_wrench_inv
from .total_dynamics_mat import total_world_link_cmtm_wrench_matvec, total_world_joint_cmtm_wrench_inv_matvec
from .total_dynamics_mat import total_world_link_wrench_to_world_joint_wrench_mat
from .total_dynamics_mat import total_world_link_wrench_to_world_joint_wrench_matvec

'''
    Gradients of link momentum with respect to world frame
'''
def total_partial_link_momentum_to_world_link_momentum_grad_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 3) -> np.ndarray:
    dof = dim_to_dof(dim)
    return total_factorial_mat(r.link_num, order-1, dof) @  total_world_link_cmtm_wrench(r, state, order-1, dim) @ total_factorial_mat_inv(r.link_num, order-1, dof) 

def total_partial_link_momentum_to_world_link_momentum_grad_matvec(r : RobotStruct, state : dict, vec : np.ndarray, order : int = 3, dim : int = 3) -> np.ndarray:
    dof = dim_to_dof(dim)
    inv_fact_vec = total_factorial_mat_inv_vec(r.link_num, vec, order-1, dof)
    cmtm_vec = total_world_link_cmtm_wrench_matvec(r, state, inv_fact_vec, order-1, dim)
    return total_factorial_matvec(r.link_num, cmtm_vec, order-1, dof)

def total_partial_link_tan_vel_to_world_link_momentum_grad_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 3) -> np.ndarray:
    dof = dim_to_dof(dim)
    total_local_link_momentum = extract_dict_total_link_cmvec(state, r.link_names, "momentum", order-1)
    return total_factorial_mat(r.link_num, order-1, dof) @ total_link_cmtm_wrench_var_x_arb_vec(r, state, total_local_link_momentum, order-1, dim)

def total_partial_link_tan_vel_to_world_link_momentum_grad_matvec(r : RobotStruct, state : dict, vec : np.ndarray, order : int = 3, dim : int = 3) -> np.ndarray:
    dof = dim_to_dof(dim)
    total_local_link_momentum = extract_dict_total_link_cmvec(state, r.link_names, "momentum", order-1)
    cmtm_vec = total_link_cmtm_wrench_var_x_arb_vec_matvec(r, state, total_local_link_momentum, vec, order-1, dim)
    return total_factorial_matvec(r.link_num, cmtm_vec, order-1, dof)

'''
    Gradients of force
'''

def total_link_sp_vel_to_link_force_grad_mat(r : RobotStruct, state : dict, force_order : int = 1, dim : int = 3) -> np.ndarray:
    n_ = dim_to_dof(dim) * force_order
    m_ = dim_to_dof(dim) * (force_order+2)
    mat = np.zeros((r.link_num * n_, r.link_num * m_))

    for i, link in enumerate(r.links):
        cmtm = state_dict_to_cmtm(state, link.name, "link", force_order+1)
        mat[i*n_:(i+1)*n_, i*m_:(i+1)*m_] = link_sp_vel_to_link_force_grad_mat(cmtm, spatial_inertia(link.mass, link.inertia, link.cog), force_order=force_order, dim=dim)
    return mat

def total_partial_momentum_to_force_grad_mat(r : RobotStruct, state : dict, force_order : int = 1, dim : int = 3) -> np.ndarray:
    n_ = dim_to_dof(dim) * force_order
    m_ = dim_to_dof(dim) * (force_order+1)
    mat = np.zeros((r.link_num * n_, r.link_num * m_))

    for i, link in enumerate(r.links):
        cmtm = state_dict_to_cmtm(state, link.name, "link", force_order+1)
        mat[i*n_:(i+1)*n_, i*m_:(i+1)*m_] = partial_momentum_to_force_grad_mat(cmtm, force_order=force_order, dim=dim)
    return mat

def total_partial_momentum_to_force_grad_matvec(r : RobotStruct, state : dict, vec : np.ndarray, force_order : int = 1, dim : int = 3) -> np.ndarray:
    n_ = dim_to_dof(dim) * force_order
    m_ = dim_to_dof(dim) * (force_order+1)
    result = np.zeros(r.link_num * n_)

    for i, link in enumerate(r.links):
        out_start = i * n_
        in_start = i * m_
        cmtm = state_dict_to_cmtm(state, link.name, "link", force_order+1)
        mat = partial_momentum_to_force_grad_mat(cmtm, force_order=force_order, dim=dim)
        result[out_start:out_start+n_] = mat @ vec[in_start:in_start+m_]
    return result

'''
    Gradients of link force
'''
def total_partial_link_sp_vel_to_link_force_grad_mat(r : RobotStruct, state : dict, force_order : int = 1, dim : int = 3) -> np.ndarray:
    n_ = dim_to_dof(dim) * force_order
    m_ = dim_to_dof(dim) * (force_order+2)
    mat = np.zeros((r.link_num * n_, r.link_num * m_))

    for i, link in enumerate(r.links):
        link_momentum = state_dict_to_cmvec(state, link.name, "link", "momentum", force_order)
        mat[i*n_:(i+1)*n_, i*m_:(i+1)*m_] = partial_link_sp_vel_to_force_grad_mat(link_momentum, force_order=force_order, dim=dim)
    return mat

def total_partial_link_sp_vel_to_link_force_grad_matvec(r : RobotStruct, state : dict, vec : np.ndarray, force_order : int = 1, dim : int = 3) -> np.ndarray:
    n_ = dim_to_dof(dim) * force_order
    m_ = dim_to_dof(dim) * (force_order+2)
    result = np.zeros(r.link_num * n_)

    for i, link in enumerate(r.links):
        out_start = i * n_
        in_start = i * m_
        link_momentum = state_dict_to_cmvec(state, link.name, "link", "momentum", force_order)
        mat = partial_link_sp_vel_to_force_grad_mat(link_momentum, force_order=force_order, dim=dim)
        result[out_start:out_start+n_] = mat @ vec[in_start:in_start+m_]
    return result

'''
    Gradients of joint force
'''
def total_partial_link_sp_vel_to_joint_force_grad_mat(r : RobotStruct, state : dict, force_order : int = 1, dim : int = 3) -> np.ndarray:
    n_ = dim_to_dof(dim) * force_order
    m_ = dim_to_dof(dim) * (force_order+2)
    mat = np.zeros((r.joint_num * n_, r.joint_num * m_))

    for i, joint in enumerate(r.joints):
        joint_momentum = state_dict_to_cmvec(state, joint.name, "joint", "momentum", force_order)
        mat[i*n_:(i+1)*n_, i*m_:(i+1)*m_] = partial_link_sp_vel_to_force_grad_mat(joint_momentum, force_order=force_order, dim=dim)
    return mat

def total_partial_link_sp_vel_to_joint_force_grad_matvec(r : RobotStruct, state : dict, vec : np.ndarray, force_order : int = 1, dim : int = 3) -> np.ndarray:
    n_ = dim_to_dof(dim) * force_order
    m_ = dim_to_dof(dim) * (force_order+2)
    result = np.zeros(r.joint_num * n_)

    for i, joint in enumerate(r.joints):
        out_start = i * n_
        in_start = i * m_
        joint_momentum = state_dict_to_cmvec(state, joint.name, "joint", "momentum", force_order)
        mat = partial_link_sp_vel_to_force_grad_mat(joint_momentum, force_order=force_order, dim=dim)
        result[out_start:out_start+n_] = mat @ vec[in_start:in_start+m_]
    return result

'''
    Gradients of joint momentum
'''

def total_partial_world_joint_momentum_to_joint_momentum_grad_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 3) -> np.ndarray:
    dof = dim_to_dof(dim)
    return total_factorial_mat(r.joint_num, order-1, dof) @  total_world_joint_cmtm_wrench_inv(r, state, order-1, dim) @ total_factorial_mat_inv(r.joint_num, order-1, dof)

def total_partial_world_joint_momentum_to_joint_momentum_grad_matvec(r : RobotStruct, state : dict, vec : np.ndarray, order : int = 3, dim : int = 3) -> np.ndarray:
    dof = dim_to_dof(dim)
    inv_fact_vec = total_factorial_mat_inv_vec(r.joint_num, vec, order-1, dof)
    cmtm_vec = total_world_joint_cmtm_wrench_inv_matvec(r, state, inv_fact_vec, order-1, dim)
    return total_factorial_matvec(r.joint_num, cmtm_vec, order-1, dof)

def total_partial_link_tan_vel_to_joint_momentum_grad_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 3) -> np.ndarray:
    dof = dim_to_dof(dim)
    total_local_link_momentum = extract_dict_total_link_cmvec(state, r.link_names, "momentum", order-1)
    total_world_joint_momentum = total_world_link_wrench_to_world_joint_wrench_mat(r, order-1, dim) @ total_world_link_cmtm_wrench(r, state, order-1, dim) @ total_local_link_momentum
    return total_factorial_mat(r.joint_num, order-1, dof) @ total_joint_cmtm_wrench_inv_var_x_arb_vec(r, state, total_world_joint_momentum, order-1, dim)

def total_partial_link_tan_vel_to_joint_momentum_grad_matvec(r : RobotStruct, state : dict, vec : np.ndarray, order : int = 3, dim : int = 3) -> np.ndarray:
    dof = dim_to_dof(dim)
    total_local_link_momentum = extract_dict_total_link_cmvec(state, r.link_names, "momentum", order-1)
    world_link_momentum = total_world_link_cmtm_wrench_matvec(r, state, total_local_link_momentum, order-1, dim)
    total_world_joint_momentum = total_world_link_wrench_to_world_joint_wrench_matvec(r, world_link_momentum, order-1, dim)
    cmtm_vec = total_joint_cmtm_wrench_inv_var_x_arb_vec_matvec(r, state, total_world_joint_momentum, vec, order-1, dim)
    return total_factorial_matvec(r.joint_num, cmtm_vec, order-1, dof)
