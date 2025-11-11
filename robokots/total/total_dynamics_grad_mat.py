import numpy as np
from mathrobo import SE3wrench

from ..basic.robot import RobotStruct
from ..basic.state_dict import extract_dict_total_link_cmvec, state_dict_to_cmtm
from ..dynamics.base import spatial_inertia
from ..dynamics.dynamics_matrix import link_to_force_tan_map_mat

from .basic import total_factorial_mat, total_factorial_mat_inv
from .total_kinematics_mat import total_cmtm_hat_commute, total_coord_arrange
from .total_dynamics_mat import total_link_inertia_mat, total_link_wrench_to_joint_wrench_mat, total_world_link_cmtm_wrench
from .total_kinematics_grad_mat import total_coord_to_joint_tan_vel_grad_mat, total_joint_tan_vel_to_link_sp_vel_grad_mat, total_coord_to_link_vel_grad_mat

def total_coord_to_link_momentum_grad_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 6) -> np.ndarray:
    return total_link_inertia_mat(r, order=order-1, dim=dim) @ total_joint_tan_vel_to_link_sp_vel_grad_mat(r, state, order, dim) @ total_coord_to_joint_tan_vel_grad_mat(r, state, order, dim)

def total_coord_to_world_link_momentum_grad_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 6) -> np.ndarray:
    total_local_link_momentum = extract_dict_total_link_cmvec(state, r.link_names, "link_momentum", order-1)
    j1 = total_world_link_cmtm_wrench(r, state, order-1, dim) @ total_coord_to_link_momentum_grad_mat(r, state, order, dim)
    j2 = total_world_link_cmtm_wrench(r, state, order-1, dim) @ total_cmtm_hat_commute(total_local_link_momentum, SE3wrench, num=r.link_num, order=order-1, dim=dim)\
        @ total_coord_to_link_vel_grad_mat(r, state, order-1, dim) @ total_coord_arrange(r, out_order=order-1, in_order=order, dim=dim)
    return j1 + j2

def total_link_sp_vel_to_link_force_grad_mat(r : RobotStruct, state : dict, force_order : int = 1, dim : int = 6) -> np.ndarray:
    n_ = dim * force_order
    m_ = dim * (force_order+2)
    mat = np.zeros((r.link_num * n_, r.link_num * m_))

    for i, link in enumerate(r.links):
        cmtm = state_dict_to_cmtm(state, link.name, force_order+1)
        mat[i*n_:(i+1)*n_, i*m_:(i+1)*m_] = link_to_force_tan_map_mat(cmtm, spatial_inertia(link.mass, link.inertia, link.cog), force_order=force_order, dim=dim)
    return mat

def total_coord_to_link_force_grad_mat(r : RobotStruct, state : dict, force_order : int = 1, dim : int = 6) -> np.ndarray:
    return total_link_sp_vel_to_link_force_grad_mat(r, state, force_order=force_order, dim=dim) @ total_coord_to_link_vel_grad_mat(r, state, force_order+2, dim)

def total_coord_to_joint_momentum_grad_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 6) -> np.ndarray:
    tf_j_m = total_factorial_mat(r.joint_num, order, dim)
    tf_l_m_inv = total_factorial_mat_inv(r.link_num, order, dim)

    j1 = tf_j_m @ total_link_wrench_to_joint_wrench_mat(r, state, order, dim) @ tf_l_m_inv \
         @ total_coord_to_link_momentum_grad_mat(r, state, order+1, dim)
    return j1

