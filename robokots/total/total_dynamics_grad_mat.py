from os import link
import numpy as np
from mathrobo import SE3wrench, CMVector

from ..basic.robot import RobotStruct
from ..basic.state import dim_to_dof
from ..basic.state_dict import extract_dict_total_link_cmvec

from .basic import total_factorial_mat, total_factorial_mat_inv
from .basic import total_link_cmtm_wrench_var_x_arb_vec, total_joint_cmtm_wrench_inv_var_x_arb_vec
from .total_kinematics_mat import total_coord_arrange
from .total_dynamics_mat import total_link_inertia_mat, total_joint_wrench_to_joint_torque_mat
from .total_dynamics_mat import total_world_link_cmtm_wrench, total_world_joint_cmtm_wrench_inv
from .total_dynamics_mat import total_world_link_wrench_to_world_joint_wrench_mat
from .total_kinematics_grad_mat import total_coord_to_joint_tan_vel_grad_mat, total_joint_tan_vel_to_link_sp_vel_grad_mat
from .total_kinematics_grad_mat import total_coord_to_link_tan_vel_grad_mat, total_coord_to_link_vel_grad_mat

from .total_partial_grad_mat import *

def total_coord_to_link_momentum_grad_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 3) -> np.ndarray:
    return total_link_inertia_mat(r, order=order-1, dim=dim) @ total_joint_tan_vel_to_link_sp_vel_grad_mat(r, state, order, dim) @ total_coord_to_joint_tan_vel_grad_mat(r, state, order, dim)

def total_coord_to_world_link_momentum_grad_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 3) -> np.ndarray:
    # dof = dim_to_dof(dim)
    # total_local_link_momentum = extract_dict_total_link_cmvec(state, r.link_names, "momentum", order-1)
    # j1 = total_world_link_cmtm_wrench(r, state, order-1, dim) @ total_factorial_mat_inv(r.link_num, order-1, dof) @ total_coord_to_link_momentum_grad_mat(r, state, order, dim)
    # j2 = total_link_cmtm_wrench_var_x_arb_vec(r, state, total_local_link_momentum, order-1, dim) \
    #     @ total_coord_to_link_tan_vel_grad_mat(r, state, order-1, dim) @ total_coord_arrange(r, out_order=order-1, in_order=order)
    # return total_factorial_mat(r.link_num, order-1, dof) @ (j1 + j2)
    return total_partial_link_momentum_to_world_link_momentum_grad_mat(r, state, order, dim) @ total_coord_to_link_momentum_grad_mat(r, state, order, dim) \
            + total_partial_link_tan_vel_to_world_link_momentum_grad_mat(r, state, order, dim) @ total_coord_to_link_tan_vel_grad_mat(r, state, out_order=order-1, in_order=order, dim=dim)

def total_coord_to_link_force_grad_mat(r : RobotStruct, state : dict, force_order : int = 1, dim : int = 3) -> np.ndarray:
    return total_partial_momentum_to_force_grad_mat(r, state, force_order=force_order, dim=dim) @ total_coord_to_link_momentum_grad_mat(r, state, order=force_order+2, dim=dim) \
              + total_partial_link_sp_vel_to_link_force_grad_mat(r, state, force_order=force_order, dim=dim) @ total_coord_to_link_vel_grad_mat(r, state, order=force_order+2, dim=dim)

def total_coord_to_world_joint_momentum_grad_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 3) -> np.ndarray:
    return total_world_link_wrench_to_world_joint_wrench_mat(r, order=order-1, dim=dim) @ total_coord_to_world_link_momentum_grad_mat(r, state, order=order, dim=dim)

def total_coord_to_joint_momentum_grad_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 3) -> np.ndarray:
    # dof = dim_to_dof(dim)
    # total_local_link_momentum = extract_dict_total_link_cmvec(state, r.link_names, "momentum", order-1)
    # total_world_joint_momentum = total_world_link_wrench_to_world_joint_wrench_mat(r, order-1, dim) @ total_world_link_cmtm_wrench(r, state, order-1, dim) @ total_local_link_momentum
    
    # j1 = total_world_joint_cmtm_wrench_inv(r, state, order-1, dim) @ total_factorial_mat_inv(r.joint_num, order-1, dof) @ total_coord_to_world_joint_momentum_grad_mat(r, state, order, dim=dim)
    # j2 = total_joint_cmtm_wrench_inv_var_x_arb_vec(r, state, total_world_joint_momentum, order-1, dim) \
    #       @ total_coord_to_link_tan_vel_grad_mat(r, state, order-1, dim=dim)[(order-1)*dof:] @ total_coord_arrange(r, out_order=order-1, in_order=order)

    # return total_factorial_mat(r.joint_num, order-1, submat_dim=dof) @ (j1 + j2)
    j1 = total_partial_world_joint_momentum_to_joint_momentum_grad_mat(r, state, order, dim) @ total_coord_to_world_joint_momentum_grad_mat(r, state, order, dim)
    j2 = total_partial_link_tan_vel_to_joint_momentum_grad_mat(r, state, order, dim) @ total_coord_to_link_tan_vel_grad_mat(r, state, out_order=order-1, in_order=order, dim=dim)[(order-1)*dim_to_dof(dim):]
    return j1 + j2

def total_coord_to_joint_force_grad_mat(r : RobotStruct, state : dict, force_order : int = 1, dim : int = 3) -> np.ndarray:
    dof = dim_to_dof(dim)
    return total_partial_momentum_to_force_grad_mat(r, state, force_order=force_order, dim=dim)[(force_order)*dof:,(force_order+1)*dof:] @ total_coord_to_joint_momentum_grad_mat(r, state, order=force_order+2, dim=dim) \
              + total_partial_link_sp_vel_to_joint_force_grad_mat(r, state, force_order=force_order, dim=dim) @ total_coord_to_link_vel_grad_mat(r, state, order=force_order+2, dim=dim)[(force_order+2)*dof:]

def total_coord_to_joint_torque_grad_mat(r : RobotStruct, state : dict, torque_order : int = 1, dim : int = 3) -> np.ndarray:
    return total_joint_wrench_to_joint_torque_mat(r, torque_order, dim=dim) @ total_coord_to_joint_force_grad_mat(r, state, torque_order, dim=dim)