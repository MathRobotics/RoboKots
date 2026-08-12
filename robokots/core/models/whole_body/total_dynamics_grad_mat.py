import numpy as np

from robokots.core import RobotStruct
from robokots.core.state import dim_to_dof

from .total_dynamics_mat import total_link_inertia_mat, total_joint_wrench_to_joint_torque_mat
from .total_dynamics_mat import total_world_link_wrench_to_world_joint_wrench_mat
from .total_dynamics_mat import total_link_inertia_matvec
from .total_kinematics_grad_mat import total_coord_to_joint_tan_vel_grad_mat, total_joint_tan_vel_to_link_sp_vel_grad_mat
from .total_kinematics_grad_mat import total_coord_to_link_tan_vel_grad_mat, total_coord_to_link_vel_grad_mat
from .total_kinematics_grad_mat import total_coord_to_link_sp_vel_grad_matvec

from .total_partial_grad_mat import (
    total_partial_link_momentum_to_world_link_momentum_grad_mat,
    total_partial_link_sp_vel_to_joint_force_grad_mat,
    total_partial_link_sp_vel_to_link_force_grad_mat,
    total_partial_link_tan_vel_to_joint_momentum_grad_mat,
    total_partial_link_tan_vel_to_world_link_momentum_grad_mat,
    total_partial_momentum_to_force_grad_mat,
    total_partial_world_joint_momentum_to_joint_momentum_grad_mat,
)
from .total_gravity_grad_mat import (
    state_gravity,
    total_coord_to_joint_gravity_force_grad_mat,
    total_coord_to_link_gravity_force_grad_mat,
)
from .topology_layout import (
    take_joint_child_link_blocks,
    take_joint_child_link_matrix_blocks,
)

def total_coord_to_link_momentum_grad_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 3) -> np.ndarray:
    return total_link_inertia_mat(r, order=order-1, dim=dim) @ total_joint_tan_vel_to_link_sp_vel_grad_mat(r, state, order, dim) @ total_coord_to_joint_tan_vel_grad_mat(r, state, order, dim)

def total_coord_to_link_momentum_grad_matvec(r : RobotStruct, state : dict, vec : np.ndarray, order : int = 3, dim : int = 3) -> np.ndarray:
    return total_link_inertia_matvec(r, total_coord_to_link_sp_vel_grad_matvec(r, state, vec, order, dim), order=order-1, dim=dim)

def total_coord_to_world_link_momentum_grad_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 3) -> np.ndarray:
    # dof = dim_to_dof(dim)
    # total_local_link_momentum = extract_dict_total_link_cmvec(state, r.link_names, "momentum", order-1)
    # j1 = total_world_link_cmtm_wrench(r, state, order-1, dim) @ total_factorial_mat_inv(r.link_num, order-1, dof) @ total_coord_to_link_momentum_grad_mat(r, state, order, dim)
    # j2 = total_link_cmtm_wrench_var_x_arb_vec(r, state, total_local_link_momentum, order-1, dim) \
    #     @ total_coord_to_link_tan_vel_grad_mat(r, state, order-1, dim) @ total_coord_arrange(r, out_order=order-1, in_order=order)
    # return total_factorial_mat(r.link_num, order-1, dof) @ (j1 + j2)
    return total_partial_link_momentum_to_world_link_momentum_grad_mat(r, state, order, dim) @ total_coord_to_link_momentum_grad_mat(r, state, order, dim) \
            + total_partial_link_tan_vel_to_world_link_momentum_grad_mat(r, state, order, dim) @ total_coord_to_link_tan_vel_grad_mat(r, state, out_order=order-1, in_order=order, dim=dim)

def total_coord_to_link_force_grad_mat(
    r : RobotStruct,
    state : dict,
    force_order : int = 1,
    dim : int = 3,
    gravity=None,
) -> np.ndarray:
    result = total_partial_momentum_to_force_grad_mat(r, state, force_order=force_order, dim=dim) @ total_coord_to_link_momentum_grad_mat(r, state, order=force_order+2, dim=dim) \
              + total_partial_link_sp_vel_to_link_force_grad_mat(r, state, force_order=force_order, dim=dim) @ total_coord_to_link_vel_grad_mat(r, state, order=force_order+2, dim=dim)
    gravity = state_gravity(state, gravity)
    if np.any(gravity):
        result = result + total_coord_to_link_gravity_force_grad_mat(
            r, state, gravity, force_order=force_order, dim=dim
        )
    return result


def total_coord_to_world_joint_momentum_grad_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 3) -> np.ndarray:
    return total_world_link_wrench_to_world_joint_wrench_mat(r, order=order-1, dim=dim) @ total_coord_to_world_link_momentum_grad_mat(r, state, order=order, dim=dim)

def total_coord_to_joint_momentum_grad_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 3) -> np.ndarray:
    dof = dim_to_dof(dim)
    j1 = total_partial_world_joint_momentum_to_joint_momentum_grad_mat(r, state, order, dim) @ total_coord_to_world_joint_momentum_grad_mat(r, state, order, dim)
    link_tan_grad = total_coord_to_link_tan_vel_grad_mat(
        r, state, out_order=order-1, in_order=order, dim=dim
    )
    child_link_tan_grad = take_joint_child_link_blocks(
        link_tan_grad, r, dof * (order - 1), axis=-2
    )
    j2 = total_partial_link_tan_vel_to_joint_momentum_grad_mat(r, state, order, dim) @ child_link_tan_grad
    return j1 + j2

def total_coord_to_joint_force_grad_mat(
    r : RobotStruct,
    state : dict,
    force_order : int = 1,
    dim : int = 3,
    gravity=None,
) -> np.ndarray:
    dof = dim_to_dof(dim)
    partial_momentum = take_joint_child_link_matrix_blocks(
        total_partial_momentum_to_force_grad_mat(r, state, force_order=force_order, dim=dim),
        r,
        dof * force_order,
        dof * (force_order + 1),
    )
    link_vel_grad = total_coord_to_link_vel_grad_mat(
        r, state, order=force_order+2, dim=dim
    )
    child_link_vel_grad = take_joint_child_link_blocks(
        link_vel_grad, r, dof * (force_order + 2), axis=-2
    )
    result = partial_momentum @ total_coord_to_joint_momentum_grad_mat(r, state, order=force_order+2, dim=dim) \
              + total_partial_link_sp_vel_to_joint_force_grad_mat(r, state, force_order=force_order, dim=dim) @ child_link_vel_grad
    gravity = state_gravity(state, gravity)
    if np.any(gravity):
        result = result + total_coord_to_joint_gravity_force_grad_mat(
            r, state, gravity, force_order=force_order, dim=dim
        )
    return result

def total_coord_to_joint_torque_grad_mat(
    r : RobotStruct,
    state : dict,
    torque_order : int = 1,
    dim : int = 3,
    gravity=None,
) -> np.ndarray:
    return total_joint_wrench_to_joint_torque_mat(
        r, torque_order, dim=dim
    ) @ total_coord_to_joint_force_grad_mat(
        r, state, torque_order, dim=dim, gravity=gravity
    )
