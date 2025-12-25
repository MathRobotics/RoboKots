import numpy as np

from ..robot import RobotStruct
from ..state import dim_to_dof
from ..state_dict import state_dict_to_cmtm, state_dict_to_rel_cmtm
from ..kinematics.kinematics_matrix import joint_select_diag_mat

from .total_kinematics_mat import total_coord_arrange

def total_joint_tan_vel_to_link_tan_vel_grad_mat(r : RobotStruct, state : dict, order : int = 1, dim : int = 3) -> np.ndarray:
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

def total_joint_tan_vel_to_link_vel_grad_mat(r : RobotStruct, state : dict, order : int = 1, dim : int = 3) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    mat = np.zeros((r.link_num * n_, r.joint_num * n_))

    for i, link in enumerate(r.links):
        link_route = []
        joint_route = []
        r.route_target_link(link, link_route, joint_route)
        for j in joint_route:
            joint = r.joints[j]
            rel_cmtm = state_dict_to_rel_cmtm(state, link.name, r.links[joint.child_link_id].name, "link", order)
            link_cmtm = state_dict_to_cmtm(state, link.name, "link", order)
            mat[i*n_:(i+1)*n_, j*n_:(j+1)*n_] = link_cmtm.tangent_mat_inv() @ rel_cmtm.mat_adj()
    return mat

def total_joint_tan_vel_to_link_sp_vel_grad_mat(r : RobotStruct, state : dict, order : int = 1, dim : int = 3) -> np.ndarray:
    n_j = dim_to_dof(dim) * order
    n_l = dim_to_dof(dim) * (order-1)
    mat = np.zeros((r.link_num * n_l, r.joint_num * n_j))

    for i, link in enumerate(r.links):
        link_route = []
        joint_route = []
        r.route_target_link(link, link_route, joint_route)
        for j in joint_route:
            joint = r.joints[j]
            rel_cmtm = state_dict_to_rel_cmtm(state, link.name, r.links[joint.child_link_id].name, "link", order)
            link_cmtm = state_dict_to_cmtm(state, link.name, "link", order)
            mat[i*n_l:(i+1)*n_l, j*n_j:(j+1)*n_j] = link_cmtm.tangent_mat_inv()[dim_to_dof(dim):] @ rel_cmtm.mat_adj()
    return mat

def total_coord_to_joint_tan_vel_grad_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 3) -> np.ndarray:
    n_ = dim_to_dof(dim) * order
    mat = np.zeros((r.joint_num * n_, r.joint_dof * order))

    for i, joint in enumerate(r.joints):
        if joint.dof == 0:  # Joint with no degree of freedom
            continue
        joint_cmtm = state_dict_to_cmtm(state, joint.name, "joint", order)
        mat[i*n_:(i+1)*n_, joint.dof_index*order:(joint.dof_index+joint.dof)*order] = \
            joint_cmtm.tangent_mat() @ joint_select_diag_mat(joint.select_mat, order)

    return mat

# def total_coord_to_joint_tan_vel_grad_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 3) -> np.ndarray:
#     n_ = dim_to_dof(dim) * (order-1)
#     mat = np.zeros((r.joint_num * n_, r.joint_dof * order))

#     for i, joint in enumerate(r.joints):
#         if joint.dof == 0:  # Joint with no degree of freedom
#             continue
#         joint_cmtm = state_dict_to_cmtm(state, joint.name, order)
#         mat[i*n_:(i+1)*n_, joint.dof_index*order:(joint.dof_index+joint.dof)*order] = \
#             joint_cmtm.tangent_mat()[dim_to_dof(dim):] @ joint_select_diag_mat(joint.select_mat, order)

#     return mat

def total_coord_to_link_tan_vel_grad_mat(r : RobotStruct, state : dict, out_order : int = 3, in_order = None, dim : int = 3) -> np.ndarray:
    if in_order is None:
        return total_joint_tan_vel_to_link_tan_vel_grad_mat(r, state, out_order, dim) @ total_coord_to_joint_tan_vel_grad_mat(r, state, out_order, dim)
    else:
        return total_joint_tan_vel_to_link_tan_vel_grad_mat(r, state, out_order, dim) @ total_coord_to_joint_tan_vel_grad_mat(r, state, out_order, dim) \
               @ total_coord_arrange(r, out_order=out_order, in_order=in_order)

def total_coord_to_link_vel_grad_mat(r : RobotStruct, state : dict, order : int = 3, dim : int = 3) -> np.ndarray:
    return total_joint_tan_vel_to_link_vel_grad_mat(r, state, order, dim) @ total_coord_to_joint_tan_vel_grad_mat(r, state, order, dim)