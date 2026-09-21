"""Whole-body matrix and gradient helpers.

The heavy submodules are loaded lazily so importing
``robokots.core.kernels.whole_body`` does not pull in the entire dynamics stack.
"""

from importlib import import_module

_LAZY_ATTRIBUTES = {
    # operators
    "total_factorial_mat": ".operators",
    "total_factorial_matvec": ".operators",
    "total_factorial_mat_inv": ".operators",
    "total_factorial_mat_inv_vec": ".operators",
    "total_link_cmtm_var_x_arb_vec": ".operators",
    "total_joint_cmtm_var_x_arb_vec": ".operators",
    "total_link_cmtm_wrench_var_x_arb_vec": ".operators",
    "total_link_cmtm_wrench_var_x_arb_vec_matvec": ".operators",
    "total_joint_cmtm_wrench_var_x_arb_vec": ".operators",
    "total_joint_cmtm_wrench_inv_var_x_arb_vec": ".operators",
    "total_joint_cmtm_wrench_inv_var_x_arb_vec_matvec": ".operators",
    # kinematics
    "total_coord_arrange": ".kinematics",
    "total_cmtm_hat": ".kinematics",
    "total_cmtm_hat_commute": ".kinematics",
    "total_world_link_cmtm": ".kinematics",
    "total_world_link_cmtm_inv": ".kinematics",
    "total_world_joint_cmtm": ".kinematics",
    "total_world_joint_cmtm_inv": ".kinematics",
    "total_link_vel_to_joint_vel_mat": ".kinematics",
    "total_joint_vel_to_link_vel_mat": ".kinematics",
    "total_coord_to_joint_vel_mat": ".kinematics",
    "total_coord_to_link_vel_mat": ".kinematics",
    # dynamics
    "total_joint_wrench_to_joint_torque_mat": ".dynamics",
    "total_joint_wrench_to_joint_torque_matvec": ".dynamics",
    "total_world_link_cmtm_wrench": ".dynamics",
    "total_world_link_cmtm_wrench_matvec": ".dynamics",
    "total_world_link_cmtm_wrench_inv": ".dynamics",
    "total_world_joint_cmtm_wrench": ".dynamics",
    "total_world_joint_cmtm_wrench_inv": ".dynamics",
    "total_world_joint_cmtm_wrench_inv_matvec": ".dynamics",
    "total_joint_wrench_to_link_wrench_mat": ".dynamics",
    "total_link_wrench_to_joint_wrench_mat": ".dynamics",
    "total_world_joint_wrench_to_world_link_wrench_mat": ".dynamics",
    "total_world_link_wrench_to_world_joint_wrench_mat": ".dynamics",
    "total_world_link_wrench_to_world_joint_wrench_matvec": ".dynamics",
    "total_link_inertia_mat": ".dynamics",
    "total_link_inertia_matvec": ".dynamics",
    "total_momentum_to_force_mat": ".dynamics",
    "total_coord_to_link_momentum_mat": ".dynamics",
    "total_coord_to_joint_momentum_mat": ".dynamics",
    "total_coord_to_link_force_mat": ".dynamics",
    "total_coord_to_joint_force_mat": ".dynamics",
    # kinematics_derivatives
    "total_coord_arrange_vec": ".kinematics_derivatives",
    "total_joint_tan_vel_to_link_tan_vel_grad_mat": ".kinematics_derivatives",
    "total_joint_tan_vel_to_link_tan_vel_grad_matvec": ".kinematics_derivatives",
    "total_joint_tan_vel_to_link_vel_grad_mat": ".kinematics_derivatives",
    "total_joint_tan_vel_to_link_vel_grad_matvec": ".kinematics_derivatives",
    "total_joint_tan_vel_to_link_sp_vel_grad_mat": ".kinematics_derivatives",
    "total_joint_tan_vel_to_link_sp_vel_grad_matvec": ".kinematics_derivatives",
    "total_coord_to_joint_tan_vel_grad_mat": ".kinematics_derivatives",
    "total_coord_to_joint_tan_vel_grad_matvec": ".kinematics_derivatives",
    "total_coord_to_link_tan_vel_grad_mat": ".kinematics_derivatives",
    "total_coord_to_link_tan_vel_grad_matvec": ".kinematics_derivatives",
    "total_coord_to_link_vel_grad_mat": ".kinematics_derivatives",
    "total_coord_to_link_vel_grad_matvec": ".kinematics_derivatives",
    "total_coord_to_link_sp_vel_grad_matvec": ".kinematics_derivatives",
    # dynamics_derivatives
    "total_coord_to_link_momentum_grad_mat": ".dynamics_derivatives",
    "total_coord_to_link_momentum_grad_matvec": ".dynamics_derivatives",
    "total_coord_to_world_link_momentum_grad_mat": ".dynamics_derivatives",
    "total_coord_to_link_force_grad_mat": ".dynamics_derivatives",
    "total_link_gravity_force": ".gravity_derivatives",
    "total_coord_to_link_gravity_force_grad_mat": ".gravity_derivatives",
    "total_coord_to_joint_gravity_force_grad_mat": ".gravity_derivatives",
    "total_coord_to_world_joint_momentum_grad_mat": ".dynamics_derivatives",
    "total_coord_to_joint_momentum_grad_mat": ".dynamics_derivatives",
    "total_coord_to_joint_force_grad_mat": ".dynamics_derivatives",
    "total_coord_to_joint_torque_grad_mat": ".dynamics_derivatives",
}

__all__ = sorted(_LAZY_ATTRIBUTES)


def __getattr__(name):
    module_name = _LAZY_ATTRIBUTES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
