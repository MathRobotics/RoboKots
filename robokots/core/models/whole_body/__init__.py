"""Whole-body matrix and gradient helpers.

The heavy submodules are loaded lazily so importing
``robokots.core.models.whole_body`` does not pull in the entire dynamics stack.
"""

from importlib import import_module

_LAZY_ATTRIBUTES = {
    # basic
    "total_factorial_mat": ".basic",
    "total_factorial_matvec": ".basic",
    "total_factorial_mat_inv": ".basic",
    "total_factorial_mat_inv_vec": ".basic",
    "total_link_cmtm_var_x_arb_vec": ".basic",
    "total_joint_cmtm_var_x_arb_vec": ".basic",
    "total_link_cmtm_wrench_var_x_arb_vec": ".basic",
    "total_link_cmtm_wrench_var_x_arb_vec_matvec": ".basic",
    "total_joint_cmtm_wrench_var_x_arb_vec": ".basic",
    "total_joint_cmtm_wrench_inv_var_x_arb_vec": ".basic",
    "total_joint_cmtm_wrench_inv_var_x_arb_vec_matvec": ".basic",
    # total_kinematics_mat
    "total_coord_arrange": ".total_kinematics_mat",
    "total_cmtm_hat": ".total_kinematics_mat",
    "total_cmtm_hat_commute": ".total_kinematics_mat",
    "total_world_link_cmtm": ".total_kinematics_mat",
    "total_world_link_cmtm_inv": ".total_kinematics_mat",
    "total_world_joint_cmtm": ".total_kinematics_mat",
    "total_world_joint_cmtm_inv": ".total_kinematics_mat",
    "total_link_vel_to_joint_vel_mat": ".total_kinematics_mat",
    "total_joint_vel_to_link_vel_mat": ".total_kinematics_mat",
    "total_coord_to_joint_vel_mat": ".total_kinematics_mat",
    "total_coord_to_link_vel_mat": ".total_kinematics_mat",
    # total_dynamics_mat
    "total_joint_wrench_to_joint_torque_mat": ".total_dynamics_mat",
    "total_joint_wrench_to_joint_torque_matvec": ".total_dynamics_mat",
    "total_world_link_cmtm_wrench": ".total_dynamics_mat",
    "total_world_link_cmtm_wrench_matvec": ".total_dynamics_mat",
    "total_world_link_cmtm_wrench_inv": ".total_dynamics_mat",
    "total_world_joint_cmtm_wrench": ".total_dynamics_mat",
    "total_world_joint_cmtm_wrench_inv": ".total_dynamics_mat",
    "total_world_joint_cmtm_wrench_inv_matvec": ".total_dynamics_mat",
    "total_joint_wrench_to_link_wrench_mat": ".total_dynamics_mat",
    "total_link_wrench_to_joint_wrench_mat": ".total_dynamics_mat",
    "total_world_joint_wrench_to_world_link_wrench_mat": ".total_dynamics_mat",
    "total_world_link_wrench_to_world_joint_wrench_mat": ".total_dynamics_mat",
    "total_world_link_wrench_to_world_joint_wrench_matvec": ".total_dynamics_mat",
    "total_link_inertia_mat": ".total_dynamics_mat",
    "total_link_inertia_matvec": ".total_dynamics_mat",
    "total_momentum_to_force_mat": ".total_dynamics_mat",
    "total_coord_to_link_momentum_mat": ".total_dynamics_mat",
    "total_coord_to_joint_momentum_mat": ".total_dynamics_mat",
    "total_coord_to_link_force_mat": ".total_dynamics_mat",
    "total_coord_to_joint_force_mat": ".total_dynamics_mat",
    # total_kinematics_grad_mat
    "total_coord_arrange_vec": ".total_kinematics_grad_mat",
    "total_joint_tan_vel_to_link_tan_vel_grad_mat": ".total_kinematics_grad_mat",
    "total_joint_tan_vel_to_link_tan_vel_grad_matvec": ".total_kinematics_grad_mat",
    "total_joint_tan_vel_to_link_vel_grad_mat": ".total_kinematics_grad_mat",
    "total_joint_tan_vel_to_link_vel_grad_matvec": ".total_kinematics_grad_mat",
    "total_joint_tan_vel_to_link_sp_vel_grad_mat": ".total_kinematics_grad_mat",
    "total_joint_tan_vel_to_link_sp_vel_grad_matvec": ".total_kinematics_grad_mat",
    "total_coord_to_joint_tan_vel_grad_mat": ".total_kinematics_grad_mat",
    "total_coord_to_joint_tan_vel_grad_matvec": ".total_kinematics_grad_mat",
    "total_coord_to_link_tan_vel_grad_mat": ".total_kinematics_grad_mat",
    "total_coord_to_link_tan_vel_grad_matvec": ".total_kinematics_grad_mat",
    "total_coord_to_link_vel_grad_mat": ".total_kinematics_grad_mat",
    "total_coord_to_link_vel_grad_matvec": ".total_kinematics_grad_mat",
    "total_coord_to_link_sp_vel_grad_matvec": ".total_kinematics_grad_mat",
    # total_dynamics_grad_mat
    "total_coord_to_link_momentum_grad_mat": ".total_dynamics_grad_mat",
    "total_coord_to_link_momentum_grad_matvec": ".total_dynamics_grad_mat",
    "total_coord_to_world_link_momentum_grad_mat": ".total_dynamics_grad_mat",
    "total_coord_to_link_force_grad_mat": ".total_dynamics_grad_mat",
    "total_link_gravity_force": ".total_gravity_grad_mat",
    "total_coord_to_link_gravity_force_grad_mat": ".total_gravity_grad_mat",
    "total_coord_to_joint_gravity_force_grad_mat": ".total_gravity_grad_mat",
    "total_coord_to_world_joint_momentum_grad_mat": ".total_dynamics_grad_mat",
    "total_coord_to_joint_momentum_grad_mat": ".total_dynamics_grad_mat",
    "total_coord_to_joint_force_grad_mat": ".total_dynamics_grad_mat",
    "total_coord_to_joint_torque_grad_mat": ".total_dynamics_grad_mat",
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
