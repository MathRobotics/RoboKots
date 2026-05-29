# outward/api.py
from importlib import import_module

from .state import (
    build_kinematics_outward_state,
    build_kinematics_state,
    build_dynamics_outward_state,
    build_dynamics_cmtm_state,
    get_value,
)
from .state import calc_link_total_point_frame

from .values import compute_outward_value, update_outward_state

from .diff.numerical_diff import (
    link_diff_kinematics_numerical,
    diff_outward_numerical,
)

from .diff.outward_total_gradient import outward_jacobian, outward_jacobian_matvec
from .diff.outward_jacobians import jacobian_numerical

_LAZY_API = {
    "build_kinematics_state_jax": ".diff.outward_jax",
    "kinematics_jax": ".diff.outward_jax",
}


def __getattr__(name):
    module_name = _LAZY_API.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name, __package__)
    value = getattr(module, name)
    globals()[name] = value
    return value

__all__ = [
    "build_kinematics_state",
    "build_kinematics_outward_state",
    "build_dynamics_outward_state",
    "build_dynamics_cmtm_state",
    "build_kinematics_state_jax",
    "kinematics_jax",
    "get_value",
    "compute_outward_value",
    "update_outward_state",
    "link_diff_kinematics_numerical",
    "diff_outward_numerical",
    "outward_jacobian",
    "outward_jacobian_matvec",
    "jacobian_numerical",
    "calc_link_total_point_frame",
]
