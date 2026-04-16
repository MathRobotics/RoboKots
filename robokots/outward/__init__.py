"""Outward computations for RoboKots.

Submodules are not imported automatically to keep optional dependencies light. A
small set of public APIs is exposed lazily for compatibility with existing
imports such as ``from robokots.outward import build_kinematics_state``. Other
symbols can be accessed by importing the desired submodule directly, e.g.
``robokots.outward.state``.
"""

from importlib import import_module

_LAZY_API = {
    "build_kinematics_state": "build_kinematics_state",
    "build_dynamics_cmtm_state": "build_dynamics_cmtm_state",
    "build_kinematics_state_jax": "build_kinematics_state_jax",
    "kinematics_jax": "kinematics_jax",
    "get_value": "get_value",
    "compute_outward_value": "compute_outward_value",
    "update_outward_state": "update_outward_state",
    "link_diff_kinematics_numerical": "link_diff_kinematics_numerical",
    "diff_outward_numerical": "diff_outward_numerical",
    "outward_jacobian": "outward_jacobian",
    "jacobian_numerical": "jacobian_numerical",
    "calc_link_total_point_frame": "calc_link_total_point_frame",
}


def __getattr__(name):
    if name in _LAZY_API:
        api = import_module("robokots.outward.api")
        return getattr(api, name)
    raise AttributeError(f"module 'robokots.outward' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_API))


__all__ = [
    "api",
    "diff",
    "state",
    "values",
    *_LAZY_API.keys(),
]
