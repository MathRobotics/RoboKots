"""Kinematics helpers loaded on demand."""

from importlib import import_module

_LAZY_ATTRIBUTES = {
    "convert_joint_to_data": ".base",
    "convert_link_to_data": ".base",
    "joint_local_cmtm": ".kinematics",
    "joint_rel_cmtm": ".kinematics",
    "joint_rel_frame": ".kinematics",
    "joint_select_diag_mat": ".kinematics_matrix",
    "soft_link_local_cmtm": ".kinematics_soft_link",
    "calc_link_local_point_frame": ".kinematics_soft_link",
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
