"""Dynamics helpers loaded on demand."""

from importlib import import_module

_LAZY_ATTRIBUTES = {
    "inertia": ".base",
    "spatial_inertia": ".base",
    "link_dynamics": ".dynamics",
    "joint_dynamics": ".dynamics",
    "joint_project_wrench": ".dynamics",
    "link_momentum_cmvec": ".dynamics",
    "link_force_cmvec": ".dynamics",
    "link_dynamics_cmvec": ".dynamics",
    "joint_dynamics_cmvec": ".dynamics",
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
