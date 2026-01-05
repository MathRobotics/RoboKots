"""Core robotics structures and utilities for RoboKots.

Submodules are intentionally not imported automatically so that optional
heavy dependencies are only loaded when requested. Import the specific
modules you need, for example ``from robokots.core import robot`` or
``from robokots.core.state import StateType``.
"""

from importlib import import_module
from typing import Dict

# Map public attributes to their defining modules for lazy loading.
_lazy_attributes: Dict[str, str] = {
    # Robot structure definitions
    "RobotStruct": "robokots.core.robot",
    "JointStruct": "robokots.core.robot",
    "LinkStruct": "robokots.core.robot",
    # State dictionary helpers
    "state_dict_to_cmtm": "robokots.core.state_dict",
    "state_dict_to_cmtm_wrench": "robokots.core.state_dict",
    "state_dict_to_cmvec": "robokots.core.state_dict",
    "state_dict_to_rel_cmtm_wrench": "robokots.core.state_dict",
    "extract_dict_link_info": "robokots.core.state_dict",
    "extract_dict_info": "robokots.core.state_dict",
    "vecs_to_state_dict": "robokots.core.state_dict",
    "cmtm_to_state_list": "robokots.core.state_dict",
    "state_dict_to_frame": "robokots.core.state_dict",
    "state_dict_to_vecs": "robokots.core.state_dict",
    # State cache utility
    "StateCache": "robokots.core.state_cache",
}

__all__ = sorted(_lazy_attributes)


def __getattr__(name):
    """Lazily import attributes from submodules on first access.

    This keeps the initial import of :mod:`robokots.core` lightweight while
    preserving the previous attribute-based API for commonly used symbols.
    """

    module_name = _lazy_attributes.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
