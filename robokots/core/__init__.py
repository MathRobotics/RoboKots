"""Core robotics structures and utilities for RoboKots.

Submodules are intentionally not imported automatically so that optional
heavy dependencies are only loaded when requested. Import the specific
modules you need, for example ``from robokots.core import robot`` or
``from robokots.core.state.spec import StateType``.
"""

from importlib import import_module
from typing import Dict

# Map public attributes to their defining modules for lazy loading.
_lazy_attributes: Dict[str, str] = {
    # Robot structure definitions
    "RobotStruct": "robokots.core.robot",
    "JointStruct": "robokots.core.robot",
    "LinkStruct": "robokots.core.robot",
    "OutwardDataView": "robokots.core.state.protocol",
    "StateValueProvider": "robokots.core.state.protocol",
    "MotionLayoutOwner": "robokots.core.motion",
    "MotionTensor": "robokots.core.motion",
    "RobotMotions": "robokots.core.motion",
    # Axis-aware tensor adapters
    "AlgorithmSpec": "robokots.core.axis_tensor",
    "AxisTensor": "robokots.core.axis_tensor",
    "LayoutPolicy": "robokots.core.axis_tensor",
    "PhysicalLayout": "robokots.core.axis_tensor",
    "JacobianTensor": "robokots.core.state.tensor",
    "StateBatch": "robokots.core.state.batch",
    "StateTensor": "robokots.core.state.tensor",
}

__all__ = sorted(_lazy_attributes)


def __getattr__(name):
    """Lazily import attributes from submodules on first access.

    This keeps the initial import of :mod:`robokots.core` lightweight.
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
