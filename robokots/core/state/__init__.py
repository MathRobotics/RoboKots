"""State specifications, reader contracts, and backend-independent containers.

Concrete NumPy/mathrobo storage lives in ``robokots.outward.data``;
computation and cache management live outside this package.
"""

from importlib import import_module

_LAZY_ATTRIBUTES = {
    "StateType": ".spec",
    "OutwardDataView": ".protocol",
    "StateValueProvider": ".protocol",
    "StateTensor": ".tensor",
    "JacobianTensor": ".tensor",
    "StateBatch": ".batch",
}

__all__ = sorted(_LAZY_ATTRIBUTES)


def __getattr__(name):
    module_name = _LAZY_ATTRIBUTES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
