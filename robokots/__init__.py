"""RoboKots robotics library.

The package exposes submodules ``core``, ``outward``, ``inward``, and
``robot_io``. They are intentionally not imported at the top level to avoid
pulling in optional heavy dependencies when they are not needed. Import the
modules you need directly, for example ``from robokots.outward import api``.
"""

__all__ = [
    "core",
    "outward",
    "inward",
    "robot_io",
    "PerturbationSpec",
    "PerturbationReport",
    "apply_perturbation",
    "NoiseSpec",
    "ParameterPerturbation",
]


def __getattr__(name):
    if name in {"PerturbationSpec", "PerturbationReport", "apply_perturbation", "NoiseSpec", "ParameterPerturbation"}:
        from . import perturbation
        return getattr(perturbation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
