"""RoboKots robotics library.

The package exposes submodules ``core``, ``outward``, ``inward``, and
``robot_io``. They are intentionally not imported at the top level to avoid
pulling in optional heavy dependencies when they are not needed. Import the
modules you need directly, for example ``from robokots.inward import problem``.
"""

__all__ = [
    "core",
    "outward",
    "inward",
    "robot_io",
]
