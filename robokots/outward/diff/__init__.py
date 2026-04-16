"""Differentiation utilities for outward computations.

This package intentionally avoids importing submodules eagerly so that local
checkouts do not get mixed with an installed ``robokots`` namespace package and
optional heavy dependencies remain deferred until requested.
"""

__all__ = [
    "numerical_diff",
    "outward_jacobians",
    "outward_jax",
    "outward_total_gradient",
]
