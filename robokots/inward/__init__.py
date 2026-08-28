"""Inward algorithms: solve generalized accelerations from applied effort.

Unlike :mod:`robokots.outward`, which expands a prescribed motion into link
states and inverse dynamics, this namespace owns solvers whose primary flow
is from generalized effort back to generalized acceleration.
"""

from .cache import InwardCache

__all__ = ["dynamics", "InwardCache"]
