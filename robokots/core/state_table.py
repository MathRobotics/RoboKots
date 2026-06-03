"""Deprecated compatibility wrapper for Polars-backed state tables.

The DataFrame implementation lives in :mod:`robokots.contrib.polars` so the
core RoboKots package can stay independent of table-processing dependencies.
"""

from ..contrib.polars.state_table import RobotDF, RobotState

__all__ = ["RobotDF", "RobotState"]
