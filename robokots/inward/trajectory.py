from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol
import numpy as np

from robokots.core.time_grid import TimeGrid

Array = np.ndarray


class Trajectory(Protocol):
    """Interface for mapping trajectory parameters to per-step joint vectors.

    Implementations should provide:
      - num_params(): total parameter dimension
      - q_at(params, k): q_k for time index k
      - dqdp_at(params, k): dq_k / d params, shape (dof, num_params)
    """

    name: str
    dof: int
    time: TimeGrid

    def num_params(self) -> int: ...

    def q_at(self, params: Array, k: int) -> Array: ...

    def dqdp_at(self, params: Array, k: int) -> Array: ...


@dataclass(frozen=True)
class GridTrajectory:
    """Direct per-step trajectory.

    Parameters are q_0, q_1, ..., q_N concatenated.
    This is the simplest layout and the default for the "grid" type.
    """

    dof: int
    time: TimeGrid
    name: str = "grid"

    def num_params(self) -> int:
        return int(self.dof * (self.time.N + 1))

    def q_at(self, params: Array, k: int) -> Array:
        params = np.asarray(params, dtype=float).reshape(-1)
        if k < 0 or k > self.time.N:
            raise ValueError(f"GridTrajectory: k={k} out of range [0, {self.time.N}]")
        start = k * self.dof
        return params[start : start + self.dof]

    def dqdp_at(self, params: Array, k: int) -> Array:
        _ = params  # unused, but kept for signature symmetry
        n = self.num_params()
        J = np.zeros((self.dof, n), dtype=float)
        start = k * self.dof
        J[:, start : start + self.dof] = np.eye(self.dof)
        return J


@dataclass(frozen=True)
class BsplineTrajectory:
    """B-spline trajectory placeholder.

    Swap-in point:
      - implement basis evaluation to compute q_k and dq_k/dp
      - keep interface identical to GridTrajectory
    """

    dof: int
    time: TimeGrid
    degree: int
    num_ctrl: int
    name: str = "bspline"

    def num_params(self) -> int:
        return int(self.dof * self.num_ctrl)

    def q_at(self, params: Array, k: int) -> Array:
        raise NotImplementedError("BsplineTrajectory.q_at is not implemented yet.")

    def dqdp_at(self, params: Array, k: int) -> Array:
        raise NotImplementedError("BsplineTrajectory.dqdp_at is not implemented yet.")


def build_trajectory(spec: Optional[dict], *, dof: int, time: TimeGrid) -> Trajectory:
    """Factory for trajectory implementations.

    Supported types:
      - "grid": direct per-step q_k parameters
      - "bspline": placeholder (NotImplemented)
    """
    if spec is None:
        return GridTrajectory(dof=dof, time=time)

    typ = str(spec.get("type", "grid")).lower()
    if typ in {"grid", "time_grid", "q_grid"}:
        return GridTrajectory(dof=dof, time=time)
    if typ in {"bspline", "b-spline"}:
        degree = int(spec.get("degree", 3))
        num_ctrl = int(spec.get("num_ctrl", 0))
        if num_ctrl <= 0:
            raise ValueError("BsplineTrajectory requires positive 'num_ctrl'.")
        return BsplineTrajectory(dof=dof, time=time, degree=degree, num_ctrl=num_ctrl)

    raise ValueError(f"Unknown trajectory type: {spec.get('type')}")
