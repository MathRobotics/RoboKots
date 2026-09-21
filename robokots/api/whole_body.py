"""Whole-body physical quantities derived from the current motion."""
from __future__ import annotations

import numpy as np

from .. import outward as outward_api
from ..core.motion import RobotMotions
from ..core.state.spec import StateType


class WholeBodyMixin:
  """Center-of-mass and whole-body angular-momentum convenience APIs.

  These quantities are evaluated on demand.  They are deliberately not stored
  as regular outward states because they describe the complete robot, rather
  than a link or joint.
  """

  def _total_mass(self) -> float:
    mass = float(sum(link.mass for link in self.robot_.links))
    if not np.isfinite(mass) or mass <= 0.0:
      raise ValueError("center of mass requires a finite, positive total link mass")
    return mass

  def _center_of_mass_from_motion(self, motion: np.ndarray) -> np.ndarray:
    state = outward_api.build_kinematics_outward_state(self.robot_, motion, order=1)
    total_mass = self._total_mass()
    result = None
    for link in self.robot_.links:
      if link.mass == 0.0:
        continue
      mat = np.asarray(state.link_cmtm[link.name].elem_mat())
      offset = (mat[..., :3, :3] @ np.asarray(link.cog)[..., None])[..., 0]
      point = mat[..., :3, 3] + offset
      term = link.mass * point
      result = term if result is None else result + term
    if result is None:
      # _total_mass() already rejects this, but keep the invariant explicit.
      raise ValueError("center of mass requires at least one positive-mass link")
    return result / total_mass

  def center_of_mass(self) -> np.ndarray:
    """Return the whole-body center of mass in world coordinates."""
    return self._center_of_mass_from_motion(self.motion(order=1))

  def _project_motion_order(self, motion: np.ndarray, input_order: int, output_order: int) -> np.ndarray:
    if input_order < output_order:
      raise ValueError("input motion order must be at least output motion order")
    motion = np.asarray(motion)
    projected = np.zeros(motion.shape[:-1] + (self.robot_.dof * output_order,), dtype=motion.dtype)
    for owner in self.robot_.motion_owners():
      src = RobotMotions.owner_vec_index(owner.dof, owner.dof_index, input_order, output_order)
      dst = RobotMotions.owner_vec_index(owner.dof, owner.dof_index, output_order)
      projected[..., dst] = motion[..., src]
    return projected

  def _numerical_motion_jacobian(self, func, order: int, eps: float) -> np.ndarray:
    self._ensure_not_batched(func.__name__)
    if eps <= 0.0 or not np.isfinite(eps):
      raise ValueError("eps must be finite and positive")
    motion = np.asarray(self.motion(order=order), dtype=float).copy()
    value = np.asarray(func(motion), dtype=float)
    jacobian = np.empty(value.shape + (self.robot_.dof * order,), dtype=float)
    for dof_index in range(self.robot_.dof):
      for derivative in range(order):
        # owner-major layout places each derivative block immediately after the
        # coordinate block for its owner, not after all robot coordinates.
        owner = next(owner for owner in self.robot_.motion_owners()
                     if owner.dof_index <= dof_index < owner.dof_index + owner.dof)
        local = dof_index - owner.dof_index
        owner_indices = RobotMotions.owner_vec_index(owner.dof, owner.dof_index, order)
        index = owner_indices.start + derivative * owner.dof + local
        plus = motion.copy()
        minus = motion.copy()
        plus[index] += eps; minus[index] -= eps
        column = (np.asarray(func(plus)) - np.asarray(func(minus))) / (2.0 * eps)
        jacobian[..., index] = column
    return jacobian

  def center_of_mass_jacobian(self, eps: float = 1e-7) -> np.ndarray:
    """Return ``d center_of_mass / dq`` with shape ``(3, dof)``.

    The center of mass depends only on coordinates, so velocity and higher
    motion derivatives are not included in the result.
    """
    return self._numerical_motion_jacobian(self._center_of_mass_from_motion, order=1, eps=eps)

  def _angular_momentum_from_motion(self, motion: np.ndarray, about: str) -> np.ndarray:
    if about not in ("world", "com"):
      raise ValueError("about must be 'world' or 'com'")
    state = outward_api.build_dynamics_outward_state(self.robot_, motion, dynamics_order=0)
    total = None
    for link in self.robot_.links:
      momentum = np.asarray(outward_api.get_value(
        self.robot_, state, StateType("link", link.name, "momentum", "world")
      ))
      total = momentum if total is None else total + momentum
    if total is None:
      return np.zeros(3)
    angular = total[..., :3]
    if about == "world":
      return angular
    coords = self._project_motion_order(motion, input_order=2, output_order=1)
    return angular - np.cross(self._center_of_mass_from_motion(coords), total[..., 3:])

  def angular_momentum(self, about: str = "world") -> np.ndarray:
    """Return whole-body angular momentum about the world origin or CoM."""
    return self._angular_momentum_from_motion(self.motion(order=2), about)

  def angular_momentum_jacobian(self, about: str = "world", eps: float = 1e-7) -> np.ndarray:
    """Return angular-momentum Jacobian wrt ``(q, qdot)``.

    The column layout is RoboKots' normal owner-major motion layout and the
    result has shape ``(3, 2 * dof)``.
    """
    return self._numerical_motion_jacobian(
      lambda motion: self._angular_momentum_from_motion(motion, about), order=2, eps=eps
    )
