"""Outward kinematics and higher-order inverse-dynamics facade methods."""
from __future__ import annotations

import numpy as np

from .. import outward as outward_api


class OutwardDynamicsMixin:
  def kinematics(self, order=None, backend: str = None, materialize_dict: bool = True):
    if order is None:
      order = self.order_
    if self._resolve_kinematics_backend(False, backend) == "rust":
      self.update_rust_data(order=order, is_dynamics=False, materialize_dict=materialize_dict)
      return
    states, batch_shape = self._build_state_result(order=order, is_dynamics=False, backend=backend)
    self._set_batch_states(states, batch_shape, materialize_dict=materialize_dict)

  def kinematics_point(self, s: float = 0.0):
    self._ensure_not_batched("kinematics_point")
    return outward_api.calc_link_total_point_frame(self.robot_, self.motions_, self.to_state_dict(), s)

  @staticmethod
  def _validate_gravity(gravity) -> np.ndarray:
    gravity = np.asarray(gravity, dtype=float)
    if gravity.shape != (3,):
      raise ValueError(f"gravity must have shape (3,), got {gravity.shape}.")
    if not np.all(np.isfinite(gravity)):
      raise ValueError("gravity must contain only finite values.")
    return gravity

  def dynamics(self, order=None, backend: str = None, materialize_dict: bool = True, gravity=(0.0, 0.0, 0.0)):
    """Compute higher-order inverse-dynamics state with world-frame gravity."""
    if order is None:
      order = self.order_
    self.gravity_ = self._validate_gravity(gravity).copy()
    resolved_backend = self._resolve_kinematics_backend(True, backend)
    if resolved_backend == "rust":
      self.update_rust_data(order=order, is_dynamics=True, materialize_dict=materialize_dict, gravity=self.gravity_)
      return
    states, batch_shape = self._build_state_result(order=order, is_dynamics=True, backend=backend, gravity=self.gravity_)
    self._set_batch_states(states, batch_shape, materialize_dict=materialize_dict)

  def _use_jax_kinematics_backend(self, backend: str = None) -> bool:
    return backend == "jax" or (
      backend is None and len(self.robot_.links) > 0
      and getattr(self.robot_.links[0], "lib", "numpy") == "jax"
      and all(link.dof == 0 for link in self.robot_.links)
    )

  def _resolve_kinematics_backend(self, is_dynamics: bool = False, backend: str = None):
    if is_dynamics:
      if backend not in (None, "numpy", "rust"):
        raise ValueError(f"Unsupported dynamics backend: {backend}. Use 'numpy' or 'rust'.")
      return backend
    if backend not in (None, "numpy", "jax", "rust"):
      raise ValueError(f"Unsupported kinematics backend: {backend}. Use 'numpy', 'jax', or 'rust'.")
    return backend
