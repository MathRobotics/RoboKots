"""Rust backend dispatch and reusable outward workspace management."""
from __future__ import annotations

import numpy as np

from ..core import batch as batch_api


class RustBackendMixin:
  def _rust_compiled_robot(self):
    if self._rust_compiled_robot_ is None:
      from ..outward.rust import _rust_compiled_robot
      self._rust_compiled_robot_ = _rust_compiled_robot(self.robot_)
    return self._rust_compiled_robot_

  def _rust_inverse_dynamics_robot(self):
    if self._rust_inverse_dynamics_robot_ is None:
      if all(joint.type in ("fixed", "revolute") for joint in self.robot_.joints):
        self._rust_inverse_dynamics_robot_ = self._rust_compiled_robot()
      else:
        from ..outward.rust.model import _rust_inverse_dynamics_robot
        self._rust_inverse_dynamics_robot_ = _rust_inverse_dynamics_robot(self.robot_)
    return self._rust_inverse_dynamics_robot_

  def _fast_backend(self, backend: str = "rust") -> str:
    if backend != "rust":
      raise ValueError(f"Unsupported fast backend: {backend}. Use 'rust'.")
    return backend

  def _fast_qva(self, q, v=None, a=None):
    q = np.asarray(q, dtype=float)
    if q.ndim not in (1, 2):
      raise ValueError("q must have shape (dof,) or (batch, dof)")
    if q.shape[-1] != self.robot_.dof:
      raise ValueError(f"q length must match robot dof: expected {self.robot_.dof}, got {q.shape[-1]}.")
    v = np.zeros_like(q) if v is None else np.asarray(v, dtype=float)
    a = np.zeros_like(q) if a is None else np.asarray(a, dtype=float)
    if v.shape != q.shape:
      raise ValueError(f"v shape must match q shape: expected {q.shape}, got {v.shape}.")
    if a.shape != q.shape:
      raise ValueError(f"a shape must match q shape: expected {q.shape}, got {a.shape}.")
    return tuple(np.ascontiguousarray(value) for value in (q, v, a))

  def _rust_fast_forward_kinematics(self, q, v=None, a=None, backend: str = "rust"):
    self._fast_backend(backend)
    q, v, a = self._fast_qva(q, v, a)
    robot = self._rust_compiled_robot()
    return robot.forward_kinematics(q, v, a) if q.ndim == 1 else robot.forward_kinematics_batch(q, v, a)

  def _rust_fast_rnea(self, q, v, a, backend: str = "rust"):
    self._fast_backend(backend)
    q, v, a = self._fast_qva(q, v, a)
    robot = self._rust_compiled_robot()
    return robot.rnea(q, v, a) if q.ndim == 1 else robot.rnea_batch(q, v, a)

  def _rust_fast_joint_jacobians(self, q, backend: str = "rust"):
    self._fast_backend(backend)
    q = np.ascontiguousarray(np.asarray(q, dtype=float))
    if q.ndim not in (1, 2):
      raise ValueError("q must have shape (dof,) or (batch, dof)")
    if q.shape[-1] != self.robot_.dof:
      raise ValueError(f"q length must match robot dof: expected {self.robot_.dof}, got {q.shape[-1]}.")
    robot = self._rust_compiled_robot()
    return robot.joint_jacobians(q) if q.ndim == 1 else robot.joint_jacobians_batch(q)

  def _create_rust_fast_data(self):
    return self._rust_compiled_robot().create_fast_data()

  def _create_rust_pinocchio_like_data(self):
    return self._rust_compiled_robot().create_pinocchio_like_data()

  def _create_rust_aba_data(self):
    return self._rust_inverse_dynamics_robot().create_aba_data()

  def _create_rust_outward_state(self, order=None):
    from ..outward.rust import create_rust_outward_state
    return create_rust_outward_state(self.robot_, self.order_ if order is None else order, compiled_robot=self._rust_compiled_robot())

  def _create_rust_batch_outward_state(self, order=None, batch_shape=None):
    from ..outward.rust import create_rust_batch_outward_state
    if batch_shape is None:
      batch_shape = self.motions_.batch_shape()
    return create_rust_batch_outward_state(self.robot_, self.order_ if order is None else order, batch_shape, compiled_robot=self._rust_compiled_robot())

  def _cached_rust_data(self, order, batch_shape=()):
    key = (int(order), tuple(batch_shape))
    data = self._rust_outward_data_cache_.get(key)
    if data is None:
      data = self._create_rust_batch_outward_state(order, key[1]) if key[1] else self._create_rust_outward_state(order)
      self._rust_outward_data_cache_[key] = data
      self._rust_outward_data_cache_state_.pop(key, None)
    return data

  def update_rust_data(self, order=None, is_dynamics=False, materialize_dict=False, gravity=None):
    if order is None:
      order = self.order_
    motion = self.motion(order)
    batch_shape = motion.shape[:-1] if batch_api.is_batched_feature_array(motion) else ()
    data = self._cached_rust_data(order, batch_shape)
    active_gravity = self.gravity_ if gravity is None else self._validate_gravity(gravity)
    gravity_key = tuple(active_gravity) if is_dynamics else None
    key = (int(order), tuple(batch_shape))
    cached = self._rust_outward_data_cache_state_.get(key)
    if cached is None or cached[0] != self.motions_.revision() or (is_dynamics and (not cached[1] or cached[2] != gravity_key)):
      if is_dynamics:
        data.compute_dynamics(motion, active_gravity) if np.any(active_gravity) else data.compute_dynamics(motion)
      else:
        data.compute_kinematics(motion)
      self._rust_outward_data_cache_state_[key] = (self.motions_.revision(), bool(is_dynamics), gravity_key)
    self.batch_shape_ = tuple(batch_shape)
    self.state_batch_ = None
    self.outward_state_ = data
    self.state_dict_ = data.to_state_dict(self.robot_) if materialize_dict else {}
    self.state_dict_source_ = data if materialize_dict else None
    return data
