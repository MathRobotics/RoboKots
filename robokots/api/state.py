"""Semantic state access and batch-state ownership for the ``Kots`` facade."""
from __future__ import annotations

import numpy as np

from .. import outward as outward_api
from ..core.state_batch import StateBatch
from ..core.state_cache import StateCache
from ..core.state_tensor import StateTensor
from ..core import batch as batch_api


class StateManagementMixin:
  def _set_batch_states(self, states, batch_shape: tuple, materialize_dict: bool = True):
    self.batch_shape_ = batch_shape
    if not batch_shape:
      return self._set_current_state(states, materialize_dict=materialize_dict)
    if hasattr(states, "to_state_dict"):
      self.state_batch_ = self.outward_state_ = states
      self.state_dict_ = states.to_state_dict(self.robot_) if materialize_dict else {}
      self.state_dict_source_ = states if materialize_dict else None
      return self.state_dict_ if materialize_dict else states
    self.state_batch_ = StateBatch.from_states(states, batch_shape, self.robot_, materialize_dict=materialize_dict)
    self.outward_state_ = self.state_batch_.outward_states
    self.state_dict_ = self.state_batch_.state_dicts if materialize_dict or self.outward_state_ is None else {}
    self.state_dict_source_ = self.state_batch_ if self.state_dict_ else None
    return self.state_dict_ if materialize_dict or self.outward_state_ is None else self.outward_state_

  def _invalidate_current_state(self):
    self.state_cache_ = self.state_cache_config_ = self.state_batch_ = self.outward_state_ = None
    self.state_dict_ = {}
    self.state_dict_source_ = None
    self.batch_shape_ = ()

  def _ensure_not_batched(self, api_name: str):
    if self.batch_shape_ or self.motions_.is_batched() or self.state_batch_ is not None:
      raise ValueError(f"{api_name} does not support batched state or motion")

  def _ensure_state_table(self):
    if self.state_ is None:
      try:
        from ..contrib.polars.state_table import RobotState
      except ImportError as e:
        raise ImportError("DataFrame state tables are optional. Install RoboKots with the `table` extra.") from e
      self.state_ = RobotState(self.robot_.link_names, self.robot_.joint_names, self._state_l_aliases, self._state_j_aliases)
    return self.state_

  def state_df(self):
    self._ensure_not_batched("state_df")
    return self._ensure_state_table().df()

  def _state_for_direct_read(self):
    return self.outward_state_ if self.outward_state_ is not None else self.state_dict_

  def state_info(self, state_type):
    if self._is_total_body_kinetic_energy(state_type):
      return self.kinetic_energy_state()
    if state_type.owner_type == "total_joint":
      values = self.state_info_list(self._state_type_list(state_type))
      return values if self.batch_shape_ else np.asarray(values).reshape(-1)
    if isinstance(self.state_batch_, StateBatch):
      return self.state_batch_.state_info(self.robot_, state_type, outward_api.get_value)
    value = outward_api.get_value(self.robot_, self._state_for_direct_read(), state_type)
    return value.mat() if self.batch_shape_ and hasattr(value, "mat") else value

  def state_info_list(self, state_type_list, list_output: bool = False):
    state_type_list = self._state_type_list(state_type_list)
    if any(self._is_total_body_kinetic_energy(st) for st in state_type_list):
      values = [self.state_info(st) for st in state_type_list]
    else:
      values = self._joint_motion_state_info_list(state_type_list)
      if values is None:
        if isinstance(self.state_batch_, StateBatch):
          return self.state_batch_.state_info_list(self.robot_, state_type_list, outward_api.get_value, list_output=list_output)
        values = [outward_api.get_value(self.robot_, self._state_for_direct_read(), st) for st in state_type_list]
    if list_output:
      return values
    if self.batch_shape_:
      return np.concatenate([np.asarray(v).reshape(self.batch_shape_ + (-1,)) for v in values], axis=-1)
    return np.concatenate([np.asarray(v).reshape(-1) for v in values], axis=-1) if any(self._is_total_body_kinetic_energy(st) for st in state_type_list) or self._joint_motion_state_info_list(state_type_list) is not None else np.vstack(values)

  def target_state_info(self, list_output: bool = False):
    if self.target_ is None:
      raise ValueError("target is not set")
    return self.state_info_list(self.target_._targets, list_output=list_output)

  def state_tensor(self, state_type):
    states = self._state_type_list(state_type)
    values = self.state_info_list(states)
    return StateTensor.from_array(values if self.batch_shape_ else np.asarray(values).reshape(-1), states)

  def target_state_tensor(self):
    if self.target_ is None:
      raise ValueError("target is not set")
    values = self.target_state_info()
    return StateTensor.from_array(values if self.batch_shape_ else np.asarray(values).reshape(-1), self.target_._targets)

  def _state_builder(self, order: int, is_dynamics: bool = False, backend: str = None, gravity=None):
    resolved = self._resolve_kinematics_backend(is_dynamics, backend)
    if is_dynamics:
      gravity = self.gravity_ if gravity is None else self._validate_gravity(gravity)
      if resolved == "rust":
        return resolved, lambda x: outward_api.build_dynamics_outward_state_rust(
          self.robot_, x, order - 2, compiled_robot=self._rust_compiled_robot(), gravity=gravity)
      return resolved, lambda x: outward_api.build_dynamics_outward_state(self.robot_, x, order - 2, gravity=gravity)
    if resolved == "rust":
      return resolved, lambda x: outward_api.build_kinematics_outward_state_rust(
        self.robot_, x, order, compiled_robot=self._rust_compiled_robot())
    if self._use_jax_kinematics_backend(resolved):
      return resolved, lambda x: outward_api.build_kinematics_state(self.robot_, x, order, backend=resolved)
    return resolved, lambda x: outward_api.build_kinematics_outward_state(self.robot_, x, order)

  def _build_state_result(self, order: int, is_dynamics: bool = False, backend: str = None, gravity=None):
    resolved, build_state = self._state_builder(order, is_dynamics=is_dynamics, backend=backend, gravity=gravity)
    motion = self.motion(order)
    if batch_api.is_batched_feature_array(motion) and resolved in (None, "numpy", "rust"):
      try:
        if is_dynamics:
          active_gravity = self.gravity_ if gravity is None else gravity
          if resolved == "rust":
            return outward_api.build_dynamics_outward_state_rust(
              self.robot_, motion, order - 2, compiled_robot=self._rust_compiled_robot(), gravity=active_gravity), motion.shape[:-1]
          return outward_api.build_dynamics_outward_state(self.robot_, motion, order - 2, gravity=active_gravity), motion.shape[:-1]
        if resolved == "rust":
          return outward_api.build_kinematics_outward_state_rust(
            self.robot_, motion, order, compiled_robot=self._rust_compiled_robot()), motion.shape[:-1]
        return outward_api.build_kinematics_outward_state(self.robot_, motion, order), motion.shape[:-1]
      except Exception:
        pass
    return batch_api.map_flat_batch(motion, build_state)

  def _set_current_state(self, state_obj, materialize_dict: bool = True):
    self.batch_shape_ = ()
    self.state_batch_ = None
    if hasattr(state_obj, "to_state_dict"):
      self.outward_state_ = state_obj
      self.state_dict_ = state_obj.to_state_dict(self.robot_) if materialize_dict else {}
      self.state_dict_source_ = state_obj if materialize_dict else None
    else:
      self.outward_state_ = None
      self.state_dict_ = state_obj
      self.state_dict_source_ = state_obj
    return self.state_dict_ if materialize_dict or not hasattr(state_obj, "to_state_dict") else state_obj

  def update_state(self, order: int = None, is_dynamics: bool = False, backend: str = None):
    if order is None:
      order = self.order_
    resolved, build_state = self._state_builder(order, is_dynamics=is_dynamics, backend=backend)
    if resolved == "rust":
      return self.update_rust_data(order=order, is_dynamics=is_dynamics)
    revision = self.motions_.revision()
    config = (bool(is_dynamics), int(order), resolved, tuple(self.gravity_) if is_dynamics else None)
    if not self.motions_.is_batched() and self.state_cache_ is not None and self.state_cache_config_ == config and self.state_cache_.is_fresh(revision):
      return self._set_current_state(self.state_cache_.state, materialize_dict=False)
    motion = self.motion(order)
    if batch_api.is_batched_feature_array(motion):
      states, batch_shape = self._build_state_result(order, is_dynamics=is_dynamics, backend=backend)
      return self._set_batch_states(states, batch_shape, materialize_dict=False)
    if self.state_cache_ is None or self.state_cache_config_ != config:
      self.state_cache_ = StateCache(build_state=lambda x_all, time=None, required=None: build_state(x_all))
      self.state_cache_config_ = config

    class MotionPack:
      def __init__(self, x, revision):
        self._x = np.asarray(x, dtype=float).reshape(-1)
        self.revision = int(revision)
      def get(self):
        return self._x

    state = outward_api.update_outward_state(
      self.robot_, MotionPack(motion, revision), self.state_cache_, is_dynamics, order, gravity=self.gravity_)
    return self._set_current_state(state, materialize_dict=False)

  def to_state_dict(self) -> dict:
    if isinstance(self.state_batch_, StateBatch):
      if self.state_dict_source_ is not self.state_batch_:
        if not self.state_batch_.state_dicts and self.state_batch_.outward_states is not None:
          self.state_batch_ = StateBatch.from_states(self.state_batch_.outward_states, self.state_batch_.batch_shape, self.robot_, materialize_dict=True)
        self.state_dict_ = self.state_batch_.state_dicts
        self.state_dict_source_ = self.state_batch_
      return self.state_dict_
    if self.outward_state_ is not None and hasattr(self.outward_state_, "to_state_dict") and self.state_dict_source_ is not self.outward_state_:
      self.state_dict_ = self.outward_state_.to_state_dict(self.robot_)
      self.state_dict_source_ = self.outward_state_
    return self.state_dict_

  def update_state_dict(self, order: int = None, is_dynamics: bool = False, backend: str = None) -> dict:
    self.update_state(order=order, is_dynamics=is_dynamics, backend=backend)
    return self.to_state_dict()

  def set_state_df(self):
    self._ensure_not_batched("set_state_df")
    self._ensure_state_table().import_state(self.to_state_dict())
