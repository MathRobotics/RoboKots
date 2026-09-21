"""Semantic state access and batch-state ownership for the ``Kots`` facade."""
from __future__ import annotations

from copy import deepcopy
import logging
from typing import Callable

import numpy as np

from .. import outward as outward_api
from ..core.state.batch import StateBatch
from .state_cache import StateCache, update_outward_state
from ..core.state.tensor import StateTensor
from ..core import batch_shape as batch_shapes


def _copy_public_state_value(value):
  """Detach public results without copying computational readers internally."""
  if isinstance(value, np.ndarray):
    return value.copy()
  return deepcopy(value)


_logger = logging.getLogger(__name__)


def _batch_state_info(batch, robot, state_type, get_value: Callable):
  states = batch.outward_states
  values = [get_value(robot, state, state_type) for state in states]
  return batch_shapes.stack_batch_values(values, batch.batch_shape)


def _batch_state_info_list(batch, robot, state_type_list, get_value: Callable, list_output: bool = False):
  states = batch.outward_states
  values = []
  for state in states:
    state_list = [get_value(robot, state, st) for st in state_type_list]
    if list_output:
      values.append(state_list)
    else:
      values.append(batch_shapes.concatenate_state_values(state_list))

  if list_output:
    return [
      batch_shapes.stack_batch_values([sample[i] for sample in values], batch.batch_shape)
      for i in range(len(state_type_list))
    ]
  return batch_shapes.stack_sample_results(values, batch.batch_shape)


class StateManagementMixin:
  def _set_batch_states(self, states, batch_shape: tuple, materialize_dict: bool = False):
    if not batch_shape:
      return self._set_current_state(states, materialize_dict=materialize_dict)
    if materialize_dict:
      from ..state_io.dictionary import export_state_dict

    if hasattr(states, "cmtm"):
      result = export_state_dict(self.robot_, states) if materialize_dict else states
      self.batch_shape_ = batch_shape
      self.state_batch_ = self.outward_state_ = states
      return result
    batch = StateBatch.from_states(states, batch_shape)
    result = export_state_dict(self.robot_, batch) if materialize_dict else batch.outward_states
    self.batch_shape_ = batch_shape
    self.state_batch_ = batch
    self.outward_state_ = self.state_batch_.outward_states
    return result

  def _invalidate_current_state(self):
    self.state_cache_ = self.state_cache_config_ = self.state_batch_ = self.outward_state_ = None
    self.batch_shape_ = ()

  def _ensure_not_batched(self, api_name: str):
    if self.batch_shape_ or self.motions_.is_batched() or self.state_batch_ is not None:
      raise ValueError(f"{api_name} does not support batched state or motion")

  def _ensure_state_table(self):
    if self.state_ is None:
      try:
        from ..contrib.polars import RobotState
      except ImportError as e:
        raise ImportError("DataFrame state tables are optional. Install RoboKots with the `table` extra.") from e
      self.state_ = RobotState(self.robot_.link_names, self.robot_.joint_names, self._state_l_aliases, self._state_j_aliases)
    return self.state_

  def state_df(self):
    self._ensure_not_batched("state_df")
    return self._ensure_state_table().df()

  def _state_for_direct_read(self):
    if self.outward_state_ is None:
      raise ValueError("No computed state. Call update_state(), kinematics(), or dynamics() first.")
    return self.outward_state_

  def state_info(self, state_type):
    """Return a detached value; modifying it never updates computed state."""
    return _copy_public_state_value(self._read_state_info(state_type))

  def _read_state_info(self, state_type):
    if self._is_total_body_kinetic_energy(state_type):
      return self.kinetic_energy_state()
    if state_type.owner_type == "total_joint":
      values = self.state_info_list(self._state_type_list(state_type))
      return values if self.batch_shape_ else np.asarray(values).reshape(-1)
    if isinstance(self.state_batch_, StateBatch):
      return _batch_state_info(self.state_batch_, self.robot_, state_type, outward_api.get_value)
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
          return _batch_state_info_list(self.state_batch_, self.robot_, state_type_list, outward_api.get_value, list_output=list_output)
        values = [outward_api.get_value(self.robot_, self._state_for_direct_read(), st) for st in state_type_list]
    if list_output:
      return [_copy_public_state_value(value) for value in values]
    return batch_shapes.concatenate_state_values(values, self.batch_shape_)

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
      return resolved, lambda x: outward_api.build_kinematics_outward_state(self.robot_, x, order, backend=resolved)
    return resolved, lambda x: outward_api.build_kinematics_outward_state(self.robot_, x, order)

  def _build_state_result(self, order: int, is_dynamics: bool = False, backend: str = None, gravity=None):
    resolved, build_state = self._state_builder(order, is_dynamics=is_dynamics, backend=backend, gravity=gravity)
    motion = self.motion(order)
    if batch_shapes.is_batched_feature_array(motion) and resolved in (None, "numpy", "rust"):
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
      except NotImplementedError as exc:
        _logger.debug("Batched state unavailable; evaluating individual samples: %s", exc)
    return batch_shapes.map_flat_batch(motion, build_state)

  def _set_current_state(self, state_obj, materialize_dict: bool = False):
    if materialize_dict:
      from ..state_io.dictionary import export_state_dict

    result = export_state_dict(self.robot_, state_obj) if materialize_dict else state_obj
    self.batch_shape_ = ()
    self.state_batch_ = None
    self.outward_state_ = state_obj
    return result

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
    if batch_shapes.is_batched_feature_array(motion):
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

    state = update_outward_state(
      self.robot_, MotionPack(motion, revision), self.state_cache_, is_dynamics, order, gravity=self.gravity_)
    return self._set_current_state(state, materialize_dict=False)

  def to_state_dict(self) -> dict:
    """Export the current state on demand; no dictionary is retained for computation."""
    from ..state_io.dictionary import export_state_dict

    if isinstance(self.state_batch_, StateBatch):
      return export_state_dict(self.robot_, self.state_batch_)
    return export_state_dict(self.robot_, self._state_for_direct_read())

  def update_state_dict(self, order: int = None, is_dynamics: bool = False, backend: str = None) -> dict:
    self.update_state(order=order, is_dynamics=is_dynamics, backend=backend)
    return self.to_state_dict()

  def set_state_df(self):
    self._ensure_not_batched("set_state_df")
    self._ensure_state_table().import_state(self.to_state_dict())
