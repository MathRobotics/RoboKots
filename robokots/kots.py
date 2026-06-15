#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki

import numpy as np
from typing import List, Any, Optional, Tuple

from .core.motion import RobotMotions
from .core.state import StateType, data_type_dof, dim_to_dof, is_in_keys_dynamics, keys_force, keys_joint_motion, keys_kinematics, keys_momentum, keys_torque
from .core.state_cache import StateCache
from .core.state_batch import StateBatch
from .core.state_tensor import JacobianTensor, StateTensor
from .core.robot import RobotStruct
from .core.target import TargetList, RobotNames
from .core.viz import show_robot, show_robot_traj, RobotColor, show_link_points
from .core import batch as batch_api

from . import outward as outward_api
from .robot_io import load_json_file
from .urdf_io import load_urdf_file

default_order = 3 
default_dim = 3
class Kots():
  robot_ : RobotStruct
  motions_ : RobotMotions
  state_dict_ : dict
  state_ : Optional[Any]
  outward_state_ : Optional[Any]
  state_dict_source_ : Optional[Any]
  target_ : TargetList
  order_ : int
  dim_ : int
  lib_ : str
  state_cache_config_ : Optional[Tuple[bool, int, Optional[str]]]

  def order_to_aliases(self, order: int) -> List[str]:
    m_aliases = []
    l_aliases = []
    j_aliases = []
    
    if order == 1:
      m_aliases = ["coord"]
      j_aliases = ["pos", "rot"]
      l_aliases = ["pos", "rot"]
    elif order == 2:
      m_aliases = ["coord", "veloc"]
      j_aliases = ["pos", "rot", "vel"]
      l_aliases = ["pos", "rot", "vel"]
    elif order > 2:
      m_aliases = ["coord", "veloc", "accel"]
      j_aliases = ["pos", "rot", "vel", "acc"]
      l_aliases = ["pos", "rot", "vel", "acc"]
      for i in range(order-3):
        m_aliases.append("accel_diff"+str(i+1))
        j_aliases.append("acc_diff"+str(i+1))
        l_aliases.append("acc_diff"+str(i+1))
        
    l_aliases.append("force")
    j_aliases.append("torque")
    j_aliases.append("force")
    
    for i in range(order-3):
      l_aliases.append("force_diff"+str(i+1))
      j_aliases.append("torque_diff"+str(i+1))
      j_aliases.append("force_diff"+str(i+1))

    return m_aliases, l_aliases, j_aliases
  
  def __init__(self, robot : RobotStruct, order : int, dim : int, lib : str = "numpy"):

    m_aliases, l_aliases, j_aliases = self.order_to_aliases(order)

    self.robot_ = robot
    self.motions_ = RobotMotions(robot.dof, m_aliases, owner_layout=robot.motion_owners())
    self.state_ = None
    self.outward_state_ = None
    self._state_l_aliases = l_aliases
    self._state_j_aliases = j_aliases
    self.state_dict_ = {}
    self.state_dict_source_ = None
    self.state_cache_ = None
    self.state_cache_config_ = None
    self.target_ = None
    self.state_batch_ = None
    self.order_ = order
    self.dim_ = dim
    self.lib_ = lib
    self.batch_shape_ = ()
    self._rust_compiled_robot_ = None
    self._rust_outward_data_cache_ = {}
    self._rust_outward_data_cache_state_ = {}

  def set_order(self, order: int):
    if order < 1:
      raise ValueError("order must be greater than 0")
    m_aliases, l_aliases, j_aliases = self.order_to_aliases(order)
    self.order_ = order
    self.motions_ = RobotMotions(self.robot_.dof, m_aliases, owner_layout=self.robot_.motion_owners())
    self.state_ = None
    self.outward_state_ = None
    self.state_dict_ = {}
    self.state_dict_source_ = None
    self._state_l_aliases = l_aliases
    self._state_j_aliases = j_aliases
    self.state_cache_ = None
    self.state_cache_config_ = None
    self.state_batch_ = None
    self.batch_shape_ = ()
    self._rust_compiled_robot_ = None
    self._rust_outward_data_cache_ = {}
    self._rust_outward_data_cache_state_ = {}

  @staticmethod
  def from_json_file(model_file_name : str, order=default_order, dim=default_dim, lib : str = "numpy") -> "Kots":
    robot = RobotStruct.from_dict(load_json_file(model_file_name), lib)

    return Kots(robot, order, dim, lib)

  @staticmethod
  def from_json_data(model_data : dict, order=default_order, dim=default_dim, lib : str = "numpy") -> "Kots":
    robot = RobotStruct.from_dict(model_data, lib=lib)

    return Kots(robot, order, dim, lib)

  @staticmethod
  def from_urdf_file(
      urdf_file_name: str,
      order=default_order,
      dim=default_dim,
      lib: str = "numpy",
      add_world_link: bool = True,
  ) -> "Kots":
    model_data = load_urdf_file(urdf_file_name, add_world_link=add_world_link)
    robot = RobotStruct.from_dict(model_data, lib=lib)
    return Kots(robot, order, dim, lib)

  def print_structure(self):
    self.robot_.print()

  def print_state_dict(self):
    from .core.state_dict import print_state_dict

    print_state_dict(self.to_state_dict())

  def targets(self):
    return self.target_
    
  def dof(self):
    return self.robot_.dof

  def order(self):
    return self.order_
  
  def link_name_list(self):
    return self.robot_.link_names
  
  def joint_name_list(self):
    return self.robot_.joint_names

  def motions(self):
    return self.motions_.motions

  def set_motion_aliases(self, aliases : list[str]):
    self.motions_.set_aliases(aliases)
    self._invalidate_current_state()
    
  def import_motions(self, vecs : np.ndarray):
    self.motions_.set_motion(vecs)
    self.motions_.increment_revision()
    self._invalidate_current_state()
    self.batch_shape_ = self.motions_.batch_shape()

  def import_motion_array(self, array : np.ndarray):
    self.motions_.set_dof_order(array)
    self.motions_.increment_revision()
    self._invalidate_current_state()
    self.batch_shape_ = self.motions_.batch_shape()

  def motion_tensor(self):
    return self.motions_.motion_tensor()

  def motion_array(self, order : int = None):
    if order is None:
      order = self.order_
    return self.motions_.to_dof_order(order)

  def motion(self, order : int = None):
    if order is None:
      order = self.order_
    return self.motions_.to_vector(order)
  
  def motion_derivative(self, order : int = None, tail = None):
    if order is None:
      order = self.order_
    return self.motions_.to_derivative_vector(order, tail=tail)

  def motion_derivative_array(self, order : int = None, tail = None):
    if order is None:
      order = self.order_
    return self.motion_tensor().derivative(tail).as_dof_order(order).data

  def motion_diff(self, order : int = None, last_diff = None):
    return self.motion_derivative(order, tail=last_diff)
  
  def motion_cm(self, order : int = None):
    if order is None:
      order = self.order_
    return self.motions_.to_vector(order, cm=True)

  def motion_cm_array(self, order : int = None):
    if order is None:
      order = self.order_
    return self.motion_tensor().cm_scaled().as_dof_order(order).data
  
  def motion_derivative_cm(self, order : int = None, tail = None):
    if order is None:
      order = self.order_
    return self.motions_.to_derivative_vector(order, tail=tail, cm=True)

  def motion_derivative_cm_array(self, order : int = None, tail = None):
    if order is None:
      order = self.order_
    return self.motion_tensor().derivative(tail).cm_scaled().as_dof_order(order).data

  def motion_diff_cm(self, order : int = None, last_diff = None):
    return self.motion_derivative_cm(order, tail=last_diff)

  def _set_batch_states(self, states, batch_shape : tuple, materialize_dict : bool = True):
    self.batch_shape_ = batch_shape
    if not batch_shape:
      return self._set_current_state(states, materialize_dict=materialize_dict)
    if hasattr(states, "to_state_dict"):
      self.state_batch_ = states
      self.outward_state_ = states
      self.state_dict_ = states.to_state_dict(self.robot_) if materialize_dict else {}
      self.state_dict_source_ = states if materialize_dict else None
      return self.state_dict_ if materialize_dict else states
    self.state_batch_ = StateBatch.from_states(states, batch_shape, self.robot_, materialize_dict=materialize_dict)
    self.outward_state_ = self.state_batch_.outward_states
    self.state_dict_ = self.state_batch_.state_dicts if materialize_dict or self.outward_state_ is None else {}
    self.state_dict_source_ = self.state_batch_ if self.state_dict_ else None
    return self.state_dict_ if materialize_dict or self.outward_state_ is None else self.outward_state_

  def _invalidate_current_state(self):
    self.state_cache_ = None
    self.state_cache_config_ = None
    self.state_batch_ = None
    self.outward_state_ = None
    self.state_dict_ = {}
    self.state_dict_source_ = None
    self.batch_shape_ = ()

  def _ensure_not_batched(self, api_name : str):
    if self.batch_shape_ or self.motions_.is_batched() or self.state_batch_ is not None:
      raise ValueError(f"{api_name} does not support batched state or motion")

  def _ensure_state_table(self):
    if self.state_ is None:
      try:
        from .contrib.polars.state_table import RobotState
      except ImportError as e:
        raise ImportError(
          "DataFrame state tables are optional. Install RoboKots with the "
          "`table` extra, for example `pip install 'robokots[table]'`, "
          "to use state_df(), set_state_df(), or trajectory helpers without "
          "an explicit traj argument."
        ) from e
      self.state_ = RobotState(
        self.robot_.link_names,
        self.robot_.joint_names,
        self._state_l_aliases,
        self._state_j_aliases,
      )
    return self.state_

  def state_df(self):
    self._ensure_not_batched("state_df")
    return self._ensure_state_table().df()

  def _state_for_direct_read(self):
    return self.outward_state_ if self.outward_state_ is not None else self.state_dict_

  def state_info(self, state_type : StateType):
    if state_type.owner_type == "total_joint":
      values = self.state_info_list(self._state_type_list(state_type))
      if not self.batch_shape_:
        values = np.asarray(values).reshape(-1)
      return values
    if isinstance(self.state_batch_, StateBatch):
      return self.state_batch_.state_info(self.robot_, state_type, outward_api.get_value)
    value = outward_api.get_value(self.robot_, self._state_for_direct_read(), state_type)
    if self.batch_shape_ and hasattr(value, "mat"):
      return value.mat()
    return value

  def state_info_list(self, state_type_list : List[StateType], list_output : bool = False) -> List[np.ndarray]:
    state_type_list = self._state_type_list(state_type_list)
    motion_values = self._joint_motion_state_info_list(state_type_list)
    if motion_values is not None:
      if list_output:
        return motion_values
      if self.batch_shape_:
        return np.concatenate([np.asarray(v).reshape(self.batch_shape_ + (-1,)) for v in motion_values], axis=-1)
      return np.concatenate([np.asarray(v).reshape(-1) for v in motion_values], axis=-1)
    if isinstance(self.state_batch_, StateBatch):
      return self.state_batch_.state_info_list(self.robot_, state_type_list, outward_api.get_value, list_output=list_output)
    state = self._state_for_direct_read()
    state_list = [outward_api.get_value(self.robot_, state, st) for st in state_type_list]
    if list_output:
        return state_list
    elif self.batch_shape_:
        return np.concatenate([np.asarray(v).reshape(self.batch_shape_ + (-1,)) for v in state_list], axis=-1)
    else:
        return np.vstack(state_list)

  def target_state_info(self, list_output : bool = False) -> np.ndarray:
    if self.target_ is None:
      raise ValueError("target is not set")

    return self.state_info_list(self.target_._targets, list_output=list_output)

  def state_tensor(self, state_type) -> StateTensor:
    state_type_list = self._state_type_list(state_type)
    values = self.state_info_list(state_type_list)
    if not self.batch_shape_:
      values = np.asarray(values).reshape(-1)
    return StateTensor.from_array(values, state_type_list)

  def target_state_tensor(self) -> StateTensor:
    if self.target_ is None:
      raise ValueError("target is not set")
    values = self.target_state_info()
    if not self.batch_shape_:
      values = np.asarray(values).reshape(-1)
    return StateTensor.from_array(values, self.target_._targets)
  
  def kinematics(self, order = None, backend : str = None, materialize_dict : bool = True):
    if order is None:
      order = self.order_
    if self._resolve_kinematics_backend(False, backend) == "rust":
      self.update_rust_data(order=order, is_dynamics=False, materialize_dict=materialize_dict)
      return
    states, batch_shape = self._build_state_result(order=order, is_dynamics=False, backend=backend)
    self._set_batch_states(states, batch_shape, materialize_dict=materialize_dict)

  # ToDo: change function name
  def kinematics_point(self, s : float = 0.0):
    self._ensure_not_batched("kinematics_point")
    return outward_api.calc_link_total_point_frame(self.robot_, self.motions_, self.to_state_dict(), s)
  
  def dynamics(self, order = None, backend : str = None, materialize_dict : bool = True):
    if order is None:
      order = self.order_
    if self._resolve_kinematics_backend(True, backend) == "rust":
      self.update_rust_data(order=order, is_dynamics=True, materialize_dict=materialize_dict)
      return
    states, batch_shape = self._build_state_result(order=order, is_dynamics=True, backend=backend)
    self._set_batch_states(states, batch_shape, materialize_dict=materialize_dict)

  def _use_jax_kinematics_backend(self, backend: str = None) -> bool:
    return (
      backend == "jax"
      or (
        backend is None
        and len(self.robot_.links) > 0
        and getattr(self.robot_.links[0], "lib", "numpy") == "jax"
        and all(link.dof == 0 for link in self.robot_.links)
      )
    )

  def _resolve_kinematics_backend(self, is_dynamics : bool = False, backend : str = None):
    if is_dynamics:
      if backend not in (None, "numpy", "rust"):
        raise ValueError(f"Unsupported dynamics backend: {backend}. Use 'numpy' or 'rust'.")
      return backend
    if backend not in (None, "numpy", "jax", "rust"):
      raise ValueError(f"Unsupported kinematics backend: {backend}. Use 'numpy', 'jax', or 'rust'.")
    return backend

  def _rust_compiled_robot(self):
    if self._rust_compiled_robot_ is None:
      from .outward.rust import _rust_compiled_robot

      self._rust_compiled_robot_ = _rust_compiled_robot(self.robot_)
    return self._rust_compiled_robot_

  def _fast_backend(self, backend : str = "rust") -> str:
    if backend != "rust":
      raise ValueError(f"Unsupported fast backend: {backend}. Use 'rust'.")
    return backend

  def _fast_qva(self, q, v = None, a = None):
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
    return q, v, a

  def _rust_fast_forward_kinematics(self, q, v = None, a = None, backend : str = "rust"):
    self._fast_backend(backend)
    q, v, a = self._fast_qva(q, v, a)
    rust_robot = self._rust_compiled_robot()
    if q.ndim == 1:
      return rust_robot.forward_kinematics(q, v, a)
    return rust_robot.forward_kinematics_batch(q, v, a)

  def _rust_fast_rnea(self, q, v, a, backend : str = "rust"):
    self._fast_backend(backend)
    q, v, a = self._fast_qva(q, v, a)
    rust_robot = self._rust_compiled_robot()
    if q.ndim == 1:
      return rust_robot.rnea(q, v, a)
    return rust_robot.rnea_batch(q, v, a)

  def _rust_fast_joint_jacobians(self, q, backend : str = "rust"):
    self._fast_backend(backend)
    q = np.asarray(q, dtype=float)
    if q.ndim not in (1, 2):
      raise ValueError("q must have shape (dof,) or (batch, dof)")
    if q.shape[-1] != self.robot_.dof:
      raise ValueError(f"q length must match robot dof: expected {self.robot_.dof}, got {q.shape[-1]}.")
    rust_robot = self._rust_compiled_robot()
    if q.ndim == 1:
      return rust_robot.joint_jacobians(q)
    return rust_robot.joint_jacobians_batch(q)

  def _create_rust_fast_data(self):
    return self._rust_compiled_robot().create_fast_data()

  def _create_rust_pinocchio_like_data(self):
    return self._rust_compiled_robot().create_pinocchio_like_data()

  def _create_rust_outward_state(self, order : int = None):
    if order is None:
      order = self.order_
    from .outward.rust import create_rust_outward_state

    return create_rust_outward_state(
      self.robot_,
      order,
      compiled_robot=self._rust_compiled_robot(),
    )

  def _create_rust_batch_outward_state(self, order : int = None, batch_shape : tuple[int, ...] = None):
    if order is None:
      order = self.order_
    if batch_shape is None:
      batch_shape = self.motions_.batch_shape()
    from .outward.rust import create_rust_batch_outward_state

    return create_rust_batch_outward_state(
      self.robot_,
      order,
      batch_shape,
      compiled_robot=self._rust_compiled_robot(),
    )

  def _cached_rust_data(self, order : int, batch_shape : tuple[int, ...] = ()):
    batch_shape = tuple(batch_shape)
    key = (int(order), batch_shape)
    data = self._rust_outward_data_cache_.get(key)
    if data is None:
      if batch_shape:
        data = self._create_rust_batch_outward_state(order, batch_shape)
      else:
        data = self._create_rust_outward_state(order)
      self._rust_outward_data_cache_[key] = data
      self._rust_outward_data_cache_state_.pop(key, None)
    return data

  def update_rust_data(self, order : int = None, is_dynamics : bool = False, materialize_dict : bool = False):
    if order is None:
      order = self.order_

    motion = self.motion(order)
    batch_shape = motion.shape[:-1] if batch_api.is_batched_feature_array(motion) else ()
    data = self._cached_rust_data(order, batch_shape)
    revision = self.motions_.revision()
    key = (int(order), tuple(batch_shape))
    cached = self._rust_outward_data_cache_state_.get(key)
    needs_compute = (
      cached is None
      or cached[0] != revision
      or (is_dynamics and not cached[1])
    )
    if needs_compute:
      if is_dynamics:
        data.compute_dynamics(motion)
      else:
        data.compute_kinematics(motion)
      self._rust_outward_data_cache_state_[key] = (revision, bool(is_dynamics))

    self.batch_shape_ = tuple(batch_shape)
    self.state_batch_ = None
    self.outward_state_ = data
    self.state_dict_ = data.to_state_dict(self.robot_) if materialize_dict else {}
    self.state_dict_source_ = data if materialize_dict else None
    return data

  def _state_builder(self, order : int, is_dynamics : bool = False, backend : str = None):
    kinematics_backend = self._resolve_kinematics_backend(is_dynamics, backend)
    if is_dynamics:
      if kinematics_backend == "rust":
        return (
          kinematics_backend,
          lambda x: outward_api.build_dynamics_outward_state_rust(
            self.robot_,
            x,
            order-2,
            compiled_robot=self._rust_compiled_robot(),
          ),
        )
      return (
        kinematics_backend,
        lambda x: outward_api.build_dynamics_outward_state(self.robot_, x, order-2),
      )
    if kinematics_backend == "rust":
      return (
        kinematics_backend,
        lambda x: outward_api.build_kinematics_outward_state_rust(
          self.robot_,
          x,
          order,
          compiled_robot=self._rust_compiled_robot(),
        ),
      )
    if self._use_jax_kinematics_backend(kinematics_backend):
      return (
        kinematics_backend,
        lambda x: outward_api.build_kinematics_state(self.robot_, x, order, backend=kinematics_backend),
      )
    return (
      kinematics_backend,
      lambda x: outward_api.build_kinematics_outward_state(self.robot_, x, order),
    )

  def _build_state_result(self, order : int, is_dynamics : bool = False, backend : str = None):
    kinematics_backend, build_state = self._state_builder(order, is_dynamics=is_dynamics, backend=backend)
    motion = self.motion(order)
    if (
      batch_api.is_batched_feature_array(motion)
      and kinematics_backend in (None, "numpy", "rust")
    ):
      try:
        if is_dynamics:
          if kinematics_backend == "rust":
            return outward_api.build_dynamics_outward_state_rust(
              self.robot_,
              motion,
              order - 2,
              compiled_robot=self._rust_compiled_robot(),
            ), motion.shape[:-1]
          return outward_api.build_dynamics_outward_state(self.robot_, motion, order - 2), motion.shape[:-1]
        if kinematics_backend == "rust":
          return outward_api.build_kinematics_outward_state_rust(
            self.robot_,
            motion,
            order,
            compiled_robot=self._rust_compiled_robot(),
          ), motion.shape[:-1]
        return outward_api.build_kinematics_outward_state(self.robot_, motion, order), motion.shape[:-1]
      except Exception:
        pass
    return batch_api.map_flat_batch(motion, build_state)

  def _set_current_state(self, state_obj, materialize_dict : bool = True):
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

  def update_state(self, order : int = None, is_dynamics: bool = False, backend : str = None):
    if order is None:
      order = self.order_

    kinematics_backend, build_state = self._state_builder(order, is_dynamics=is_dynamics, backend=backend)
    if kinematics_backend == "rust":
      return self.update_rust_data(order=order, is_dynamics=is_dynamics)

    motion_revision = self.motions_.revision()

    cache_config = (bool(is_dynamics), int(order), kinematics_backend)
    if not self.motions_.is_batched():
      if self.state_cache_ is None or self.state_cache_config_ != cache_config:
        self.state_cache_ = StateCache(build_state=lambda x_all, time=None, required=None: build_state(x_all))
        self.state_cache_config_ = cache_config
      elif self.state_cache_.is_fresh(motion_revision):
        if self.outward_state_ is self.state_cache_.state or self.state_dict_ is self.state_cache_.state:
          return self.state_cache_.state
        return self._set_current_state(self.state_cache_.state, materialize_dict=False)

    motion = self.motion(order)
    if batch_api.is_batched_feature_array(motion):
      states, batch_shape = self._build_state_result(order=order, is_dynamics=is_dynamics, backend=backend)
      return self._set_batch_states(states, batch_shape, materialize_dict=False)

    if self.state_cache_ is None or self.state_cache_config_ != cache_config:
      self.state_cache_ = StateCache(build_state=lambda x_all, time=None, required=None: build_state(x_all))
      self.state_cache_config_ = cache_config

    if self.state_cache_.is_fresh(motion_revision):
      return self._set_current_state(self.state_cache_.state, materialize_dict=False)

    class _MotionPack:
      def __init__(self, x: np.ndarray, revision: int):
        self._x = np.asarray(x, dtype=float).reshape(-1)
        self.revision = int(revision)

      def get(self) -> np.ndarray:
        return self._x

    motion_pack = _MotionPack(motion, motion_revision)

    state_obj = outward_api.update_outward_state(self.robot_, motion_pack, self.state_cache_, is_dynamics, order)
    return self._set_current_state(state_obj, materialize_dict=False)

  def to_state_dict(self) -> dict:
    if isinstance(self.state_batch_, StateBatch):
      if self.state_dict_source_ is self.state_batch_:
        return self.state_dict_
      if not self.state_batch_.state_dicts and self.state_batch_.outward_states is not None:
        self.state_batch_ = StateBatch.from_states(
          self.state_batch_.outward_states,
          self.state_batch_.batch_shape,
          self.robot_,
          materialize_dict=True,
        )
      self.state_dict_ = self.state_batch_.state_dicts
      self.state_dict_source_ = self.state_batch_
      return self.state_dict_

    if self.outward_state_ is not None and hasattr(self.outward_state_, "to_state_dict"):
      if self.state_dict_source_ is self.outward_state_:
        return self.state_dict_
      self.state_dict_ = self.outward_state_.to_state_dict(self.robot_)
      self.state_dict_source_ = self.outward_state_
    return self.state_dict_

  def update_state_dict(self, order : int = None, is_dynamics: bool = False, backend : str = None) -> dict:
    self.update_state(order=order, is_dynamics=is_dynamics, backend=backend)
    return self.to_state_dict()

  def set_state_df(self):
    self._ensure_not_batched("set_state_df")
    self._ensure_state_table().import_state(self.to_state_dict())
    
  def set_target_from_file(self, target_file : str):
    if not target_file:
      raise ValueError("target_file is empty")
    if not isinstance(target_file, str):
      raise TypeError("target_file must be a string")
    self.target_ = TargetList.from_dict(
      load_json_file(target_file),
      RobotNames(self.robot_.joint_names, self.robot_.link_names, self._active_joint_names()),
    )
    self.set_order(self.target_._max_order)

  def link_diff_kinematics_numerical(self, link_name_list : list[str], data_type = "vel", order = None, eps = 1e-8, update_method = "poly", update_direction = None):
    if order is None:
      order = self.order_
    self._ensure_not_batched("link_diff_kinematics_numerical")
    
    motion = self.motion(order)

    return outward_api.link_diff_kinematics_numerical(self.robot_, motion, link_name_list, data_type, order, eps, update_method, update_direction)
  
  def diff_outward_numerical(self, state_type : StateType, order : int = None, eps : float = 1e-8, update_method : str = "poly", update_direction = None):
    if order is None:
      order = self.order_
    self._ensure_not_batched("diff_outward_numerical")

    motion = self.motion(order)
    
    return outward_api.diff_outward_numerical(self.robot_, motion, state_type, order, eps, update_method, update_direction)

  def _active_joint_names(self):
    return [joint.name for joint in self.robot_.joints if joint.dof > 0]

  def _state_type_list(self, state_type):
    state_type_list = state_type if type(state_type) is list else [state_type]
    expanded = []
    for st in state_type_list:
      if st.owner_type != "total_joint":
        expanded.append(st)
        continue
      if st.data_type not in keys_joint_motion and not is_in_keys_dynamics([st.data_type]):
        raise ValueError("total_joint state types support joint motion or dynamics data types")
      expanded.extend(
        StateType(
          owner_type="joint",
          owner_name=joint_name,
          data_type=st.data_type,
          frame_name=st.frame_name,
        )
        for joint_name in self._active_joint_names()
      )
    return expanded

  def _joint_motion_index(self, data_type : str):
    order_map = {
      "coord": 0,
      "veloc": 1,
      "accel": 2,
      "jerk": 3,
    }
    return order_map.get(data_type)

  def _joint_motion_state_info_list(self, state_type_list):
    if not any(st.owner_type == "joint" and st.data_type in keys_joint_motion for st in state_type_list):
      return None
    values = []
    motion = self.motion(self.order_)
    for st in state_type_list:
      if st.owner_type == "joint" and st.data_type in keys_joint_motion:
        joint = self.robot_.joint(st.owner_name)
        if joint is None or joint.dof <= 0:
          raise ValueError(f"Invalid active joint for joint motion state: {st.owner_name}")
        motion_index = self._joint_motion_index(st.data_type)
        if motion_index is None or motion_index >= self.order_:
          raise ValueError(f"{st.data_type} is not available for order={self.order_}")
        start = joint.dof_index * self.order_ + motion_index * joint.dof
        values.append(np.asarray(motion[..., start:start + joint.dof]))
      else:
        values.append(outward_api.get_value(self.robot_, self._state_for_direct_read(), st))
    return values

  def _sample_motions(self, motion : np.ndarray, order : int) -> RobotMotions:
    sample_motions = RobotMotions(
      self.robot_.dof,
      self.motions_.aliases[:order],
      owner_layout=self.robot_.motion_owners(),
    )
    sample_motions.set_motion(motion)
    return sample_motions

  def _jacobian_numerical(self, state_type_list, max_order : int, list_output : bool = False):
    if not self.motions_.is_batched():
      jacobs = [outward_api.jacobian_numerical(self.robot_, self.motions_, st, max_order) for st in state_type_list]
      return jacobs if list_output else np.vstack(jacobs)

    flat_motion, batch_shape = batch_api.flatten_feature_batch(self.motion(max_order))
    sample_results = [
      [
        outward_api.jacobian_numerical(self.robot_, self._sample_motions(x, max_order), st, max_order)
        for st in state_type_list
      ]
      for x in flat_motion
    ]
    return batch_api.stack_results(
      sample_results,
      batch_shape,
      list_output,
      len(state_type_list),
      combine=np.vstack,
    )

  def _jacobian_from_state(self, state, state_type_list, max_order : int, list_output : bool = False):
    fast = self._joint_motion_torque_jacobian(state, state_type_list, max_order, list_output=list_output)
    if fast is not None:
      return fast
    fast = self._rust_torque_jacobian(state_type_list, max_order, list_output=list_output)
    if fast is not None:
      return fast
    fast = self._rust_link_local_jacobian(state_type_list, max_order, list_output=list_output)
    if fast is not None:
      return fast

    if not isinstance(state, list):
      if self.batch_shape_:
        try:
          return outward_api.outward_jacobian(self.robot_, state, state_type_list, max_time_order=max_order, dim = self.dim_, list_output = list_output)
        except (AttributeError, IndexError, TypeError, ValueError):
          # Fall back when the cached state cannot be consumed as a batched
          # outward state. Unexpected runtime errors should still surface.
          pass
        is_dynamics = any(st.is_dynamics for st in state_type_list)
        _, build_state = self._state_builder(max_order, is_dynamics=is_dynamics)
        flat_motion, batch_shape = batch_api.flatten_feature_batch(self.motion(max_order))
        states = [build_state(x) for x in flat_motion]
        sample_results = [
          outward_api.outward_jacobian(self.robot_, st, state_type_list, max_time_order=max_order, dim = self.dim_, list_output = list_output)
          for st in states
        ]
        return batch_api.stack_results(
          sample_results,
          batch_shape,
          list_output,
          len(state_type_list),
        )
      return outward_api.outward_jacobian(self.robot_, state, state_type_list, dim = self.dim_, list_output = list_output)

    sample_results = [
      outward_api.outward_jacobian(self.robot_, st, state_type_list, max_time_order=max_order, dim = self.dim_, list_output = list_output)
      for st in state
    ]
    return batch_api.stack_results(
      sample_results,
      self.batch_shape_,
      list_output,
      len(state_type_list),
    )

  def _jacobian_matvec_numerical(self, state_type_list, max_order : int, vec, list_output : bool = False):
    if not self.motions_.is_batched():
      results = [outward_api.jacobian_numerical(self.robot_, self.motions_, st, max_order) @ vec for st in state_type_list]
      return results if list_output else np.concatenate(results)

    flat_motion, batch_shape = batch_api.flatten_feature_batch(self.motion(max_order))
    sample_results = []
    for x, v in zip(flat_motion, vec):
      sample_motions = self._sample_motions(x, max_order)
      parts = [
        outward_api.jacobian_numerical(self.robot_, sample_motions, st, max_order) @ v
        for st in state_type_list
      ]
      sample_results.append(parts if list_output else np.concatenate(parts))
    return batch_api.stack_results(
      sample_results,
      batch_shape,
      list_output,
      len(state_type_list),
    )

  def _jacobian_matvec_from_state(self, state, state_type_list, max_order : int, vec, batch_shape : tuple, list_output : bool = False):
    fast = self._joint_motion_torque_jacobian_apply(state, state_type_list, max_order, vec, batch_shape, rhs_is_matrix=False, list_output=list_output)
    if fast is not None:
      return fast
    fast = self._rust_torque_jacobian_apply(state_type_list, max_order, vec, batch_shape, rhs_is_matrix=False, list_output=list_output)
    if fast is not None:
      return fast
    fast = self._rust_link_local_jacobian_apply(state_type_list, max_order, vec, batch_shape, rhs_is_matrix=False, list_output=list_output)
    if fast is not None:
      return fast

    if not isinstance(state, list):
      if batch_shape:
        try:
          direct_vec = vec.reshape(batch_shape + (vec.shape[-1],))
          return outward_api.outward_jacobian_matvec(self.robot_, state, state_type_list, direct_vec, max_time_order=max_order, dim = self.dim_, list_output = list_output)
        except (AttributeError, IndexError, TypeError, ValueError):
          # Fall back when the cached state cannot be consumed as a batched
          # outward state. Unexpected runtime errors should still surface.
          pass
        is_dynamics = any(st.is_dynamics for st in state_type_list)
        _, build_state = self._state_builder(max_order, is_dynamics=is_dynamics)
        flat_motion, _ = batch_api.flatten_feature_batch(self.motion(max_order))
        states = [build_state(x) for x in flat_motion]
        sample_results = [
          outward_api.outward_jacobian_matvec(self.robot_, st, state_type_list, v, max_time_order=max_order, dim = self.dim_, list_output = list_output)
          for st, v in zip(states, vec)
        ]
        return batch_api.stack_results(
          sample_results,
          batch_shape,
          list_output,
          len(state_type_list),
        )
      return outward_api.outward_jacobian_matvec(self.robot_, state, state_type_list, vec, dim = self.dim_, list_output = list_output)

    sample_results = [
      outward_api.outward_jacobian_matvec(self.robot_, st, state_type_list, v, max_time_order=max_order, dim = self.dim_, list_output = list_output)
      for st, v in zip(state, vec)
    ]
    return batch_api.stack_results(
      sample_results,
      batch_shape,
      list_output,
      len(state_type_list),
    )

  def _jacobian_matmul_rhs_from_state(self, state, state_type_list, max_order : int, rhs, batch_shape : tuple, list_output : bool = False):
    fast = self._joint_motion_torque_jacobian_apply(state, state_type_list, max_order, rhs, batch_shape, rhs_is_matrix=True, list_output=list_output)
    if fast is not None:
      return fast
    fast = self._rust_torque_jacobian_apply(state_type_list, max_order, rhs, batch_shape, rhs_is_matrix=True, list_output=list_output)
    if fast is not None:
      return fast
    fast = self._rust_link_local_jacobian_apply(state_type_list, max_order, rhs, batch_shape, rhs_is_matrix=True, list_output=list_output)
    if fast is not None:
      return fast

    if not isinstance(state, list):
      if batch_shape:
        try:
          direct_rhs = rhs.reshape(batch_shape + rhs.shape[-2:])
          return outward_api.outward_jacobian_matmul_rhs(self.robot_, state, state_type_list, direct_rhs, max_time_order=max_order, dim = self.dim_, list_output = list_output)
        except (AttributeError, IndexError, TypeError, ValueError):
          # Fall back when the cached state cannot be consumed as a batched
          # outward state. Unexpected runtime errors should still surface.
          pass
        is_dynamics = any(st.is_dynamics for st in state_type_list)
        _, build_state = self._state_builder(max_order, is_dynamics=is_dynamics)
        flat_motion, _ = batch_api.flatten_feature_batch(self.motion(max_order))
        states = [build_state(x) for x in flat_motion]
        sample_results = [
          outward_api.outward_jacobian_matmul_rhs(self.robot_, st, state_type_list, r, max_time_order=max_order, dim = self.dim_, list_output = list_output)
          for st, r in zip(states, rhs)
        ]
        return batch_api.stack_results(
          sample_results,
          batch_shape,
          list_output,
          len(state_type_list),
        )
      return outward_api.outward_jacobian_matmul_rhs(self.robot_, state, state_type_list, rhs, dim = self.dim_, list_output = list_output)

    sample_results = [
      outward_api.outward_jacobian_matmul_rhs(self.robot_, st, state_type_list, r, max_time_order=max_order, dim = self.dim_, list_output = list_output)
      for st, r in zip(state, rhs)
    ]
    return batch_api.stack_results(
      sample_results,
      batch_shape,
      list_output,
      len(state_type_list),
    )

  def _stack_mul_columns(self, column_results, list_output : bool, item_count : int):
    if list_output:
      return [
        np.stack([column[i] for column in column_results], axis=-1)
        for i in range(item_count)
      ]
    return np.stack(column_results, axis=-1)

  def _jacobian_mul_numerical(self, state_type_list, max_order : int, rhs, rhs_is_matrix : bool, list_output : bool = False):
    if not rhs_is_matrix:
      return self._jacobian_matvec_numerical(state_type_list, max_order, rhs, list_output)

    rhs_count = rhs.shape[-1]
    column_results = [
      self._jacobian_matvec_numerical(state_type_list, max_order, rhs[..., i], list_output)
      for i in range(rhs_count)
    ]
    return self._stack_mul_columns(column_results, list_output, len(state_type_list))

  def _jacobian_mul_from_state(self, state, state_type_list, max_order : int, rhs, batch_shape : tuple, rhs_is_matrix : bool, list_output : bool = False):
    if not rhs_is_matrix:
      return self._jacobian_matvec_from_state(state, state_type_list, max_order, rhs, batch_shape, list_output)

    if not isinstance(state, list):
      return self._jacobian_matmul_rhs_from_state(state, state_type_list, max_order, rhs, batch_shape, list_output)

    rhs_count = rhs.shape[-1]
    column_results = [
      self._jacobian_matvec_from_state(state, state_type_list, max_order, rhs[..., i], batch_shape, list_output)
      for i in range(rhs_count)
    ]
    return self._stack_mul_columns(column_results, list_output, len(state_type_list))

  def _jacobian_output_dim(self, state_type_list) -> int:
    dim_dof = dim_to_dof(self.dim_)
    output_dim = 0
    for st in state_type_list:
      if self._is_joint_motion_state(st):
        joint = self.robot_.joint(st.owner_name)
        if joint is None:
          raise ValueError(f"Invalid joint name: {st.owner_name}")
        output_dim += joint.dof
      elif st.data_type in keys_kinematics:
        output_dim += data_type_dof(st.data_type, dim=self.dim_)
      elif st.data_type in keys_momentum or st.data_type in keys_force:
        output_dim += dim_dof
      elif st.data_type in keys_torque:
        if st.owner_type != "joint":
          raise ValueError("torque can be specified only for joint owner type")
        joint = self.robot_.joint(st.owner_name)
        if joint is None:
          raise ValueError(f"Invalid joint name: {st.owner_name}")
        output_dim += joint.dof
      else:
        raise ValueError(f"Unsupported data_type: {st.data_type}")
    return output_dim

  def _jacobian_transpose_matvec_numerical(self, state_type_list, max_order : int, vec):
    if not self.motions_.is_batched():
      jacob = self._jacobian_numerical(state_type_list, max_order)
      return jacob.T @ vec

    flat_motion, batch_shape = batch_api.flatten_feature_batch(self.motion(max_order))
    sample_results = []
    for x, v in zip(flat_motion, vec):
      sample_motions = self._sample_motions(x, max_order)
      parts = [
        outward_api.jacobian_numerical(self.robot_, sample_motions, st, max_order)
        for st in state_type_list
      ]
      jacob = np.vstack(parts)
      sample_results.append(jacob.T @ v)
    return batch_api.stack_sample_results(sample_results, batch_shape)

  def _jacobian_transpose_matvec_from_state(self, state, state_type_list, max_order : int, vec, batch_shape : tuple):
    fast = self._joint_motion_torque_jacobian_transpose_apply(state, state_type_list, max_order, vec, batch_shape, rhs_is_matrix=False)
    if fast is not None:
      return fast
    fast = self._rust_torque_jacobian_transpose_apply(state_type_list, max_order, vec, batch_shape, rhs_is_matrix=False)
    if fast is not None:
      return fast
    fast = self._rust_link_local_jacobian_transpose_apply(state_type_list, max_order, vec, batch_shape, rhs_is_matrix=False)
    if fast is not None:
      return fast

    if not isinstance(state, list):
      if batch_shape:
        try:
          direct_vec = vec.reshape(batch_shape + (vec.shape[-1],))
          return outward_api.outward_jacobian_transpose_matvec(
            self.robot_,
            state,
            state_type_list,
            direct_vec,
            max_time_order=max_order,
            dim=self.dim_,
          )
        except (AttributeError, IndexError, TypeError, ValueError):
          # Fall back when the cached state cannot be consumed as a batched
          # outward state. Unexpected runtime errors should still surface.
          pass
        is_dynamics = any(st.is_dynamics for st in state_type_list)
        _, build_state = self._state_builder(max_order, is_dynamics=is_dynamics)
        flat_motion, _ = batch_api.flatten_feature_batch(self.motion(max_order))
        states = [build_state(x) for x in flat_motion]
        sample_results = [
          outward_api.outward_jacobian_transpose_matvec(self.robot_, st, state_type_list, v, max_time_order=max_order, dim=self.dim_)
          for st, v in zip(states, vec)
        ]
        return batch_api.stack_sample_results(sample_results, batch_shape)
      return outward_api.outward_jacobian_transpose_matvec(self.robot_, state, state_type_list, vec, dim=self.dim_)

    sample_results = [
      outward_api.outward_jacobian_transpose_matvec(self.robot_, st, state_type_list, v, max_time_order=max_order, dim=self.dim_)
      for st, v in zip(state, vec)
    ]
    return batch_api.stack_sample_results(sample_results, batch_shape)

  def _jacobian_transpose_mul_numerical(self, state_type_list, max_order : int, rhs, rhs_is_matrix : bool):
    if not rhs_is_matrix:
      return self._jacobian_transpose_matvec_numerical(state_type_list, max_order, rhs)

    rhs_count = rhs.shape[-1]
    column_results = [
      self._jacobian_transpose_matvec_numerical(state_type_list, max_order, rhs[..., i])
      for i in range(rhs_count)
    ]
    return np.stack(column_results, axis=-1)

  def _jacobian_transpose_mul_from_state(self, state, state_type_list, max_order : int, rhs, batch_shape : tuple, rhs_is_matrix : bool):
    if not rhs_is_matrix:
      return self._jacobian_transpose_matvec_from_state(state, state_type_list, max_order, rhs, batch_shape)

    fast = self._joint_motion_torque_jacobian_transpose_apply(state, state_type_list, max_order, rhs, batch_shape, rhs_is_matrix=True)
    if fast is not None:
      return fast
    fast = self._rust_torque_jacobian_transpose_apply(state_type_list, max_order, rhs, batch_shape, rhs_is_matrix=True)
    if fast is not None:
      return fast
    fast = self._rust_link_local_jacobian_transpose_apply(state_type_list, max_order, rhs, batch_shape, rhs_is_matrix=True)
    if fast is not None:
      return fast

    if (
      not batch_shape
      and not isinstance(state, list)
    ):
      block_result = self._jacobian_transpose_matvec_from_state(
        state,
        state_type_list,
        max_order,
        np.swapaxes(rhs, -1, -2),
        batch_shape,
      )
      return np.swapaxes(block_result, -1, -2)

    if batch_shape and not isinstance(state, list):
      jacob = self._jacobian_from_state(state, state_type_list, max_order)
      rhs_batch = rhs.reshape(batch_shape + rhs.shape[-2:])
      return np.swapaxes(jacob, -1, -2) @ rhs_batch

    rhs_count = rhs.shape[-1]
    column_results = [
      self._jacobian_transpose_matvec_from_state(state, state_type_list, max_order, rhs[..., i], batch_shape)
      for i in range(rhs_count)
    ]
    return np.stack(column_results, axis=-1)

  def _is_joint_motion_state(self, st : StateType) -> bool:
    return st.owner_type == "joint" and st.data_type in keys_joint_motion

  def _is_joint_torque_state(self, st : StateType) -> bool:
    return st.owner_type == "joint" and st.data_type in keys_torque and st.frame_name is None

  def _joint_motion_torque_supported(self, state_type_list) -> bool:
    has_motion = False
    for st in state_type_list:
      if self._is_joint_motion_state(st):
        has_motion = True
      elif not self._is_joint_torque_state(st):
        return False
    return has_motion

  def _joint_state_dof(self, st : StateType) -> int:
    joint = self.robot_.joint(st.owner_name)
    if joint is None or joint.dof <= 0:
      raise ValueError(f"Invalid active joint state: {st.owner_name}")
    return joint.dof

  def _joint_motion_col_slice(self, st : StateType, max_order : int) -> slice:
    joint = self.robot_.joint(st.owner_name)
    if joint is None or joint.dof <= 0:
      raise ValueError(f"Invalid active joint motion state: {st.owner_name}")
    motion_index = self._joint_motion_index(st.data_type)
    if motion_index is None or motion_index >= max_order:
      raise ValueError(f"{st.data_type} is not available for order={max_order}")
    start = joint.dof_index * max_order + motion_index * joint.dof
    return slice(start, start + joint.dof)

  def _joint_motion_selector_jacobian(self, st : StateType, max_order : int, batch_shape : tuple):
    joint = self.robot_.joint(st.owner_name)
    col_slice = self._joint_motion_col_slice(st, max_order)
    out = np.zeros(tuple(batch_shape) + (joint.dof, self.robot_.dof * max_order), dtype=float)
    diag = np.arange(joint.dof)
    out[..., diag, col_slice.start + diag] = 1.0
    return out

  def _joint_torque_jacobian_parts(self, state, states : list[StateType], max_order : int):
    if not states:
      return []
    fast = self._rust_torque_jacobian(states, max_order, list_output=True)
    if fast is not None:
      return fast
    return outward_api.outward_jacobian(
      self.robot_,
      state,
      states,
      max_time_order=max_order,
      dim=self.dim_,
      list_output=True,
    )

  def _joint_motion_torque_jacobian(self, state, state_type_list, max_order : int, list_output : bool = False):
    if not self._joint_motion_torque_supported(state_type_list):
      return None
    batch_shape = self.batch_shape_
    parts = [None] * len(state_type_list)
    torque_indices = []
    torque_states = []
    for i, st in enumerate(state_type_list):
      if self._is_joint_motion_state(st):
        parts[i] = self._joint_motion_selector_jacobian(st, max_order, batch_shape)
      else:
        torque_indices.append(i)
        torque_states.append(st)
    for i, part in zip(torque_indices, self._joint_torque_jacobian_parts(state, torque_states, max_order)):
      parts[i] = part
    if list_output:
      return parts
    return np.concatenate(parts, axis=-2)

  def _joint_torque_jacobian_apply_parts(self, state, states : list[StateType], max_order : int, rhs, batch_shape : tuple, rhs_is_matrix : bool):
    if not states:
      return []
    fast = self._rust_torque_jacobian_apply(states, max_order, rhs, batch_shape, rhs_is_matrix=rhs_is_matrix, list_output=True)
    if fast is not None:
      return fast
    if rhs_is_matrix:
      direct_rhs = rhs.reshape(batch_shape + rhs.shape[-2:]) if batch_shape else rhs
      return outward_api.outward_jacobian_matmul_rhs(
        self.robot_,
        state,
        states,
        direct_rhs,
        max_time_order=max_order,
        dim=self.dim_,
        list_output=True,
      )
    direct_vec = rhs.reshape(batch_shape + (rhs.shape[-1],)) if batch_shape else rhs
    return outward_api.outward_jacobian_matvec(
      self.robot_,
      state,
      states,
      direct_vec,
      max_time_order=max_order,
      dim=self.dim_,
      list_output=True,
    )

  def _joint_motion_torque_jacobian_apply(self, state, state_type_list, max_order : int, rhs, batch_shape : tuple, rhs_is_matrix : bool, list_output : bool = False):
    if not self._joint_motion_torque_supported(state_type_list):
      return None
    rhs = np.asarray(rhs)
    if rhs_is_matrix:
      rhs_matrix = rhs.reshape(batch_shape + rhs.shape[-2:]) if batch_shape else rhs
    else:
      rhs_vec = rhs.reshape(batch_shape + (rhs.shape[-1],)) if batch_shape else rhs
    parts = [None] * len(state_type_list)
    torque_indices = []
    torque_states = []
    for i, st in enumerate(state_type_list):
      if self._is_joint_motion_state(st):
        col_slice = self._joint_motion_col_slice(st, max_order)
        parts[i] = rhs_matrix[..., col_slice, :] if rhs_is_matrix else rhs_vec[..., col_slice]
      else:
        torque_indices.append(i)
        torque_states.append(st)
    for i, part in zip(torque_indices, self._joint_torque_jacobian_apply_parts(state, torque_states, max_order, rhs, batch_shape, rhs_is_matrix)):
      parts[i] = part
    if list_output:
      return parts
    return np.concatenate(parts, axis=-2 if rhs_is_matrix else -1)

  def _joint_torque_jacobian_transpose_apply_parts(self, state, states : list[StateType], rhs_parts : list[np.ndarray], max_order : int, batch_shape : tuple, rhs_is_matrix : bool):
    if not states:
      return None
    if len(states) == 1:
      fast = self._rust_torque_jacobian_transpose_apply(states, max_order, rhs_parts[0], batch_shape, rhs_is_matrix=rhs_is_matrix)
      if fast is not None:
        return fast

    if rhs_is_matrix:
      rhs_data = [
        part.reshape(batch_shape + part.shape[-2:]) if batch_shape else part
        for part in rhs_parts
      ]
      combined = np.concatenate(rhs_data, axis=-2)
      if not batch_shape:
        packed = outward_api.outward_jacobian_transpose_matvec(
          self.robot_,
          state,
          states,
          np.moveaxis(combined, -1, 0),
          max_time_order=max_order,
          dim=self.dim_,
        )
        return np.moveaxis(packed, 0, -1)
      cols = [
        outward_api.outward_jacobian_transpose_matvec(
          self.robot_,
          state,
          states,
          combined[..., i],
          max_time_order=max_order,
          dim=self.dim_,
        )
        for i in range(combined.shape[-1])
      ]
      return np.stack(cols, axis=-1)

    rhs_data = [
      part.reshape(batch_shape + (part.shape[-1],)) if batch_shape else part
      for part in rhs_parts
    ]
    combined = np.concatenate(rhs_data, axis=-1)
    return outward_api.outward_jacobian_transpose_matvec(
      self.robot_,
      state,
      states,
      combined,
      max_time_order=max_order,
      dim=self.dim_,
    )

  def _joint_motion_torque_jacobian_transpose_apply(self, state, state_type_list, max_order : int, rhs, batch_shape : tuple, rhs_is_matrix : bool):
    if not self._joint_motion_torque_supported(state_type_list):
      return None
    rhs = np.asarray(rhs)
    if rhs_is_matrix:
      rhs_data = rhs.reshape(batch_shape + rhs.shape[-2:]) if batch_shape else rhs
      result = np.zeros(tuple(batch_shape) + (self.robot_.dof * max_order, rhs_data.shape[-1]), dtype=rhs_data.dtype)
    else:
      rhs_data = rhs.reshape(batch_shape + (rhs.shape[-1],)) if batch_shape else rhs
      result = np.zeros(tuple(batch_shape) + (self.robot_.dof * max_order,), dtype=rhs_data.dtype)
    row_start = 0
    torque_states = []
    torque_rhs_parts = []
    for st in state_type_list:
      rows = self._joint_state_dof(st)
      rhs_part = rhs_data[..., row_start:row_start + rows, :] if rhs_is_matrix else rhs_data[..., row_start:row_start + rows]
      if self._is_joint_motion_state(st):
        col_slice = self._joint_motion_col_slice(st, max_order)
        if rhs_is_matrix:
          result[..., col_slice, :] += rhs_part
        else:
          result[..., col_slice] += rhs_part
      else:
        torque_states.append(st)
        torque_rhs_parts.append(rhs_part)
      row_start += rows
    if torque_states:
      result += self._joint_torque_jacobian_transpose_apply_parts(
        state,
        torque_states,
        torque_rhs_parts,
        max_order,
        batch_shape,
        rhs_is_matrix,
      )
    if not rhs_is_matrix:
      return result
    return result

  def _rust_torque_row_parts(self, state_type_list, max_order : int):
    if self.dim_ != 3 or max_order != 3:
      return None
    rows = []
    part_sizes = []
    for st in state_type_list:
      if st.owner_type != "joint" or st.data_type != "torque" or st.frame_name is not None:
        return None
      joint = self.robot_.joint(st.owner_name)
      if joint is None or joint.dof <= 0:
        return None
      rows.extend(range(joint.dof_index, joint.dof_index + joint.dof))
      part_sizes.append(joint.dof)
    return rows, part_sizes

  def _rust_qva_order3(self):
    motion = np.asarray(self.motion(3), dtype=float)
    if motion.shape[-1] != self.robot_.dof * 3:
      return None
    if motion.ndim == 1:
      return (
        np.ascontiguousarray(motion[0::3]),
        np.ascontiguousarray(motion[1::3]),
        np.ascontiguousarray(motion[2::3]),
        (),
      )
    batch_shape = motion.shape[:-1]
    flat = motion.reshape((-1, motion.shape[-1]))
    return (
      np.ascontiguousarray(flat[:, 0::3]),
      np.ascontiguousarray(flat[:, 1::3]),
      np.ascontiguousarray(flat[:, 2::3]),
      batch_shape,
    )

  def _rust_torque_jacobian(self, state_type_list, max_order : int, list_output : bool = False):
    if not hasattr(self.outward_state_, "raw_data"):
      return None
    spec = self._rust_torque_row_parts(state_type_list, max_order)
    if spec is None:
      return None
    rows, part_sizes = spec
    qva = self._rust_qva_order3()
    if qva is None:
      return None
    q, v, a, batch_shape = qva
    try:
      if batch_shape:
        jacob = np.asarray(self._rust_compiled_robot().dynamics_jacobian_batch(q, v, a))
        jacob = jacob.reshape(batch_shape + jacob.shape[-2:])
      else:
        jacob = np.asarray(self._rust_compiled_robot().dynamics_jacobian(q, v, a))
    except Exception:
      return None
    selected = jacob[..., rows, :]
    if not list_output:
      return selected
    parts = []
    start = 0
    for size in part_sizes:
      parts.append(selected[..., start:start + size, :])
      start += size
    return parts

  def _rust_torque_jacobian_apply(self, state_type_list, max_order : int, rhs, batch_shape : tuple, rhs_is_matrix : bool, list_output : bool = False):
    spec = self._rust_torque_row_parts(state_type_list, max_order)
    if spec is None or not hasattr(self.outward_state_, "raw_data"):
      return None
    rows, part_sizes = spec
    qva = self._rust_qva_order3()
    if qva is None:
      return None
    q, v, a, motion_batch_shape = qva
    if tuple(batch_shape) != tuple(motion_batch_shape):
      return None
    try:
      if rhs_is_matrix:
        rhs_matrix = rhs.reshape(batch_shape + rhs.shape[-2:]) if batch_shape else rhs
      else:
        rhs_vec = rhs.reshape(batch_shape + (rhs.shape[-1],)) if batch_shape else rhs
        rhs_matrix = rhs_vec[..., :, None]
      if batch_shape:
        flat_rhs = np.ascontiguousarray(rhs_matrix.reshape((-1,) + rhs_matrix.shape[-2:]))
        applied = np.asarray(self._rust_compiled_robot().dynamics_jacobian_matmul_rhs_batch(q, v, a, flat_rhs))
        applied = applied.reshape(batch_shape + applied.shape[-2:])
      else:
        applied = np.asarray(self._rust_compiled_robot().dynamics_jacobian_matmul_rhs(q, v, a, np.ascontiguousarray(rhs_matrix)))
    except Exception:
      return None
    selected = applied[..., rows, :]

    if not rhs_is_matrix:
      selected = selected[..., 0]

    if list_output:
      parts = []
      start = 0
      for size in part_sizes:
        parts.append(selected[..., start:start + size] if not rhs_is_matrix else selected[..., start:start + size, :])
        start += size
      return parts
    return selected

  def _rust_torque_jacobian_transpose_apply(self, state_type_list, max_order : int, rhs, batch_shape : tuple, rhs_is_matrix : bool):
    spec = self._rust_torque_row_parts(state_type_list, max_order)
    if spec is None or not hasattr(self.outward_state_, "raw_data"):
      return None
    rows, _ = spec
    qva = self._rust_qva_order3()
    if qva is None:
      return None
    q, v, a, motion_batch_shape = qva
    if tuple(batch_shape) != tuple(motion_batch_shape):
      return None
    try:
      if rhs_is_matrix:
        rhs_part = rhs.reshape(batch_shape + rhs.shape[-2:]) if batch_shape else rhs
      else:
        rhs_vec = rhs.reshape(batch_shape + (rhs.shape[-1],)) if batch_shape else rhs
        rhs_part = rhs_vec[..., :, None]
      if len(rows) == self.robot_.dof and rows == list(range(self.robot_.dof)):
        full_rhs = rhs_part
      else:
        full_rhs = np.zeros(rhs_part.shape[:-2] + (self.robot_.dof, rhs_part.shape[-1]), dtype=rhs_part.dtype)
        full_rhs[..., rows, :] = rhs_part
      if batch_shape:
        flat_rhs = np.ascontiguousarray(full_rhs.reshape((-1,) + full_rhs.shape[-2:]))
        out = np.asarray(self._rust_compiled_robot().dynamics_jacobian_transpose_matmul_rhs_batch(q, v, a, flat_rhs))
        out = out.reshape(batch_shape + out.shape[-2:])
      else:
        out = np.asarray(self._rust_compiled_robot().dynamics_jacobian_transpose_matmul_rhs(q, v, a, np.ascontiguousarray(full_rhs)))
    except Exception:
      return None
    if rhs_is_matrix:
      return out
    return out[..., 0]

  def _rust_link_local_specs(self, state_type_list, max_order : int):
    if self.dim_ != 3 or max_order != 3:
      return None
    code_map = {
      "vel": 0,
      "acc": 1,
      "momentum": 2,
      "momentum_diff1": 3,
      "force": 4,
    }
    link_ids = []
    data_codes = []
    for st in state_type_list:
      if st.owner_type != "link" or st.data_type not in code_map or st.frame_name is not None:
        return None
      link = self.robot_.link(st.owner_name)
      if link is None:
        return None
      link_ids.append(link.id)
      data_codes.append(code_map[st.data_type])
    return np.asarray(link_ids, dtype=np.int64), np.asarray(data_codes, dtype=np.int64)

  def _rust_link_local_jacobian(self, state_type_list, max_order : int, list_output : bool = False):
    if not hasattr(self.outward_state_, "raw_data"):
      return None
    spec = self._rust_link_local_specs(state_type_list, max_order)
    if spec is None:
      return None
    link_ids, data_codes = spec
    qva = self._rust_qva_order3()
    if qva is None:
      return None
    q, v, a, batch_shape = qva
    try:
      if batch_shape:
        parts = [
          np.asarray(self._rust_compiled_robot().link_local_jacobian(q[i], v[i], a[i], link_ids, data_codes))
          for i in range(q.shape[0])
        ]
        jacob = np.stack(parts, axis=0).reshape(batch_shape + parts[0].shape)
      else:
        jacob = np.asarray(self._rust_compiled_robot().link_local_jacobian(q, v, a, link_ids, data_codes))
    except Exception:
      return None
    if not list_output:
      return jacob
    return [jacob[..., i * 6:(i + 1) * 6, :] for i in range(len(link_ids))]

  def _rust_link_local_jacobian_apply(self, state_type_list, max_order : int, rhs, batch_shape : tuple, rhs_is_matrix : bool, list_output : bool = False):
    if not hasattr(self.outward_state_, "raw_data"):
      return None
    spec = self._rust_link_local_specs(state_type_list, max_order)
    if spec is None:
      return None
    link_ids, data_codes = spec
    qva = self._rust_qva_order3()
    if qva is None:
      return None
    q, v, a, motion_batch_shape = qva
    if tuple(batch_shape) != tuple(motion_batch_shape):
      return None
    try:
      if rhs_is_matrix:
        rhs_matrix = rhs.reshape(batch_shape + rhs.shape[-2:]) if batch_shape else rhs
      else:
        rhs_vec = rhs.reshape(batch_shape + (rhs.shape[-1],)) if batch_shape else rhs
        rhs_matrix = rhs_vec[..., :, None]
      if batch_shape:
        flat_rhs = np.ascontiguousarray(rhs_matrix.reshape((-1,) + rhs_matrix.shape[-2:]))
        parts = [
          np.asarray(self._rust_compiled_robot().link_local_jacobian_matmul_rhs(q[i], v[i], a[i], flat_rhs[i], link_ids, data_codes))
          for i in range(q.shape[0])
        ]
        applied = np.stack(parts, axis=0).reshape(batch_shape + parts[0].shape)
      else:
        applied = np.asarray(self._rust_compiled_robot().link_local_jacobian_matmul_rhs(q, v, a, np.ascontiguousarray(rhs_matrix), link_ids, data_codes))
    except Exception:
      return None
    if not rhs_is_matrix:
      applied = applied[..., 0]
    if list_output:
      return [applied[..., i * 6:(i + 1) * 6] if not rhs_is_matrix else applied[..., i * 6:(i + 1) * 6, :] for i in range(len(link_ids))]
    return applied

  def _rust_link_local_jacobian_transpose_apply(self, state_type_list, max_order : int, rhs, batch_shape : tuple, rhs_is_matrix : bool):
    jacob = self._rust_link_local_jacobian(state_type_list, max_order, list_output=False)
    if jacob is None:
      return None
    jac_t = np.swapaxes(jacob, -1, -2)
    if rhs_is_matrix:
      rhs_part = rhs.reshape(batch_shape + rhs.shape[-2:]) if batch_shape else rhs
      return jac_t @ rhs_part
    rhs_part = rhs.reshape(batch_shape + (rhs.shape[-1],)) if batch_shape else rhs
    return (jac_t @ rhs_part[..., None])[..., 0]

  def jacobian(self, state_type, numerical : bool = False, list_output : bool = False):
    state_type_list = self._state_type_list(state_type)
    max_order = StateType.max_time_order(state_type_list)
    if numerical:
      return self._jacobian_numerical(state_type_list, max_order, list_output)

    state = self.outward_state_ if self.outward_state_ is not None else self.state_dict_
    return self._jacobian_from_state(state, state_type_list, max_order, list_output)

  def jacobian_mul(self, state_type, rhs : np.ndarray, numerical : bool = False, list_output : bool = False):
    """
    Compute J @ rhs for the Jacobian of ``state_type``.

    ``rhs`` may be a vector with shape ``(motion_dim,)`` or
    ``batch_shape + (motion_dim,)``, or a matrix with shape
    ``(motion_dim, rhs_dim)`` or ``batch_shape + (motion_dim, rhs_dim)``.
    """
    state_type_list = self._state_type_list(state_type)
    max_order = StateType.max_time_order(state_type_list)
    input_dim = self.robot_.dof * max_order
    batch_shape = self.batch_shape_ if self.batch_shape_ else self.motions_.batch_shape()
    rhs, rhs_is_matrix = batch_api.broadcast_feature_rhs(rhs, batch_shape, input_dim, name="rhs")

    if numerical:
      return self._jacobian_mul_numerical(state_type_list, max_order, rhs, rhs_is_matrix, list_output)

    state = self.outward_state_ if self.outward_state_ is not None else self.state_dict_
    return self._jacobian_mul_from_state(state, state_type_list, max_order, rhs, batch_shape, rhs_is_matrix, list_output)

  def jacobian_transpose_mul(self, state_type, rhs : np.ndarray, numerical : bool = False):
    """
    Compute J.T @ rhs for the Jacobian of ``state_type``.

    ``rhs`` may be a vector with shape ``(total_state_dim,)`` or
    ``batch_shape + (total_state_dim,)``, or a matrix with shape
    ``(total_state_dim, rhs_dim)`` or ``batch_shape + (total_state_dim, rhs_dim)``.
    """
    state_type_list = self._state_type_list(state_type)
    max_order = StateType.max_time_order(state_type_list)
    output_dim = self._jacobian_output_dim(state_type_list)
    batch_shape = self.batch_shape_ if self.batch_shape_ else self.motions_.batch_shape()
    rhs, rhs_is_matrix = batch_api.broadcast_feature_rhs(rhs, batch_shape, output_dim, name="rhs")

    if numerical:
      return self._jacobian_transpose_mul_numerical(state_type_list, max_order, rhs, rhs_is_matrix)

    state = self.outward_state_ if self.outward_state_ is not None else self.state_dict_
    return self._jacobian_transpose_mul_from_state(state, state_type_list, max_order, rhs, batch_shape, rhs_is_matrix)

  def jacobian_tensor(self, state_type, numerical : bool = False) -> JacobianTensor:
    state_type_list = self._state_type_list(state_type)
    return JacobianTensor.from_array(self.jacobian(state_type_list, numerical=numerical), state_type_list)
  
  def jacobian_target(self, numerical : bool = False, list_output : bool = False):
    if self.target_ is None:
      raise ValueError("target is not set")
    
    return self.jacobian(self.target_._targets, numerical=numerical, list_output=list_output)

  def jacobian_target_mul(self, rhs : np.ndarray, numerical : bool = False, list_output : bool = False):
    if self.target_ is None:
      raise ValueError("target is not set")

    return self.jacobian_mul(self.target_._targets, rhs, numerical=numerical, list_output=list_output)

  def jacobian_target_transpose_mul(self, rhs : np.ndarray, numerical : bool = False):
    if self.target_ is None:
      raise ValueError("target is not set")

    return self.jacobian_transpose_mul(self.target_._targets, rhs, numerical=numerical)

  def jacobian_target_tensor(self, numerical : bool = False) -> JacobianTensor:
    if self.target_ is None:
      raise ValueError("target is not set")
    return JacobianTensor.from_array(self.jacobian_target(numerical=numerical), self.target_._targets)
  
  def inverse_kinematics(self, target_type : List[StateType], target_value : List[np.ndarray],
                    q_init : np.ndarray, opt_func : None = None) -> np.ndarray:
    raise NotImplementedError(
      "inward module was removed. inverse_kinematics is no longer available in robokots."
    )

  def show_robot(self, save = False, ax = None, color : RobotColor = None):
    self._ensure_not_batched("show_robot")
    from .core.state_dict import state_dict_to_links_pos

    conectivity = np.zeros((self.robot_.joint_num, 2), dtype='int64')
    for i in range(self.robot_.joint_num):
      joint = self.robot_.joints[i]
      conectivity[i, 0] = joint.child_link_id
      conectivity[i, 1] = joint.parent_link_id

    show_robot(conectivity, state_dict_to_links_pos(self.to_state_dict(), self.robot_.link_names), save, ax, color)

  def show_robot_traj(self, traj = None, save = False, ax = None, color : RobotColor = None):
    conectivity = np.zeros((self.robot_.joint_num, 2), dtype='int64')
    for i in range(self.robot_.joint_num):
      joint = self.robot_.joints[i]
      conectivity[i, 0] = joint.child_link_id
      conectivity[i, 1] = joint.parent_link_id

    if traj is None:
      self._ensure_not_batched("show_robot_traj")
      link_pos_traj = self._ensure_state_table().extract_links_info_traj("pos", self.robot_.link_names)
    else:
      link_pos_traj = traj
    show_robot_traj(conectivity, link_pos_traj, save, ax, color)

  def show_link_points(self):
    self._ensure_not_batched("show_link_points")
    from .core.state_dict import state_dict_to_links_pos

    show_link_points(state_dict_to_links_pos(self.to_state_dict(), self.robot_.link_names))

  def show_target_link_points(self, plt = None, dimension=3):
    self._ensure_not_batched("show_target_link_points")
    from .core.state_dict import state_dict_to_links_pos

    if not self.target_:
      raise ValueError("target_ is not set")
    
    owner_link_names = []
    for t in self.target_._targets:
      if t._state_type.owner_type == "link":
        owner_link_names.append(t._state_type.owner_name)
    show_link_points(state_dict_to_links_pos(self.to_state_dict(), owner_link_names), plt, dimension)

  def target_link_pos_traj(self):
    self._ensure_not_batched("target_link_pos_traj")
    if not self.target_:
      raise ValueError("target_ is not set")
    
    owner_link_names = []
    for t in self.target_._targets:
      if t._state_type.owner_type == "link":
        owner_link_names.append(t._state_type.owner_name)
    return self._ensure_state_table().extract_links_info_traj("pos", owner_link_names)

  def show_points(self, points, ax = None, dimension=3):
    show_link_points(points, ax, dimension)
