#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki

import numpy as np
from typing import List, Any, Optional, Tuple

from .core.motion import RobotMotions
from .core.state import StateType
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

    print_state_dict(self.state_dict_)

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
    if isinstance(self.state_batch_, StateBatch):
      return self.state_batch_.state_info(self.robot_, state_type, outward_api.get_value)
    value = outward_api.get_value(self.robot_, self._state_for_direct_read(), state_type)
    if self.batch_shape_ and hasattr(value, "mat"):
      return value.mat()
    return value

  def state_info_list(self, state_type_list : List[StateType], list_output : bool = False) -> List[np.ndarray]:
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
    if type(state_type) is list:
      state_type_list = state_type
    else:
      state_type_list = [state_type]
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
  
  def kinematics(self, order = None, backend : str = None):
    if order is None:
      order = self.order_
    states, batch_shape = self._build_state_result(order=order, is_dynamics=False, backend=backend)
    self._set_batch_states(states, batch_shape)

  # ToDo: change function name
  def kinematics_point(self, s : float = 0.0):
    self._ensure_not_batched("kinematics_point")
    return outward_api.calc_link_total_point_frame(self.robot_, self.motions_, self.state_dict_, s)
  
  def dynamics(self, order = None):
    if order is None:
      order = self.order_
    states, batch_shape = self._build_state_result(order=order, is_dynamics=True)
    self._set_batch_states(states, batch_shape)

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
    kinematics_backend = None if is_dynamics else backend
    if kinematics_backend not in (None, "numpy", "jax"):
      raise ValueError(f"Unsupported kinematics backend: {kinematics_backend}. Use 'numpy' or 'jax'.")
    return kinematics_backend

  def _state_builder(self, order : int, is_dynamics : bool = False, backend : str = None):
    kinematics_backend = self._resolve_kinematics_backend(is_dynamics, backend)
    if is_dynamics:
      return (
        kinematics_backend,
        lambda x: outward_api.build_dynamics_outward_state(self.robot_, x, order-2),
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
      and kinematics_backend in (None, "numpy")
    ):
      try:
        if is_dynamics:
          return outward_api.build_dynamics_outward_state(self.robot_, motion, order - 2), motion.shape[:-1]
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
    self._ensure_state_table().import_state(self.state_dict_)
    
  def set_target_from_file(self, target_file : str):
    if not target_file:
      raise ValueError("target_file is empty")
    if not isinstance(target_file, str):
      raise TypeError("target_file must be a string")
    self.target_ = TargetList.from_dict(load_json_file(target_file), RobotNames(self.robot_.joint_names, self.robot_.link_names))
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

  def _state_type_list(self, state_type):
    return state_type if type(state_type) is list else [state_type]

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
    if not isinstance(state, list):
      if self.batch_shape_:
        try:
          return outward_api.outward_jacobian(self.robot_, state, state_type_list, max_time_order=max_order, dim = self.dim_, list_output = list_output)
        except Exception:
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
    if not isinstance(state, list):
      if batch_shape:
        try:
          direct_vec = vec.reshape(batch_shape + (vec.shape[-1],))
          return outward_api.outward_jacobian_matvec(self.robot_, state, state_type_list, direct_vec, max_time_order=max_order, dim = self.dim_, list_output = list_output)
        except Exception:
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

  def jacobian(self, state_type, numerical : bool = False, list_output : bool = False):
    state_type_list = self._state_type_list(state_type)
    max_order = StateType.max_time_order(state_type_list)
    if numerical:
      return self._jacobian_numerical(state_type_list, max_order, list_output)

    state = self.outward_state_ if self.outward_state_ is not None else self.state_dict_
    return self._jacobian_from_state(state, state_type_list, max_order, list_output)

  def jacobian_matvec(self, state_type, vec : np.ndarray, numerical : bool = False, list_output : bool = False):
    state_type_list = self._state_type_list(state_type)
    max_order = StateType.max_time_order(state_type_list)
    expected_shape = (self.robot_.dof * max_order,)
    batch_shape = self.batch_shape_ if self.batch_shape_ else self.motions_.batch_shape()
    vec = batch_api.broadcast_feature_vector(vec, batch_shape, expected_shape)

    if numerical:
      return self._jacobian_matvec_numerical(state_type_list, max_order, vec, list_output)

    state = self.outward_state_ if self.outward_state_ is not None else self.state_dict_
    return self._jacobian_matvec_from_state(state, state_type_list, max_order, vec, batch_shape, list_output)

  def jacobian_tensor(self, state_type, numerical : bool = False) -> JacobianTensor:
    state_type_list = self._state_type_list(state_type)
    return JacobianTensor.from_array(self.jacobian(state_type_list, numerical=numerical), state_type_list)
  
  def jacobian_target(self, numerical : bool = False, list_output : bool = False):
    if self.target_ is None:
      raise ValueError("target is not set")
    
    return self.jacobian(self.target_._targets, numerical=numerical, list_output=list_output)

  def jacobian_target_matvec(self, vec : np.ndarray, numerical : bool = False, list_output : bool = False):
    if self.target_ is None:
      raise ValueError("target is not set")

    return self.jacobian_matvec(self.target_._targets, vec, numerical=numerical, list_output=list_output)

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

    show_robot(conectivity, state_dict_to_links_pos(self.state_dict_, self.robot_.link_names), save, ax, color)

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

    show_link_points(state_dict_to_links_pos(self.state_dict_, self.robot_.link_names))

  def show_target_link_points(self, plt = None, dimension=3):
    self._ensure_not_batched("show_target_link_points")
    from .core.state_dict import state_dict_to_links_pos

    if not self.target_:
      raise ValueError("target_ is not set")
    
    owner_link_names = []
    for t in self.target_._targets:
      if t._state_type.owner_type == "link":
        owner_link_names.append(t._state_type.owner_name)
    show_link_points(state_dict_to_links_pos(self.state_dict_, owner_link_names), plt, dimension)

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
