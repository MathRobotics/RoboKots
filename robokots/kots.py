#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki

import numpy as np
from typing import List, Any, Optional

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
from .api import DerivativesMixin, FastDerivativesMixin, InwardDynamicsMixin, OutwardDynamicsMixin, RustBackendMixin, RustDerivativesMixin, StateManagementMixin

default_order = 3 
default_dim = 3
class Kots(DerivativesMixin, RustDerivativesMixin, FastDerivativesMixin, RustBackendMixin, InwardDynamicsMixin, OutwardDynamicsMixin, StateManagementMixin):
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
  state_cache_config_ : Optional[tuple]

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
    self._rust_inverse_dynamics_robot_ = None
    self._rust_outward_data_cache_ = {}
    self._rust_outward_data_cache_state_ = {}
    self.gravity_ = np.zeros(3, dtype=float)

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
    self._rust_inverse_dynamics_robot_ = None
    self._rust_outward_data_cache_ = {}
    self._rust_outward_data_cache_state_ = {}
    self.gravity_ = np.zeros(3, dtype=float)

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

  @staticmethod
  def _is_total_body_kinetic_energy(state_type) -> bool:
    return (
      state_type.owner_type == "total_body"
      and state_type.owner_name == "total_body"
      and state_type.data_type == "kinetic_energy"
      and state_type.frame_name is None
    )

  def _select_motion_order_rhs(self, rhs, source_order: int, target_order: int, rhs_is_matrix: bool):
    """Take q..source_order coefficients from a larger scalar-major motion RHS."""
    trailing = (rhs.shape[-1],) if rhs_is_matrix else ()
    head = rhs.shape[:-(2 if rhs_is_matrix else 1)]
    shaped = rhs.reshape(head + (self.robot_.dof, target_order) + trailing)
    return shaped[..., :source_order, :].reshape(head + (self.robot_.dof * source_order,) + trailing) if rhs_is_matrix \
      else shaped[..., :source_order].reshape(head + (self.robot_.dof * source_order,))

  def _embed_motion_order_rhs(self, rhs, source_order: int, target_order: int, rhs_is_matrix: bool):
    """Zero-pad a q..source_order motion result into a larger motion order."""
    trailing = (rhs.shape[-1],) if rhs_is_matrix else ()
    head = rhs.shape[:-(2 if rhs_is_matrix else 1)]
    shaped = rhs.reshape(head + (self.robot_.dof, source_order) + trailing)
    out = np.zeros(head + (self.robot_.dof, target_order) + trailing, dtype=rhs.dtype)
    if rhs_is_matrix:
      out[..., :source_order, :] = shaped
    else:
      out[..., :source_order] = shaped
    return out.reshape(head + (self.robot_.dof * target_order,) + trailing)

  def _embed_motion_order_jacobian(self, jacobian, source_order: int, target_order: int):
    head = jacobian.shape[:-2]
    rows = jacobian.shape[-2]
    shaped = jacobian.reshape(head + (rows, self.robot_.dof, source_order))
    out = np.zeros(head + (rows, self.robot_.dof, target_order), dtype=jacobian.dtype)
    out[..., :source_order] = shaped
    return out.reshape(head + (rows, self.robot_.dof * target_order))

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
