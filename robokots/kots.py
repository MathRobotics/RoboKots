#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki

import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from .core.motion import RobotMotions
from .core.state_table import RobotState
from .core.state import StateType
from .core.state_cache import StateCache
from .core.state_dict import state_dict_to_links_pos, print_state_dict
from .core.robot import RobotStruct
from .core.target import TargetList, RobotNames
from .core.viz import show_robot, show_robot_traj, RobotColor, show_link_points

from .robot_io import *
from .outward import (
    build_kinematics_state,
    build_dynamics_cmtm_state,
    get_value,
    link_diff_kinematics_numerical,
    diff_outward_numerical,
    jacobian_numerical,
    outward_jacobian,
    calc_link_total_point_frame,
    update_outward_state,
)

default_order = 3 
default_dim = 3
class Kots():
  robot_ : RobotStruct
  motions_ : RobotMotions
  state_dict_ : dict
  state_ : RobotState
  target_ : TargetList
  order_ : int
  dim_ : int
  lib_ : str
  state_cache_config_ : Optional[Tuple[bool, int]]

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
        
    l_aliases.append("link_force")
    j_aliases.append("joint_torque")
    j_aliases.append("joint_force")
    
    for i in range(order-3):
      l_aliases.append("link_force_diff"+str(i+1))
      j_aliases.append("joint_torque_diff"+str(i+1))
      j_aliases.append("joint_force_diff"+str(i+1))

    return m_aliases, l_aliases, j_aliases
  
  def __init__(self, robot : RobotStruct, order : int, dim : int, lib : str = "numpy"):

    m_aliases, l_aliases, j_aliases = self.order_to_aliases(order)

    self.robot_ = robot
    self.motions_ = RobotMotions(robot.dof, m_aliases)
    self.state_ = RobotState(robot.link_names, robot.joint_names, l_aliases, j_aliases)
    self.state_dict_ = {}
    self.state_cache_ = None
    self.state_cache_config_ = None
    self.order_ = order
    self.dim_ = dim
    self.lib_ = lib

  def set_order(self, order: int):
    if order < 1:
      raise ValueError("order must be greater than 0")
    m_aliases, l_aliases, j_aliases = self.order_to_aliases(order)
    self.order_ = order
    self.motions_ = RobotMotions(self.robot_.dof, m_aliases)
    self.state_ = RobotState(self.robot_.link_names, self.robot_.joint_names, l_aliases, j_aliases)
    self.state_cache_ = None
    self.state_cache_config_ = None

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
    
  def import_motions(self, vecs : np.ndarray):
    self.motions_.set_motion(vecs)
    self.motions_.increment_revision()

  def motion(self, order : int = None):
    if order is None:
      order = self.order_
    motion = np.zeros(self.robot_.dof * order)
    for joint in self.robot_.joints:
      m = self.motions_.joint_motions(joint.dof, joint.dof_index, order)
      motion[joint.dof_index*order:joint.dof_index*order+joint.dof*order] = m.flatten()
    for link in self.robot_.links:
      m = self.motions_.link_motions(link.dof, link.dof_index, order)
      motion[link.dof_index*order:link.dof_index*order+link.dof*order] = m.flatten()
    return motion
  
  def motion_diff(self, order : int = None, last_diff = None):
    if order is None:
      order = self.order_
    if last_diff is None:
      last_diff = np.zeros(self.robot_.dof)
    motion_diff = np.zeros(self.robot_.dof * order)
    for joint in self.robot_.joints:
      m = self.motions_.joint_motions(joint.dof, joint.dof_index, self.order_)
      m = np.append(m, last_diff[joint.dof_index:joint.dof_index+joint.dof])
      motion_diff[joint.dof_index*order:joint.dof_index*order+joint.dof*order] = m.flatten()[joint.dof:order+joint.dof]
    for link in self.robot_.links:
      m = self.motions_.link_motions(link.dof, link.dof_index, self.order_)
      m = np.append(m, last_diff[link.dof_index:link.dof_index+link.dof])
      motion_diff[link.dof_index*order:link.dof_index*order+link.dof*order] = m.flatten()[link.dof:order+link.dof]
    return motion_diff
  
  def motion_cm(self, order : int = None):
    if order is None:
      order = self.order_
    motion = np.zeros(self.robot_.dof * order)
    for joint in self.robot_.joints:
      m = self.motions_.joint_motions_cm(joint.dof, joint.dof_index, order)
      motion[joint.dof_index*order:joint.dof_index*order+joint.dof*order] = m.flatten()
    for link in self.robot_.links:
      m = self.motions_.link_motions_cm(link.dof, link.dof_index, order)
      motion[link.dof_index*order:link.dof_index*order+link.dof*order] = m.flatten()
    return motion
  
  def motion_diff_cm(self, order : int = None, last_diff = None):
    if order is None:
      order = self.order_
    if last_diff is None:
      last_diff = np.zeros(self.robot_.dof)
    motion_diff = np.zeros(self.robot_.dof * order)
    for joint in self.robot_.joints:
      m = self.motions_.joint_motions_cm(joint.dof, joint.dof_index, self.order_)
      m = np.append(m, last_diff[joint.dof_index:joint.dof_index+joint.dof])
      motion_diff[joint.dof_index*order:joint.dof_index*order+joint.dof*order] = m.flatten()[joint.dof:order+joint.dof]
    for link in self.robot_.links:
      m = self.motions_.link_motions_cm(link.dof, link.dof_index, self.order_)
      m = np.append(m, last_diff[link.dof_index:link.dof_index+link.dof])
      motion_diff[link.dof_index*order:link.dof_index*order+link.dof*order] = m.flatten()[link.dof:order+link.dof]
    return motion_diff

  def state_df(self):
    return self.state_.df()

  def state_info(self, state_type : StateType):
    return get_value(self.robot_, self.state_dict_, state_type)

  def state_info_list(self, state_type_list : List[StateType], list_output : bool = False) -> List[np.ndarray]:
    state_list = [get_value(self.robot_, self.state_dict_, st) for st in state_type_list]
    if list_output:
        return state_list
    else:
        return np.vstack(state_list)

  def target_state_info(self, list_output : bool = False) -> np.ndarray:
    if self.target_ is None:
      raise ValueError("target is not set")

    return self.state_info_list(self.target_._targets, list_output=list_output)
  
  def kinematics(self, order = None, backend : str = None):
    if order is None:
      order = self.order_
    self.state_dict_ = build_kinematics_state(self.robot_, self.motion(order), order, backend=backend)

  # ToDo: change function name
  def kinematics_point(self, s : float = 0.0):
    return calc_link_total_point_frame(self.robot_, self.motions_, self.state_dict_, s)
  
  def dynamics(self, order = None):
    if order is None:
      order = self.order_
    self.state_dict_ = build_dynamics_cmtm_state(self.robot_, self.motion(order), order-2)

  def update_state_dict(self, order : int = None, is_dynamics: bool = False, backend : str = None) -> dict:
    if order is None:
      order = self.order_

    kinematics_backend = None if is_dynamics else backend
    cache_config = (bool(is_dynamics), int(order), kinematics_backend)
    if self.state_cache_ is None or self.state_cache_config_ != cache_config:
      if not is_dynamics:
        self.state_cache_ = StateCache(
          build_state=lambda x_all, time=None, required=None: build_kinematics_state(self.robot_, x_all, order, backend=kinematics_backend)
        )
      else:
        self.state_cache_ = StateCache(
          build_state=lambda x_all, time=None, required=None: build_dynamics_cmtm_state(self.robot_, x_all, order-2)
        )
      self.state_cache_config_ = cache_config

    motion_revision = self.motions_.revision()
    if self.state_cache_.is_fresh(motion_revision):
      self.state_dict_ = self.state_cache_.state
      return self.state_dict_

    class _MotionPack:
      def __init__(self, x: np.ndarray, revision: int):
        self._x = np.asarray(x, dtype=float).reshape(-1)
        self.revision = int(revision)

      def get(self) -> np.ndarray:
        return self._x

    motion_pack = _MotionPack(self.motion(order), motion_revision)

    self.state_dict_ = update_outward_state(self.robot_, motion_pack, self.state_cache_, is_dynamics, order)
    return self.state_dict_

  def set_state_df(self):
    self.state_.import_state(self.state_dict_)
    
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
    
    motion = self.motion(order)

    return link_diff_kinematics_numerical(self.robot_, motion, link_name_list, data_type, order, eps, update_method, update_direction)
  
  def diff_outward_numerical(self, state_type : StateType, order : int = None, eps : float = 1e-8, update_method : str = "poly", update_direction = None):
    if order is None:
      order = self.order_

    motion = self.motion(order)
    
    return diff_outward_numerical(self.robot_, motion, state_type, order, eps, update_method, update_direction)

  def jacobian(self, state_type, numerical : bool = False, list_output : bool = False):
    if type(state_type) is list:
      state_type_list = state_type
    else:
      state_type_list = [state_type]
    
    if numerical:
      max_order = StateType.max_time_order(state_type_list)
      jacobs = [jacobian_numerical(self.robot_, self.motions_, st, max_order) for st in state_type_list]
      if list_output:
        return jacobs
      else:
        return np.vstack(jacobs)

    return outward_jacobian(self.robot_, self.state_dict_, state_type_list, dim = self.dim_, list_output = list_output)
  
  def jacobian_target(self, numerical : bool = False, list_output : bool = False):
    if self.target_ is None:
      raise ValueError("target is not set")
    
    return self.jacobian(self.target_._targets, numerical=numerical, list_output=list_output)
  
  def inverse_kinematics(self, target_type : List[StateType], target_value : List[np.ndarray],
                    q_init : np.ndarray, opt_func : None = None) -> np.ndarray:
    raise NotImplementedError(
      "inward module was removed. inverse_kinematics is no longer available in robokots."
    )

  def show_robot(self, save = False, ax = None, color : RobotColor = None):
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
      link_pos_traj = self.state_.extract_links_info_traj("pos", self.robot_.link_names)
    else:
      link_pos_traj = traj
    show_robot_traj(conectivity, link_pos_traj, save, ax, color)

  def show_link_points(self):
    show_link_points(state_dict_to_links_pos(self.state_dict_, self.robot_.link_names))

  def show_target_link_points(self, plt = None, dimension=3):
    if not self.target_:
      raise ValueError("target_ is not set")
    
    owner_link_names = []
    for t in self.target_._targets:
      if t._state_type.owner_type == "link":
        owner_link_names.append(t._state_type.owner_name)
    show_link_points(state_dict_to_links_pos(self.state_dict_, owner_link_names), plt, dimension)

  def target_link_pos_traj(self):
    if not self.target_:
      raise ValueError("target_ is not set")
    
    owner_link_names = []
    for t in self.target_._targets:
      if t._state_type.owner_type == "link":
        owner_link_names.append(t._state_type.owner_name)
    return self.state_.extract_links_info_traj("pos", owner_link_names)

  def show_points(self, points, ax = None, dimension=3):
    show_link_points(points, ax, dimension)
