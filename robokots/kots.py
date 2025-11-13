#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki

import numpy as np
from typing import List, Dict, Any

from robokots.outward import outward_state

from .basic.motion import RobotMotions
from .basic.state_df import RobotState
from .basic.state import StateType, keys_order, keys_time_order, filter_keys_kinematics, filter_keys_momentum, filter_keys_force, filter_keys_torque
from .basic.state_dict import state_dict_to_links_pos, print_state_dict
from .basic.robot import RobotStruct
from .basic.target import TargetList
from .basic.robot_drow import show_robot, show_robot_traj, RobotColor, show_link_points

from .robot_io import *
from .misc import check_valid_str_list, check_valid_data_type_list, count_time_order, filter_cmtm_row_data_to_target

from .outward.outward import kinematics as outward_kinematics
from .outward.outward import dynamics_cmtm as outward_dynamics
from .outward.outward import link_diff_kinematics_numerical, calc_link_total_point_frame
from .outward.outward_state import outward_state
from .outward.outward_gradient import jacobian_numerical, link_jacobian, link_cmtm_jacobian, link_jacobian_numerical
from .outward.outward_total_gradient import link_momentum_jacobian, world_link_momentum_jacobian, world_joint_momentum_jacobian
from .outward.outward_total_gradient import link_force_jacobian, joint_momentum_jacobian, dynamics_jacobian_numerical

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

  @staticmethod
  def from_json_file(model_file_name : str, order=default_order, dim=default_dim, lib : str = "numpy") -> "Kots":
    robot = load_robot_json_file(model_file_name, lib=lib)

    return Kots(robot, order, dim, lib)

  @staticmethod
  def from_json_data(model_data : dict, order=default_order, dim=default_dim, lib : str = "numpy") -> "Kots":
    robot = load_robot_json(model_data, lib=lib)

    return Kots(robot, order, dim, lib)

  def print_structure(self):
    print_robot_structure(self.robot_)

  def print_state_dict(self):
    print_state_dict(self.state_dict_)
    
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
    return outward_state(self.robot_, self.state_dict_, state_type)

  def state_info_list(self, state_type_list : List[StateType]) -> List[np.ndarray]:
    return [outward_state(self.robot_, self.state_dict_, st) for st in state_type_list]
  
  def target_state_info(self, data_type : str, frame_name : str = None) -> np.ndarray:
    if self.target_ is None:
        raise ValueError("target is not set")
    
    owner_types = self.target_.target_owner_types
    owner_names = self.target_.target_owner_names

    if data_type is None:
        data_types = self.target_.target_data_types
    else:
        data_types = [data_type for _ in owner_names]

    if frame_name is None:
        frame_names = self.target_.target_frame_names
    else:
        frame_names = [frame_name for _ in owner_names]

    state_type_list = [StateType(o_type, o_name, d_type, f_name) for o_type, o_name, d_type, f_name in zip(owner_types, owner_names, data_types, frame_names)]

    return self.state_info_list(state_type_list)
  
  def target_info(self, data_type : List = None) -> Dict[str, Dict[str, Any]]:
      if self.target_ is None:
          raise ValueError("target is not set")

      owner_types = self.target_.target_owner_types
      owner_names = self.target_.target_owner_names
      if data_type is None:
        data_types_list = self.target_.target_data_types
      else:
        data_types_list = [data_type for _ in owner_names]
      frame_names = self.target_.target_frame_names

      if len(owner_names) != len(data_types_list):
          raise ValueError(f"length mismatch: target_owner_names={len(owner_names)} vs target_data_types={len(data_types_list)}")

      out: Dict[str, Dict[str, Any]] = {}
      for owner_type, owner_name, data_types, frame_name in zip(owner_types, owner_names, data_types_list, frame_names):
          out[owner_name] = {t: outward_state(self.robot_, self.state_dict_, StateType(owner_type, owner_name, t, frame_name)) for t in data_types}
      return out
  
  def target_info_vecs(self) -> np.ndarray:
      target_info = self.target_info()
      vecs = []
      for _, info in target_info.items():
          for _, vec in info.items():
              if vec.size > 0:
                  vecs.append(vec)
      return np.concatenate(vecs) if vecs else np.array([]) 
  
  def kinematics(self, order = None):
    if order is None:
      order = self.order_
    self.state_dict_ = outward_kinematics(self.robot_, self.motion(order), order)

  # ToDo: change function name
  def kinematics_point(self, s : float = 0.0):
    return calc_link_total_point_frame(self.robot_, self.motions_, self.state_dict_, s)
  
  def dynamics(self, order = None):
    if order is None:
      order = self.order_
    self.state_dict_ = outward_dynamics(self.robot_, self.motion(order), order-2)

  def set_state_df(self):
    self.state_.import_state(self.state_dict_)
    
  def set_target_from_file(self, target_file : str):
    if not target_file:
      raise ValueError("target_file is empty")
    if not isinstance(target_file, str):
      raise TypeError("target_file must be a string")
    self.target_ = load_target_json_file(target_file)
    type_order = [keys_time_order[item] for sublist in self.target_.target_data_types for item in sublist]
    max_order = max(type_order)
    self.set_order(max_order)

  def print_targets(self):
    print_target_list(self.target_)

  def link_diff_kinematics_numerical(self, link_name_list : list[str], data_type = "vel", order = None, eps = 1e-8, update_method = "poly", update_direction = None):
    if order is None:
      order = keys_order(data_type)
      
    motion = np.zeros(self.robot_.dof * order)
    for joint in self.robot_.joints:
      m = self.motions_.joint_motions(joint.dof, joint.dof_index, order)
      motion[joint.dof_index*order:joint.dof_index*order+joint.dof*order] = m.flatten()

    return link_diff_kinematics_numerical(self.robot_, motion, link_name_list, data_type, order, eps, update_method, update_direction)

  def __jacobian(self, state_type : StateType, numerical : bool = False):
    max_order = 0

    data_type_list = check_valid_data_type_list(state_type.data_type)
    max_order = count_time_order(data_type_list)

    if state_type.owner_name not in self.robot_.link_names and state_type.owner_name not in self.robot_.joint_names:
      raise ValueError(f"Invalid name: {state_type.owner_name}. Must be a link or joint name.")
      
    data_type_list_kinematics = filter_keys_kinematics(data_type_list) 
    data_type_list_momentum = filter_keys_momentum(data_type_list)
    data_type_list_force = filter_keys_force(data_type_list)
    data_type_list_torque = filter_keys_torque(data_type_list)

    total_jacobian_kinematics = np.zeros((0, self.robot_.dof*max_order))
    total_jacobian_momentum = np.zeros((0, self.robot_.dof*max_order))
    total_jacobian_force = np.zeros((0, self.robot_.dof*max_order))

    if numerical:
      if state_type.owner_type == "link":
        total_jacobian_kinematics = link_jacobian_numerical(self.robot_, self.motions_, [state_type.owner_name], "cmtm", order_=max_order)
        if any(data_type_list_momentum):
          total_jacobian_momentum = dynamics_jacobian_numerical(self.robot_, self.motions_, [state_type.owner_name], "momentum", "link", state_type.frame_name, output_order_=max_order-1)
        if any(data_type_list_force):
          total_jacobian_force = dynamics_jacobian_numerical(self.robot_, self.motions_, [state_type.owner_name], "force", "link", state_type.frame_name, output_order_=max_order-2)
      elif state_type.owner_type == "joint":
        if any(data_type_list_momentum):
          total_jacobian_momentum = dynamics_jacobian_numerical(self.robot_, self.motions_, [state_type.owner_name], "momentum", "joint", state_type.frame_name, output_order_=max_order-1)
        if any(data_type_list_force):
          total_jacobian_force = jacobian_numerical(self.robot_, self.motions_, state_type)
    else:
      if state_type.owner_type == "link":
        total_jacobian_kinematics = link_cmtm_jacobian(self.robot_, self.motions_, self.state_dict_, [state_type.owner_name], max_order)
        if any(data_type_list_momentum):
          if state_type.frame_name is None:
            total_jacobian_momentum = link_momentum_jacobian(self.robot_, self.state_dict_, [state_type.owner_name], momentum_order=max_order-1)
          else:
            total_jacobian_momentum = world_link_momentum_jacobian(self.robot_, self.state_dict_, [state_type.owner_name], momentum_order=max_order-1)
        if any(data_type_list_force):
          total_jacobian_force = link_force_jacobian(self.robot_, self.state_dict_, [state_type.owner_name], force_order=max_order-2)
      elif state_type.owner_type == "joint":
        if any(data_type_list_momentum):
          if state_type.frame_name is None:
            total_jacobian_momentum = joint_momentum_jacobian(self.robot_, self.state_dict_, [state_type.owner_name], momentum_order=max_order-1)
          else:
            total_jacobian_momentum = world_joint_momentum_jacobian(self.robot_, self.state_dict_, [state_type.owner_name], momentum_order=max_order-1)
        if any(data_type_list_force):
          total_jacobian_force = joint_force_jacobian(self.robot_, self.state_dict_, [state_type.owner_name], force_order=max_order-2)

    jacobian_kinematics = filter_cmtm_row_data_to_target(total_jacobian_kinematics, state_type.owner_name, data_type_list_kinematics, dim=self.dim_)
    jacobian_momentum = filter_cmtm_row_data_to_target(total_jacobian_momentum, state_type.owner_name, data_type_list_momentum, dim=self.dim_)
    jacobian_force = filter_cmtm_row_data_to_target(total_jacobian_force, state_type.owner_name, data_type_list_force, dim=self.dim_)

    jacobian = np.vstack((jacobian_kinematics, jacobian_momentum, jacobian_force))

    return jacobian

  def jacobian(self, state_type : StateType):
    return self.__jacobian(state_type, numerical=False)

  def jacobian_numerical(self, state_type : StateType):
    return self.__jacobian(state_type, numerical=True)

  def __check_target(self, data_type_list : List[str] = None, frame_name_list : List[str] = None):
    if not self.target_:
      raise ValueError("target_ is not set")
    
    owner_type_list = self.target_.target_owner_types
    owner_name_list = self.target_.target_owner_names
    if data_type_list is None:
      data_type_list = self.target_.target_data_types
    if frame_name_list is None:
      frame_name_list = self.target_.target_frame_names
    return owner_type_list, owner_name_list, data_type_list, frame_name_list

  def jacobian_target(self, data_type_list : List[str] = None, frame_name_list : List[str] = None):
    owner_type_list, owner_name_list, data_type_list, frame_name_list = self.__check_target(data_type_list, frame_name_list)
    return self.jacobian(StateType(owner_type_list, owner_name_list, data_type_list, frame_name_list))

  def jacobian_target_numerical(self, data_type_list : List[str] = None, frame_name_list : List[str] = None):
    owner_type_list, owner_name_list, data_type_list, frame_name_list = self.__check_target(data_type_list, frame_name_list)
    return self.jacobian_numerical(StateType(owner_type_list, owner_name_list, data_type_list, frame_name_list))

  def jacobian_cmtm(self, name_list : List[str], order = None):
    if order is None:
      order = self.order_
    if order < 1:
      raise ValueError("order must be greater than 0")
    if order > self.order_:
      raise ValueError(f"order must be less than or equal to {self.order_}")

    if not name_list:
      raise ValueError("name_list is empty")
    if not all(name in self.robot_.link_names or name in self.robot_.joint_names for name in name_list):
      raise ValueError("name_list contains invalid names")

    return link_cmtm_jacobian(self.robot_, self.motions_, self.state_dict_, name_list, order)

  def show_robot(self, save = False):
    conectivity = np.zeros((self.robot_.joint_num, 2), dtype='int64')
    for i in range(self.robot_.joint_num):
      joint = self.robot_.joints[i]
      conectivity[i, 0] = joint.child_link_id
      conectivity[i, 1] = joint.parent_link_id

    show_robot(conectivity, state_dict_to_links_pos(self.state_dict_, self.robot_.link_names), save)

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
    show_link_points(state_dict_to_links_pos(self.state_dict_, self.target_.target_owner_names), plt, dimension)

  def target_link_pos_traj(self):
    if not self.target_:
      raise ValueError("target_ is not set")
    return self.state_.extract_links_info_traj("pos", self.target_.target_owner_names)

  def show_points(self, points, ax = None, dimension=3):
    show_link_points(points, ax, dimension)