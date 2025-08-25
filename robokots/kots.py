#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki

import numpy as np
from typing import List, Dict, Any

from .basic.motion import RobotMotions
from .basic.state_df import RobotState
from .basic.state import keys_order, data_type_dof, dim_to_dof
from .basic.state_dict import extract_dict_link_info, extract_dict_joint_info, dict_to_links_pos
from .basic.robot import RobotStruct
from .basic.target import TargetList
from .basic.robot_drow import *

from .robot_io import *

from .outward.outward import kinematics as outward_kinematics
from .outward.outward import dynamics_cmtm as outward_dynamics
from .outward.outward import link_diff_kinematics_numerical, calc_link_total_point_frame
from .outward.outward_gradient import link_jacobian, link_cmtm_jacobian, link_jacobian_numerical 
  
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
    
  def dof(self):
    return self.robot_.dof

  def order(self):
    return self.order_
  
  def link_name_list(self):
    return self.robot_.link_names
  
  def link_list(self, name_list : list[str]):
    return self.robot_.link_list(name_list)
  
  def joint_name_list(self):
    return self.robot_.joint_names
  
  def joint_list(self, name_list : list[str]):
    return self.robot_.joint_list(name_list)

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
      motion_diff[joint.dof_index*order:joint.dof_index*order+joint.dof*order] = m.flatten()[1:order+1]
    for link in self.robot_.links:
      m = self.motions_.link_motions(link.dof, link.dof_index, self.order_)
      m = np.append(m, last_diff[link.dof_index:link.dof_index+link.dof])
      motion_diff[link.dof_index*order:link.dof_index*order+link.dof*order] = m.flatten()[1:order+1]
    return motion_diff

  def state_df(self):
    return self.state_.df()
  
  def state_link_info(self, data_type : str, link_name : str):
    return extract_dict_link_info(self.state_dict_, data_type, link_name)

  def state_link_info_list(self, data_type : str, name_list : list[str]):
    return [extract_dict_link_info(self.state_dict_, data_type, name) for name in name_list]
  
  def state_target_link_info(self, data_type : str):
    return self.state_link_info_list(data_type, self.target_.target_names)
  
  def state_joint_info(self, data_type : str, joint_name : str):
    return extract_dict_joint_info(self.state_dict_, data_type, joint_name)

  def state_joint_info_list(self, data_type : str, name_list : list[str]):
    return [extract_dict_joint_info(self.state_dict_, data_type, name) for name in name_list]
  
  def state_joint_vecs(self, data_type : str):
    values = ()
    for name in self.robot_.joint_names:
      values += (extract_dict_joint_info(self.state_dict_, data_type, name),)
    vecs = [v for v in values if v.size]
    return np.array(vecs)
  
  def target_info(self) -> Dict[str, Dict[str, Any]]:
      if self.target_ is None:
          raise ValueError("target is not set")

      names = self.target_.target_names
      types_list = self.target_.target_types
      if len(names) != len(types_list):
          raise ValueError(f"length mismatch: target_names={len(names)} vs target_types={len(types_list)}")

      out: Dict[str, Dict[str, Any]] = {}
      for name, t_list in zip(names, types_list):
          out[name] = {t: extract_dict_link_info(self.state_dict_, t, name) for t in t_list}
      return out
  
  def target_info_vecs(self) -> np.ndarray:
      target_info = self.target_info()
      vecs = []
      for name, info in target_info.items():
          for data_type, vec in info.items():
              if vec.size > 0:
                  vecs.append(vec)
      return np.concatenate(vecs) if vecs else np.array([]) 
  
  def kinematics(self, order = None):
    if order is None:
      order = self.order_
    self.state_dict_ = outward_kinematics(self.robot_, self.motion(order), order)

  def kinematics_point(self, s : float = 0.0):
    return calc_link_total_point_frame(self.robot_, self.motions_, self.state_dict_, s)
  
  def dynamics(self):
    self.state_dict_ = outward_dynamics(self.robot_, self.motion(3), self.order_-2)

  def set_state_df(self):
    self.state_.import_state(self.state_dict_)
    
  def set_target_from_file(self, target_file : str):
    if not target_file:
      raise ValueError("target_file is empty")
    if not isinstance(target_file, str):
      raise TypeError("target_file must be a string")
    self.target_ = load_target_json_file(target_file)
    type_order = [keys_order[item] for sublist in self.target_.target_types for item in sublist]
    max_order = max(type_order)
    self.set_order(max_order)

  def print_targets(self):
    print_target_list(self.target_)

  def link_jacobian(self, link_name_list : list[str], order = default_order):
    if order < 1:
      raise ValueError("order must be greater than 0")
    if order > self.order_:
      raise ValueError(f"order must be less than or equal to {self.order_}")

    if not link_name_list:
      raise ValueError("link_name_list is empty")
    if not all(link_name in self.robot_.link_names for link_name in link_name_list):
      raise ValueError("link_name_list contains invalid link names")

    if order == 1:
        return link_jacobian(self.robot_, self.motions_, self.state_dict_, link_name_list)
    else:
        return link_cmtm_jacobian(self.robot_, self.motions_, self.state_dict_, link_name_list, order)
  
  def link_jacobian_target(self, order = 3):
    if not self.target_:
      raise ValueError("target_ is not set")
    return self.link_jacobian(self.target_.target_names, order)

  def jacobian(self, name_list : List[str], data_type_list : List[str]):
    if not name_list:
      raise ValueError("name_list is empty")
    if not data_type_list:
      raise ValueError("data_type_list is empty")

    if type(name_list) is str:
      name_list = [name_list]
    if type(data_type_list) is str:
      data_type_list = [[data_type_list]]

    if len(name_list) != len(data_type_list):
      raise ValueError("name_list and data_type_list must have the same length")

    max_order = 0
    total_target_dof = 0
    for name, link_dt_list in zip(name_list, data_type_list):
      if name not in self.robot_.link_names and name not in self.robot_.joint_names:
        raise ValueError(f"Invalid name: {name}. Must be a link or joint name.")
      for data_type in link_dt_list:
        if data_type not in keys_order:
          raise ValueError(f"Invalid data_type: {data_type}. Must be one of {list(keys_order.keys())}.")
        order = keys_order[data_type]
        if order > max_order:
          max_order = order
        total_target_dof += data_type_dof(data_type, dim=self.dim_)
    
    total_jacobian = link_cmtm_jacobian(self.robot_, self.motions_, self.state_dict_, name_list, max_order)
    jacobian = np.zeros((total_target_dof, self.robot_.dof * max_order))

    index = 0
    for i, name in enumerate(name_list):
      index = i * max_order * dim_to_dof(self.dim_)
      for data_type in data_type_list[i]:
        order = keys_order[data_type]
        dof = data_type_dof(data_type, order, dim=self.dim_)
        jacobian[index:index+dof, :] = total_jacobian[dim_to_dof(self.dim_)*(order-1):dim_to_dof(self.dim_)*(order-1)+dof, :]
        index += dof

    return jacobian
  
  def jacobian_target(self, data_type_list : List[str] = None):
    if not self.target_:
      raise ValueError("target_ is not set")
    
    name_list = self.target_.target_names
    if data_type_list is None:
      data_type_list = self.target_.target_types

    return self.jacobian(name_list, data_type_list)
  
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

  def link_diff_kinematics_numerical(self, link_name_list : list[str], data_type = "vel", order = None, eps = 1e-8, update_method = "poly", update_direction = None):
    if order is None:
      order = keys_order(data_type)
      
    motion = np.zeros(self.robot_.dof * order)
    for joint in self.robot_.joints:
      m = self.motions_.joint_motions(joint.dof, joint.dof_index, order)
      motion[joint.dof_index*order:joint.dof_index*order+joint.dof*order] = m.flatten()

    return link_diff_kinematics_numerical(self.robot_, motion, link_name_list, data_type, order, eps, update_method, update_direction)
  
  def link_jacobian_numerical(self, link_name_list : list[str], data_type = "vel", order = None):
    return link_jacobian_numerical(self.robot_, self.motions_, link_name_list, data_type, order)
  
  def link_jacobian_target_numerical(self, data_type = "vel", order = None):
    if not self.target_:
      raise ValueError("target_ is not set")
    return self.link_jacobian_numerical(self.target_.target_names, data_type, order)
  
  def jacobian_numerical(self, name_list : List[str], data_type_list : List[str]):
    if not name_list:
      raise ValueError("name_list is empty")
    if not data_type_list:
      raise ValueError("data_type_list is empty")

    if type(name_list) is str:
      name_list = [name_list]
    if type(data_type_list) is str:
      data_type_list = [[data_type_list]]

    if len(name_list) != len(data_type_list):
      raise ValueError("name_list and data_type_list must have the same length")

    max_order = 0
    total_target_dof = 0
    for name, link_dt_list in zip(name_list, data_type_list):
      if name not in self.robot_.link_names and name not in self.robot_.joint_names:
        raise ValueError(f"Invalid name: {name}. Must be a link or joint name.")
      for data_type in link_dt_list:
        if data_type not in keys_order:
          raise ValueError(f"Invalid data_type: {data_type}. Must be one of {list(keys_order.keys())}.")
        order = keys_order[data_type]
        if order > max_order:
          max_order = order
        total_target_dof += data_type_dof(data_type, dim=self.dim_)
    
    total_jacobian = link_jacobian_numerical(self.robot_, self.motions_, name_list, "cmtm", max_order)
    jacobian = np.zeros((total_target_dof, self.robot_.dof * max_order))

    index = 0
    for i, name in enumerate(name_list):
      index = i * max_order * dim_to_dof(self.dim_)
      for data_type in data_type_list[i]:
        order = keys_order[data_type]
        dof = data_type_dof(data_type, order, dim=self.dim_)
        jacobian[index:index+dof, :] = total_jacobian[dim_to_dof(self.dim_)*(order-1):dim_to_dof(self.dim_)*(order-1)+dof, :]
        index += dof

    return jacobian
  
  def jacobian_target_numerical(self, data_type_list : List[str] = None):
    if not self.target_:
      raise ValueError("target_ is not set")
    
    name_list = self.target_.target_names
    if data_type_list is None:
      data_type_list = self.target_.target_types

    return self.jacobian_numerical(name_list, data_type_list)
  
  def show_robot(self, save = False):
    conectivity = np.zeros((self.robot_.joint_num, 2), dtype='int64')
    for i in range(self.robot_.joint_num):
      joint = self.robot_.joints[i]
      conectivity[i, 0] = joint.child_link_id
      conectivity[i, 1] = joint.parent_link_id

    d_show_robot(conectivity, dict_to_links_pos(self.state_dict_, self.robot_.link_names), save)

  def show_link_points(self):
    print(dict_to_links_pos(self.state_dict_, self.robot_.link_names))
    d_show_link_points(dict_to_links_pos(self.state_dict_, self.robot_.link_names))

  def show_target_link_points(self):
    if not self.target_:
      raise ValueError("target_ is not set")
    d_show_link_points(dict_to_links_pos(self.state_dict_, self.target_.target_names))

  def target_link_pos_traj(self):
    if not self.target_:
      raise ValueError("target_ is not set")
    return self.state_.extract_links_info_traj("pos", self.target_.target_names)

  def show_points(self, points):
    d_show_link_points(points)

  