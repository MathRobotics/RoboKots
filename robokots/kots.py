#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki

import numpy as np

from .basic.motion import RobotMotions
from .basic.state import RobotState
from .basic.state_dict import extract_dict_link_info, extract_dict_joint_info, dict_to_links_pos
from .basic.robot import RobotStruct, JointStruct, LinkStruct
from .basic.target import TargetList
from .basic.robot_drow import *

from .robot_io import *

from .outward.outward import kinematics as outward_kinematics
from .outward.outward import dynamics_cmtm, link_diff_kinematics_numerical, calc_link_total_point_frame
from .outward.outward_gradient import link_jacobian, link_cmtm_jacobian, link_jacobian_numerical 
  
class Kots():
  robot_ : RobotStruct
  motions_ : RobotMotions
  state_dict_ : dict
  state_ : RobotState
  target_ : TargetList
  order_ : int
  dim_ : int
  
  def __init__(self, robot : RobotStruct, order : int, dim : int):

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

    self.robot_ = robot
    self.motions_ = RobotMotions(robot.dof, m_aliases)
    self.state_ = RobotState(robot.link_names, robot.joint_names, l_aliases, j_aliases)
    self.state_dict_ = {}
    self.order_ = order
    self.dim_ = dim

  @staticmethod
  def from_json_file(model_file_name : str, order=3, dim=3) -> "Kots":
    robot = load_robot_json_file(model_file_name)

    return Kots(robot, order, dim)
  
  @staticmethod
  def from_json_data(model_data : dict, order=3, dim=3) -> "Kots":
    robot = load_robot_json(model_data)

    return Kots(robot, order, dim)

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
    
  def motion(self, name : str):
    return self.motions_.gen_values(name)

  def joint_motions(self, joint : JointStruct):
    return self.motions_.joint_motions(joint)
  
  def link_motions(self, link : LinkStruct):
    return self.motions_.link_motions(link)
  
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

  def kinematics(self):
    self.state_dict_ = outward_kinematics(self.robot_, self.motions_, self.order_)

  def kinematics_point(self, s : float = 0.0):
    return calc_link_total_point_frame(self.robot_, self.motions_, self.state_dict_, s)
  
  def dynamics(self):
    self.state_dict_ = dynamics_cmtm(self.robot_, self.motions_, self.order_-2)

  def set_state_df(self):
    self.state_.import_state(self.state_dict_)
    
  def set_target_from_file(self, target_file : str):
    if not target_file:
      raise ValueError("target_file is empty")
    if not isinstance(target_file, str):
      raise TypeError("target_file must be a string")
    self.target_ = load_target_json_file(target_file)

  def print_targets(self):
    print_target_list(self.target_)

  def link_jacobian(self, link_name_list : list[str], order = 3):
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

  def link_diff_kinematics_numerical(self, link_name_list : list[str], data_type = "vel", order = None, eps = 1e-8, update_method = "poly", update_direction = None):
    return link_diff_kinematics_numerical(self.robot_, self.motions_, link_name_list, data_type, order, eps, update_method, update_direction)

  def link_jacobian_numerical(self, link_name_list : list[str], data_type = "vel", order = None):
    return link_jacobian_numerical(self.robot_, self.motions_, link_name_list, data_type, order)
  
  def link_jacobian_target_numerical(self, data_type = "vel", order = None):
    if not self.target_:
      raise ValueError("target_ is not set")
    return self.link_jacobian_numerical(self.target_.target_names, data_type, order)
  
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

  