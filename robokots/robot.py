#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki

import numpy as np

from .motion import *
from .state import *
from .robot_model import *
from .robot_io import *
from .robot_drow import *
from .forward import *
  
class Robot():

  def __init__(self, robot_, motions_, state_, level_):
    self.robot_ = robot_
    self.motions_ = motions_
    self.state_ = state_
    self.level_ = level_

  @staticmethod
  def from_json_file(model_file_name, level_="kinematics"):
    robot_ = io_from_json_file(model_file_name)
    motions_ = RobotMotions(robot_)
    if level_ == "kinematics":
      state_ = RobotState(robot_)
    elif level_ == "dynamics":
      state_ = RobotState(robot_, l_aliases=["pos", "rot", "vel", "acc","link_force"],j_aliases=["joint_torque", "joint_force"])
    return Robot(robot_, motions_, state_, level_)
  
  def print_structure(self):
    io_print_structure(self.robot_)
    
  def dof(self):
    return self.robot_.dof
  
  def link_list(self, name_list):
    return self.robot_.link_list(name_list)
  
  def joint_list(self, name_list):
    return self.robot_.joint_list(name_list)

  def motions(self):
    return self.motions_.motions  

  def set_motion_aliases(self, aliases):
    self.motions_.set_aliases(aliases)
    
  def import_motions(self, vecs):
    self.motions_.set_motion(vecs)
    
  def motion(self, name):
    return self.motions_.gen_values(name)
  
  def state_df(self):
    return self.state_.df()
  
  def state_link_info(self, type, name):
    return self.state_.extract_info('link', type, name)

  def state_link_info_list(self, type, name_list):
    return [self.state_.extract_info('link', type, name) for name in name_list]
  
  def state_target_link_info(self, type):
    return self.state_link_info_list(type, self.target_.target_names)

  def kinematics(self):
    self.state_.import_state(f_kinematics(self.robot_, self.motions_))
  
  def dynamics(self):
    self.state_.import_state(f_dynamics(self.robot_, self.motions_))
    
  def set_target_from_file(self, target_file):
    self.target_ = io_from_target_json(target_file)
    
  def print_targets(self):
    io_print_targets(self.target_)
  
  def link_jacobian(self, link_name_list):
    return f_link_jacobian(self.robot_, self.state_, link_name_list)
  
  def link_jacobian_target(self):
    return f_link_jacobian(self.robot_, self.state_, self.target_.target_names)
      
  def show_robot(self, save = False):
    conectivity = np.zeros((self.robot_.joint_num, 2), dtype='int64')
    for i in range(self.robot_.joint_num):
      joint = self.robot_.joints[i]
      conectivity[i, 0] = joint.child_link_id
      conectivity[i, 1] = joint.parent_link_id

    d_show_robot(conectivity, self.state_.all_link_pos(self.robot_), save)