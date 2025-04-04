#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki

import numpy as np
import matplotlib.pyplot as plt

from mathrobo import *

from .motion import *
from .state import *
from .robot_model import *
from .robot_io import *
from .forward import *
  
class Robot():

  def __init__(self, robot_, motions_, state_):
    self.robot_ = robot_
    self.motions_ = motions_
    self.state_ = state_

  @staticmethod
  def from_json_file(model_file_name):
    robot_ = io_from_json_file(model_file_name)
    motions_ = RobotMotions(robot_)
    state_ = RobotState(robot_)
    return Robot(robot_, motions_, state_)
  
  def print_structure(self):
    io_print_structure(self.robot_)

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
    
  def set_target_from_file(self, target_file):
    self.target_ = io_from_target_json(target_file)
    
  def print_targets(self):
    io_print_targets(self.target_)
  
  def link_jacobian(self, link_name_list):
    return f_link_jacobian(self.robot_, self.state_, link_name_list)
  
  def link_jacobian_target(self):
    return f_link_jacobian(self.robot_, self.state_, self.target_.target_names)

  def __set_equall_aspect_3d(self, ax, data, margin):
    margin = 0.1
    ax_min = np.zeros(3)
    ax_max = np.zeros(3)
    box_length = np.zeros(3)
    for i in range(3):
      ax_min[i] = min(data[:,i])-margin
      ax_max[i] = max(data[:,i])+margin
      box_length[i] = ax_max[i] - ax_min[i]
      
    box_length_max = max((box_length[0], box_length[1], box_length[2]))
    box_ratio = box_length_max / box_length    

    ax.set_box_aspect((box_length_max,box_length_max,box_length_max))
    ax.set_xlim3d(ax_min[0]*box_ratio[0], ax_max[0]*box_ratio[0])
    ax.set_ylim3d(ax_min[1]*box_ratio[1], ax_max[1]*box_ratio[1])
    ax.set_zlim3d(ax_min[2]*box_ratio[2], ax_max[2]*box_ratio[2])
    
  def show_robot(self, save = False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    pos = self.state_.all_link_pos(self.robot_)
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], c='r', marker='o')

    for joint in self.robot_.joints:
      c_id = joint.child_link_id
      p_id = joint.parent_link_id
      ax.plot(
        [pos[c_id,0], pos[p_id,0]], 
        [pos[c_id,1], pos[p_id,1]], 
        [pos[c_id,2], pos[p_id,2]], 'b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    self.__set_equall_aspect_3d(ax, pos, 0.1)

    plt.show()
    if save:  
      plt.savefig('simple_draw.png')