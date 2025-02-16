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
from .kinematics import *
  
class Robot():

  def __init__(self, robot_, motions_, state_):
    self.robot_ = robot_
    self.motions_ = motions_
    self.state_ = state_

  @staticmethod
  def from_json_file(model_file_name):
    robot_ = RobotIO.from_json_file(model_file_name)
    motions_ = RobotMotions(robot_)
    state_ = RobotState(robot_)
    return Robot(robot_, motions_, state_)
  
  def print_structure(self):
    RobotIO.print_structure(self.robot_)
  
  def set_motion_aliases(self, aliases):
    self.motions_.set_aliases(aliases)
    
  def import_motions(self, vecs):
    self.motions_.set_motion(vecs)
    
  def motions(self):
    return self.motions_.motions
  
  def state_df(self):
    return self.state_.df()

  def kinematics(self):
    state_data = {}
    
    world_name = self.robot_.links[self.robot_.joints[0].parent_link].name
    state_data.update([(world_name + "_pos" , [0.,0.,0.])])
    state_data.update([(world_name + "_rot" , [1.,0.,0.,0.,1.,0.,0.,0.,1.])])
    state_data.update([(world_name + "_vel" , [0.,0.,0.,0.,0.,0.])])
    state_data.update([(world_name + "_acc" , [0.,0.,0.,0.,0.,0.])])
    
    for joint in self.robot_.joints:    
      parent = self.robot_.links[joint.parent_link]
      child = self.robot_.links[joint.child_link]
      
      joint_coord = self.motions_.joint_coord(joint)
      joint_veloc = self.motions_.joint_veloc(joint)
      joint_accel = self.motions_.joint_accel(joint)
      
      rot = np.array(state_data[parent.name + "_rot"]).reshape((3,3))
      p_link_frame = SE3(rot, state_data[parent.name + "_pos"])
      p_link_vel = state_data[parent.name + "_vel"]  
      p_link_acc = state_data[parent.name + "_acc"]  

      frame = kinematics(joint, p_link_frame, joint_coord)  
      veloc = vel_kinematics(joint, p_link_vel, joint_coord, joint_veloc)  
      accel = acc_kinematics(joint, p_link_vel, p_link_acc, joint_coord, joint_veloc, joint_accel)       
      
      pos = frame.pos()
      rot_vec = frame.rot().ravel()

      state_data.update([(child.name + "_pos" , pos.tolist())])
      state_data.update([(child.name + "_rot" , rot_vec.tolist())])
      state_data.update([(child.name + "_vel" , veloc.tolist())])
      state_data.update([(child.name + "_acc" , accel.tolist())])
      
    self.state_.import_state(state_data)
    
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
      c_id = joint.child_link
      p_id = joint.parent_link
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