#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki

import numpy as np
import matplotlib.pyplot as plt

from mathrobo import *

from .motion import *
from .state import *
from .robot_model import *
from .kinematics import *
  
class Robot():
  state : RobotState 

  def __init__(self, robot_, motions_, state_):
    self.robot = robot_
    self.motions = motions_
    self.state = state_

  @staticmethod
  def from_json_file(model_file_name):
    robot = RobotStruct.from_json_file(model_file_name)
    motinos = RobotMotions(robot)
    state = RobotState(robot)
    return Robot(robot, motinos, state)
    
  def import_motions(self, vecs):
    self.motions.set_motion(vecs)

  def kinematics(self):
    state_data = {}
    
    world_name = self.robot.links[self.robot.joints[0].parent_link].name
    state_data.update([(world_name + "_pos" , [0.,0.,0.])])
    state_data.update([(world_name + "_rot" , [1.,0.,0.,0.,1.,0.,0.,0.,1.])])
    state_data.update([(world_name + "_vel" , [0.,0.,0.,0.,0.,0.])])
    state_data.update([(world_name + "_acc" , [0.,0.,0.,0.,0.,0.])])
    
    for joint in self.robot.joints:    
      child = self.robot.links[joint.child_link]
      
      frame = Kinematics.kinematics(joint, self.robot.links, self.motions, state_data)  
      veloc = Kinematics.vel_kinematics(joint, self.robot.links, self.motions, state_data)  
      accel = Kinematics.acc_kinematics(joint, self.robot.links, self.motions, state_data) 
      
      a = SE3().set_adj_mat(frame)

      pos = a.pos()
      rot_vec = a.rot().ravel()
      
      state_data.update([(child.name + "_pos" , pos.tolist())])
      state_data.update([(child.name + "_rot" , rot_vec.tolist())])
      state_data.update([(child.name + "_vel" , veloc.tolist())])
      state_data.update([(child.name + "_acc" , accel.tolist())])
      
    self.state.import_state(state_data)
    
  def show_robot(self, save = False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    pos = self.state.all_link_pos(self.robot)
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], c='r', marker='o')

    for joint in self.robot.joints:
      c_id = joint.child_link
      p_id = joint.parent_link
      ax.plot(
        [pos[c_id,0], pos[p_id,0]], 
        [pos[c_id,1], pos[p_id,1]], 
        [pos[c_id,2], pos[p_id,2]], 'b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    if save:  
      plt.savefig('simple_draw.png')