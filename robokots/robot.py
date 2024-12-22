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
  state : RobotState 

  def __init__(self, robot_, motions_, state_):
    self.robot = robot_
    self.motions = motions_
    self.state = state_

  @staticmethod
  def from_json_file(model_file_name):
    robot = RobotIO.from_json_file(model_file_name)
    motions = RobotMotions(robot)
    state = RobotState(robot)
    return Robot(robot, motions, state)
    
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
      parent = self.robot.links[joint.parent_link]
      child = self.robot.links[joint.child_link]
      
      joint_coord = self.motions.joint_coord(joint)
      joint_veloc = self.motions.joint_veloc(joint)
      joint_accel = self.motions.joint_accel(joint)
      
      rot = np.array(state_data[parent.name + "_rot"]).reshape((3,3))
      p_link_frame = SE3(rot, state_data[parent.name + "_pos"]).adj_mat()
      p_link_vel = state_data[parent.name + "_vel"]  
      p_link_acc = state_data[parent.name + "_acc"]  

      frame = Kinematics.kinematics(joint, p_link_frame, joint_coord)  
      veloc = Kinematics.vel_kinematics(joint, p_link_vel, joint_coord, joint_veloc)  
      accel = Kinematics.acc_kinematics(joint, p_link_vel, p_link_acc, joint_coord, joint_veloc, joint_accel)       
      
      tmp = SE3.set_adj_mat(frame)
      pos = tmp.pos()
      rot_vec = tmp.rot().ravel()

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