#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki

import numpy as np

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
    self.robot_init()

  @staticmethod
  def from_json_file(model_file_name):
    robot = RobotStruct.from_json_file(model_file_name)
    motinos = RobotMotions(robot)
    state = RobotState(robot)
    return Robot(robot, motinos, state)
    
  def import_motions(self, vecs):
    self.motions.set_motion(vecs)

  def __kinematics_tree(self, joint, links, motions, state_data):
    parent = links[joint.parent_id]
    child = links[joint.child_id]
    frame = Kinematics.kinematics(child, joint, parent, motions, state_data)  
    veloc = Kinematics.vel_kinematics(child, joint, parent, motions, state_data)  
    accel = Kinematics.acc_kinematics(child, joint, parent, motions, state_data) 

    # for link_id in joint.connect_link:
    #   if LinkStruct.link_id(link) != link_id or link == None:
    #     l = self.links[link_id]
    #     frame = Kinematics.kinematics(l, joint, link, motions, state_data)  
    #     veloc = Kinematics.vel_kinematics(l, joint, link, motions, state_data)  
    #     accel = Kinematics.acc_kinematics(l, joint, link, motions, state_data) 
      
    #     a = SE3()
    #     a.set_adj_mat(frame)

    #     pos = a.pos()
    #     rot_vec = RobotState.mat_to_vec(a.rot())
        
    #     state_data.update([(l.name + "_pos" , pos.tolist())])
    #     state_data.update([(l.name + "_rot" , rot_vec.tolist())])
    #     state_data.update([(l.name + "_vel" , veloc.tolist())])
    #     state_data.update([(l.name + "_acc" , accel.tolist())])

    #     for joint_id in l.connect_joint:
    #       if joint.id != joint_id:
    #         j = self.joints[joint_id]
    #         self.__kinematics_tree(l, j, motions, state_data)    

  def update_kinematics(self):
    state_data = {}
    self.__kinematics_tree(self.robot.joints[0], self.motions, state_data)

    self.state.import_state(state_data)