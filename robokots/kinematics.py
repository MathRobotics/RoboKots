#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki

import numpy as np

from mathrobo import *

from .motion import *
from .state import *

class Kinematics:
  @staticmethod
  def joint_local_frame(joint, joint_angle):
    if len(joint_angle) != 0:
      v = joint.joint_select_mat@joint_angle
    else:
      v = joint.joint_select_mat@np.zeros(1)
    frame = SE3.exp_adj(v)
    return frame
  
  @staticmethod
  def joint_local_vel(joint, joint_vel):
    if len(joint_vel) != 0:
      vel = joint.joint_select_mat @ joint_vel
    else:
      vel = np.zeros(6)
    return vel
  
  @staticmethod
  def joint_local_acc(joint, joint_acc):
    if len(joint_acc) != 0:
      acc = joint.joint_select_mat @ joint_acc
    else:
      acc = np.zeros(6)
    return acc
  
  @staticmethod
  def link_rel_frame(joint, joint_coord):
    joint_frame = Kinematics.joint_local_frame(joint, joint_coord)
    rel_frame =  joint.origin.adj_mat() @ joint_frame
    return rel_frame

  @staticmethod
  def link_rel_vel(joint, joint_vel):
    local_vel = Kinematics.joint_local_vel(joint, joint_vel)
    rel_vel = np.linalg.inv(joint.origin.adj_mat()) @ local_vel
    return rel_vel
  
  @staticmethod
  def link_rel_acc(joint, joint_acc):
    local_acc = Kinematics.joint_local_acc(joint, joint_acc)
    rel_acc = np.linalg.inv(joint.origin.adj_mat()) @ local_acc
    return rel_acc
  
  @staticmethod
  def kinematics(joint, links, motinos, state):
    joint_coord = motinos.joint_coord(joint)
    parent = links[joint.parent_link]

    rot = np.array(state[parent.name + "_rot"]).reshape((3,3))
    p_link_frame = SE3(rot, state[parent.name + "_pos"]).adj_mat()

    rel_frame = Kinematics.link_rel_frame(joint, joint_coord)
    frame = p_link_frame @ rel_frame
    return frame

  @staticmethod
  def vel_kinematics(joint, links, motinos, state):
    joint_coord = motinos.joint_coord(joint)
    joint_veloc = motinos.joint_veloc(joint)
    
    parent = links[joint.parent_link]

    link_vel = state[parent.name + "_vel"]    

    rel_frame = Kinematics.link_rel_frame(joint, joint_coord)
    rel_vel = Kinematics.link_rel_vel(joint, joint_veloc)
    
    vel = np.linalg.inv(rel_frame) @ link_vel  + rel_vel
    return vel

  @staticmethod
  def acc_kinematics(joint, links, motinos, state):
    joint_coord = motinos.joint_coord(joint)
    joint_veloc = motinos.joint_veloc(joint)
    joint_accel = motinos.joint_accel(joint)
    
    parent = links[joint.parent_link]

    link_vel = state[parent.name + "_vel"]  
    link_acc = state[parent.name + "_acc"]  

    rel_frame = Kinematics.link_rel_frame(joint, joint_coord)
    rel_vel = Kinematics.link_rel_vel(joint, joint_veloc)
    rel_acc = Kinematics.link_rel_acc(joint, joint_accel)
    
    acc = rel_frame @ link_acc + SE3.hat_adj( rel_frame @ rel_vel ) @ link_vel + rel_acc
    return acc