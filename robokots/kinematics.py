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
    frame = SE3.set_mat(SE3.exp(v))
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
    rel_frame =  joint.origin @ joint_frame
    return rel_frame

  @staticmethod
  def link_rel_vel(joint, joint_vel):
    local_vel = Kinematics.joint_local_vel(joint, joint_vel)
    rel_vel = joint.origin.adj_inv() @ local_vel
    return rel_vel
  
  @staticmethod
  def link_rel_acc(joint, joint_acc):
    local_acc = Kinematics.joint_local_acc(joint, joint_acc)
    rel_acc = joint.origin.adj_inv() @ local_acc
    return rel_acc
  
  @staticmethod
  def kinematics(joint, p_link_frame, joint_coord):
    rel_frame = Kinematics.link_rel_frame(joint, joint_coord)
    frame = p_link_frame @ rel_frame
    return frame

  @staticmethod
  def vel_kinematics(joint, p_link_vel, joint_coord, joint_veloc):
    rel_frame = Kinematics.link_rel_frame(joint, joint_coord)
    rel_vel = Kinematics.link_rel_vel(joint, joint_veloc)
    
    vel = rel_frame.adj_inv() @ p_link_vel  + rel_vel
    return vel

  @staticmethod
  def acc_kinematics(joint, p_link_vel, p_link_acc, joint_coord, joint_veloc, joint_accel):
    rel_frame = Kinematics.link_rel_frame(joint, joint_coord)
    rel_vel = Kinematics.link_rel_vel(joint, joint_veloc)
    rel_acc = Kinematics.link_rel_acc(joint, joint_accel)
    
    acc =  rel_frame.adj_inv() @ p_link_acc + SE3.hat_adj( rel_frame @ rel_vel ) @ p_link_vel + rel_acc
    return acc