#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.07.23 Created by T.Ishigaki

import numpy as np

from mathrobo import *

from ..basic import *

from ..gen_value import *
from ..state import *

class LinkKinematics:
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
  def link_rel_frame(link, joint, parent, joint_coord):
    if parent:
      p_connect_frame = parent.connect_adj_frames[joint.id] 
    else:
      p_connect_frame = np.identity(6)
    joint_frame = LinkKinematics.joint_local_frame(joint, joint_coord)
    connect_frame = link.connect_adj_frames[joint.id]
    rel_frame = p_connect_frame @ joint_frame @ connect_frame
    return rel_frame

  @staticmethod
  def link_rel_vel(link, joint, joint_vel):
    local_vel = LinkKinematics.joint_local_vel(joint, joint_vel)
    connect_frame = link.connect_adj_frames[joint.id]
    rel_vel = np.linalg.inv(connect_frame) @ local_vel
    return rel_vel
  
  @staticmethod
  def link_rel_acc(link, joint, joint_acc):
    local_acc = LinkKinematics.joint_local_acc(joint, joint_acc)
    connect_frame = link.connect_adj_frames[joint.id]
    rel_acc = np.linalg.inv(connect_frame) @ local_acc
    return rel_acc
  
  @staticmethod
  def kinematics(link, joint, parent, motinos, state):
    joint_coord = motinos.joint_coord(joint)
    if parent:
      rot = RobotState.vec_to_mat(state[parent.name + "_rot"])
      h = SE3(rot, state[parent.name + "_pos"])
      p_link_frame = h.adjoint()
    else:
      p_link_frame = np.identity(6)
    rel_frame = LinkKinematics.link_rel_frame(link, joint, parent, joint_coord)
    frame = p_link_frame @ rel_frame
    return frame

  @staticmethod
  def vel_kinematics(link, joint, parent, motinos, state):
    joint_coord = motinos.joint_coord(joint)
    joint_veloc = motinos.joint_veloc(joint)
    if parent:
      link_vel = state[parent.name + "_vel"]
    else:
      link_vel = np.zeros(6)
    rel_frame = LinkKinematics.link_rel_frame(link, joint, parent, joint_coord)
    rel_vel = LinkKinematics.link_rel_vel(link, joint, joint_veloc)
    vel = np.linalg.inv(rel_frame) @ link_vel  + rel_vel
    return vel

  @staticmethod
  def acc_kinematics(link, joint, parent, motinos, state):
    joint_coord = motinos.joint_coord(joint)
    joint_veloc = motinos.joint_veloc(joint)
    joint_accel = motinos.joint_accel(joint)
    if parent:
      link_vel = state[parent.name + "_vel"]
      link_acc = state[parent.name + "_acc"]
    else:
      link_vel = np.zeros(6)
      link_acc = np.zeros(6)
    rel_frame = LinkKinematics.link_rel_frame(link,  joint, parent, joint_coord)
    rel_vel = LinkKinematics.link_rel_vel(link, joint, joint_veloc)
    rel_acc = LinkKinematics.link_rel_acc(link, joint, joint_accel)
    acc = rel_frame @ link_acc + SE3.hat_adj( rel_frame @ rel_vel ) @ link_vel + rel_acc
    return acc