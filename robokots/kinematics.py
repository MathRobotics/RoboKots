#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025.04.05 Created by T.Ishigaki
# kinematics module

import numpy as np

from mathrobo import SE3, CMTM

from .robot_model import JointStruct

def joint_local_frame(joint : JointStruct, joint_coord : np.ndarray) -> SE3:  
  if len(joint_coord) != 0:
    v = joint.joint_select_mat@joint_coord
  else:
    v = joint.joint_select_mat@np.zeros(1)
  frame = SE3.set_mat(SE3.exp(v))
  return frame

def joint_local_vel(joint : JointStruct, joint_vel : np.ndarray) -> np.ndarray:
  if len(joint_vel) != 0:
    vel = joint.joint_select_mat @ joint_vel
  else:
    vel = np.zeros(6)
  return vel

def joint_local_acc(joint : JointStruct, joint_acc : np.ndarray) -> np.ndarray:
  if len(joint_acc) != 0:
    acc = joint.joint_select_mat @ joint_acc
  else:
    acc = np.zeros(6)
  return acc

def link_rel_frame(joint : JointStruct, joint_coord : np.ndarray) -> SE3:
  joint_frame = joint_local_frame(joint, joint_coord)
  rel_frame =  joint.origin @ joint_frame
  return rel_frame

def link_rel_vel(joint : JointStruct, joint_vel : np.ndarray) -> np.ndarray:
  local_vel = joint_local_vel(joint, joint_vel)
  rel_vel = joint.origin.mat_inv_adj() @ local_vel
  return rel_vel

def link_rel_acc(joint : JointStruct, joint_acc : np.ndarray) -> np.ndarray:
  local_acc = joint_local_acc(joint, joint_acc)
  rel_acc = joint.origin.mat_inv_adj() @ local_acc
  return rel_acc

def kinematics(joint : JointStruct, p_link_frame : SE3, joint_coord : np.ndarray) -> SE3:
  rel_frame = link_rel_frame(joint, joint_coord)
  frame = p_link_frame @ rel_frame
  return frame

def vel_kinematics(joint : JointStruct, p_link_vel : np.ndarray, joint_coord : np.ndarray, joint_veloc : np.ndarray) -> np.ndarray:
  rel_frame = link_rel_frame(joint, joint_coord)
  rel_vel = link_rel_vel(joint, joint_veloc)
  
  vel = rel_frame.mat_inv_adj() @ p_link_vel  + rel_vel
  return vel

def acc_kinematics(joint : JointStruct, p_link_vel : np.ndarray, p_link_acc : np.ndarray, joint_coord : np.ndarray, joint_veloc : np.ndarray, joint_accel : np.ndarray) -> np.ndarray:
  rel_frame = link_rel_frame(joint, joint_coord)
  rel_vel = link_rel_vel(joint, joint_veloc)
  rel_acc = link_rel_acc(joint, joint_accel)
  
  acc =  rel_frame.mat_inv_adj() @ p_link_acc + SE3.hat_adj( rel_frame @ rel_vel ) @ p_link_vel + rel_acc
  return acc

def part_link_jacob(joint : JointStruct, rel_frame : np.ndarray) -> np.ndarray:
  return rel_frame.mat_inv_adj() @ joint.origin.mat_inv_adj() @ joint.joint_select_mat

def link_rel_cmtm(joint : JointStruct, joint_coord : np.ndarray, joint_vel : np.ndarray, joint_acc : np.ndarray) -> CMTM:
  frame = link_rel_frame(joint, joint_coord)
  vel = link_rel_vel(joint, joint_vel)
  acc = link_rel_acc(joint, joint_acc)
  vec = np.array((vel, acc))
  m = CMTM[SE3](frame, vec)
  return m

def kinematics_cmtm(joint : JointStruct, p_link_cmtm : CMTM, joint_coord : np.ndarray, joint_veloc : np.ndarray, joint_accel : np.ndarray) -> CMTM:
  rel_m = link_rel_cmtm(joint, joint_coord, joint_veloc, joint_accel)
  m = p_link_cmtm @ rel_m
  return m
