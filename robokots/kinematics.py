#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025.04.05 Created by T.Ishigaki
# kinematics module

import numpy as np

from mathrobo import SE3, CMTM

from .robot import JointStruct

def joint_local_frame(joint : JointStruct, joint_coord : np.ndarray) -> SE3:  
  if len(joint_coord) != 0:
    v = joint.select_mat@joint_coord
  else:
    v = joint.select_mat@np.zeros(1)
  frame = SE3.set_mat(SE3.exp(v))
  return frame

def joint_local_vel(joint : JointStruct, joint_vel : np.ndarray) -> np.ndarray:
  if len(joint_vel) != 0:
    vel = joint.select_mat @ joint_vel
  else:
    vel = np.zeros(6)
  return vel

def joint_local_acc(joint : JointStruct, joint_acc : np.ndarray) -> np.ndarray:
  if len(joint_acc) != 0:
    acc = joint.select_mat @ joint_acc
  else:
    acc = np.zeros(6)
  return acc

#specific 3d-CMTM
def joint_local_cmtm(joint : JointStruct, joint_motions : np.ndarray, order = 3) -> CMTM:
  if order < 1 or order > 3:
    print("order is out of range")
    return 
  
  dof = joint.dof

  frame = joint_local_frame(joint, joint_motions[:dof].reshape(dof))
  vecs = np.zeros((order-1, 6))
  if order > 1:
    vecs[0] = joint_local_vel(joint, joint_motions[dof:2*dof].reshape(dof))
  if order > 2:
    vecs[1] = joint_local_acc(joint, joint_motions[2*dof:3*dof].reshape(dof))
  m = CMTM[SE3](frame, vecs)
  return m

def link_rel_frame(joint : JointStruct, joint_coord : np.ndarray) -> SE3:
  joint_frame = joint_local_frame(joint, joint_coord)
  rel_frame =  joint.origin @ joint_frame
  return rel_frame

def link_rel_vel(joint : JointStruct, joint_vel : np.ndarray) -> np.ndarray:
  local_vel = joint_local_vel(joint, joint_vel)
  rel_vel = local_vel
  return rel_vel

def link_rel_acc(joint : JointStruct, joint_acc : np.ndarray) -> np.ndarray:
  local_acc = joint_local_acc(joint, joint_acc)
  rel_acc = local_acc
  return rel_acc

#specific 3d-CMTM
def link_rel_cmtm(joint : JointStruct, joint_motions : np.ndarray, order = 3) -> CMTM:
  if order < 1 or order > 3:
    print("order is out of range")
    return 
  
  dof = joint.dof

  frame = link_rel_frame(joint, joint_motions[:dof].reshape(dof))
  vecs = np.zeros((order-1, 6))
  if order > 1:
    vecs[0] = link_rel_vel(joint, joint_motions[dof:2*dof].reshape(dof))
  if order > 2:
    vecs[1] = link_rel_acc(joint, joint_motions[2*dof:3*dof].reshape(dof))
  m = CMTM[SE3](frame, vecs)
  return m

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
  
  acc =  rel_frame.mat_inv_adj() @ p_link_acc + SE3.hat_adj( rel_frame.mat_inv_adj() @ p_link_vel ) @ rel_vel + rel_acc
  return acc

#specific 3d-CMTM
def kinematics_cmtm(joint : JointStruct, p_link_cmtm : CMTM, joint_motions : np.ndarray, order = 3) -> CMTM:
  rel_m = link_rel_cmtm(joint, joint_motions)
  m = p_link_cmtm @ rel_m
  return m

def part_link_jacob(joint : JointStruct, rel_frame : np.ndarray) -> np.ndarray:
  return joint.selector(rel_frame.mat_inv_adj())

# specific 3D space (magic number 6)
def part_link_cmtm_jacob(joint : JointStruct, rel_cmtm : CMTM, joint_cmtm : CMTM) -> np.ndarray:
  mat = np.zeros((rel_cmtm._n * 6, rel_cmtm._n * joint.dof))
  tmp = rel_cmtm.mat_inv_adj() @ CMTM.ptan_to_tan(6, rel_cmtm._n) @ joint_cmtm.tan_mat_adj()

  for i in range(rel_cmtm._n):
    for j in range(i+1):
      mat[i*6:(i+1)*6, j*joint.dof:(j+1)*joint.dof] = joint.selector(tmp[i*6:(i+1)*6,j*6:(j+1)*6])
      
  return mat