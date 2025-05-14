#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025.04.05 Created by T.Ishigaki
# kinematics module

import numpy as np

from mathrobo import SE3, CMTM

from dataclasses import dataclass

@dataclass
class JointData:
    origin: SE3 # origin frame
    select_mat: np.ndarray # selection matrix
    dof: int = 0 # degree of freedom
    select_indeces: np.ndarray = None # indeces of the selection matrix

def joint_local_frame(joint : JointData, joint_coord : np.ndarray) -> SE3:  
  if len(joint_coord) != 0:
    v = joint.select_mat@joint_coord
  else:
    v = joint.select_mat@np.zeros(1)
  frame = SE3.set_mat(SE3.exp(v))
  return frame

def joint_local_tan_vec(joint : JointData, joint_vec : np.ndarray) -> np.ndarray:
  if len(joint_vec) != 0:
    vec = joint.select_mat @ joint_vec
  else:
    vec = np.zeros(6)
  return vec

def joint_local_vel(joint : JointData, joint_vel : np.ndarray) -> np.ndarray:
  return joint_local_tan_vec(joint, joint_vel)

def joint_local_acc(joint : JointData, joint_acc : np.ndarray) -> np.ndarray:
  return joint_local_tan_vec(joint, joint_acc)

def joint_local_cmtm(joint : JointData, joint_motions : np.ndarray, order = 3) -> CMTM:
  if order < 1:
    raise ValueError(f"Invalid order: {order}. Must be over 1.")
  
  dof = joint.dof

  frame = joint_local_frame(joint, joint_motions[:dof].reshape(dof))
  vecs = np.zeros((order-1, 6))

  for i in range(order-1):
    vecs[i] = joint_local_tan_vec(joint, joint_motions[(i+1)*dof:(i+2)*dof].reshape(dof))

  m = CMTM[SE3](frame, vecs)
  return m

def link_rel_frame(joint : JointData, joint_coord : np.ndarray) -> SE3:
  joint_frame = joint_local_frame(joint, joint_coord)
  rel_frame =  joint.origin @ joint_frame
  return rel_frame

def link_rel_tan_vec(joint : JointData, joint_vec : np.ndarray) -> np.ndarray:
  return joint_local_tan_vec(joint, joint_vec)

def link_rel_vel(joint : JointData, joint_vel : np.ndarray) -> np.ndarray:
  return link_rel_tan_vec(joint, joint_vel)

def link_rel_acc(joint : JointData, joint_acc : np.ndarray) -> np.ndarray:
  return link_rel_tan_vec(joint, joint_acc)

def link_rel_cmtm(joint : JointData, joint_motions : np.ndarray, order = 3) -> CMTM:
  if order < 1:
    raise ValueError(f"Invalid order: {order}. Must be over 1.")
  
  dof = joint.dof

  frame = link_rel_frame(joint, joint_motions[:dof].reshape(dof))
  vecs = np.zeros((order-1, 6))
  for i in range(order-1):
    vecs[i] = link_rel_tan_vec(joint, joint_motions[(i+1)*dof:(i+2)*dof].reshape(dof))

  m = CMTM[SE3](frame, vecs)
  return m

def kinematics(joint : JointData, p_link_frame : SE3, joint_coord : np.ndarray) -> SE3:
  rel_frame = link_rel_frame(joint, joint_coord)
  frame = p_link_frame @ rel_frame
  return frame

def vel_kinematics(joint : JointData, p_link_vel : np.ndarray, joint_coord : np.ndarray, joint_veloc : np.ndarray) -> np.ndarray:
  rel_frame = link_rel_frame(joint, joint_coord)
  rel_vel = link_rel_vel(joint, joint_veloc)
  
  vel = rel_frame.mat_inv_adj() @ p_link_vel  + rel_vel
  return vel

def acc_kinematics(joint : JointData, p_link_vel : np.ndarray, p_link_acc : np.ndarray, joint_coord : np.ndarray, joint_veloc : np.ndarray, joint_accel : np.ndarray) -> np.ndarray:
  rel_frame = link_rel_frame(joint, joint_coord)
  rel_vel = link_rel_vel(joint, joint_veloc)
  rel_acc = link_rel_acc(joint, joint_accel)
  
  acc =  rel_frame.mat_inv_adj() @ p_link_acc + SE3.hat_adj( rel_frame.mat_inv_adj() @ p_link_vel ) @ rel_vel + rel_acc
  return acc

def kinematics_cmtm(joint : JointData, p_link_cmtm : CMTM, joint_motions : np.ndarray, order = 3) -> CMTM:
  rel_m = link_rel_cmtm(joint, joint_motions, order)
  m = p_link_cmtm @ rel_m
  return m

def part_link_jacob(joint : JointData, rel_frame : np.ndarray) -> np.ndarray:
  return rel_frame.mat_inv_adj()[:, joint.select_indeces]
  

# specific 3D space (magic number 6)
def part_link_cmtm_jacob(joint : JointData, rel_cmtm : CMTM, joint_cmtm : CMTM) -> np.ndarray:
  '''
  jacobian matrix which map joint space to cmtm space wrt to a link
  Args:
    joint : JointData
    rel_cmtm : CMTM of the relative link
    joint_cmtm : CMTM of the joint
    return : jacobian matrix (joint space -> cmtm tangent space)
  '''

  mat = np.zeros((rel_cmtm._n * 6, rel_cmtm._n * joint.dof))
  tmp = rel_cmtm.mat_inv_adj() @ joint_cmtm.tan_map_adj() 

  for i in range(rel_cmtm._n):
    for j in range(i+1):
      mat[i*6:(i+1)*6, j*joint.dof:(j+1)*joint.dof] = (tmp[i*6:(i+1)*6,j*6:(j+1)*6])[:, joint.select_indeces]
      
  return mat