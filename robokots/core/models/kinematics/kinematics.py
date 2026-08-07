#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025.04.05 Created by T.Ishigaki
# kinematics module

import numpy as np

from mathrobo import SE3, CMTM

from .base import JointData

def local_tangent_mat(select_mat : np.ndarray, joint_coord : np.ndarray) -> np.ndarray:
  if select_mat.shape[1] == 0:
    return np.zeros((6, 0))
  tan = local_tan_vec(select_mat, joint_coord)
  return SE3.exp_integ_adj(-tan, 1.0) @ select_mat

def local_tan_vec(select_mat : np.ndarray, joint_vec : np.ndarray) -> np.ndarray:
  if len(joint_vec) == 0:
    return np.zeros(6)
  else:
    return select_mat @ joint_vec

def local_tan_vel(select_mat : np.ndarray, joint_coord : np.ndarray, joint_veloc : np.ndarray) -> np.ndarray:
  return local_tangent_mat(select_mat, joint_coord) @ joint_veloc

def local_tan_acc(
    select_mat : np.ndarray,
    joint_coord : np.ndarray,
    joint_veloc : np.ndarray,
    joint_accel : np.ndarray,
    eps : float = 1e-7,
) -> np.ndarray:
  tangent = local_tangent_mat(select_mat, joint_coord)
  if select_mat.shape[1] == 0:
    return np.zeros(6)
  if np.linalg.norm(joint_veloc) == 0.0:
    tangent_dot_vel = np.zeros(6)
  else:
    tangent_plus = local_tangent_mat(select_mat, joint_coord + eps * joint_veloc)
    tangent_minus = local_tangent_mat(select_mat, joint_coord - eps * joint_veloc)
    tangent_dot_vel = ((tangent_plus - tangent_minus) / (2.0 * eps)) @ joint_veloc
  return tangent @ joint_accel + tangent_dot_vel

def local_frame(select_mat : np.ndarray, joint_coord : np.ndarray) -> SE3:  
  return SE3.set_mat(SE3.exp(local_tan_vec(select_mat, joint_coord)))

def local_cmtm(select_mat : np.ndarray, joint_motions : np.ndarray, dof : int = 1, order = 3) -> CMTM:
  if order < 1:
    raise ValueError(f"Invalid order: {order}. Must be over 1.")

  frame = local_frame(select_mat, joint_motions[:dof].flatten())
  vecs = np.zeros((order-1, 6))

  for i in range(order-1):
    vecs[i] = local_tan_vec(select_mat, joint_motions[(i+1)*dof:(i+2)*dof].flatten())

  m = CMTM[SE3](frame, vecs)
  return m

def joint_local_frame(joint : JointData, joint_coord : np.ndarray) -> SE3:  
  return local_frame(joint.select_mat, joint_coord)

def joint_local_vel(joint_or_select, joint_coord_or_vel : np.ndarray, joint_veloc : np.ndarray = None) -> np.ndarray:
  if joint_veloc is None:
    return local_tan_vec(joint_or_select, joint_coord_or_vel)
  return local_tan_vel(joint_or_select.select_mat, joint_coord_or_vel, joint_veloc)

def joint_local_acc(joint_or_select, joint_coord_or_acc : np.ndarray, joint_veloc : np.ndarray = None, joint_accel : np.ndarray = None) -> np.ndarray:
  if joint_veloc is None and joint_accel is None:
    return local_tan_vec(joint_or_select, joint_coord_or_acc)
  return local_tan_acc(joint_or_select.select_mat, joint_coord_or_acc, joint_veloc, joint_accel)

joint_local_jerk = local_tan_vec

def joint_local_cmtm(joint : JointData, joint_motions : np.ndarray, order = 3) -> CMTM:
  if order < 1:
    raise ValueError(f"Invalid order: {order}. Must be over 1.")
  dof = joint.dof
  if dof <= 1:
    return local_cmtm(joint.select_mat, joint_motions, joint.dof, order)
  frame = joint_local_frame(joint, joint_motions[:dof].flatten())
  vecs = np.zeros((order-1, 6))
  if order > 1:
    coord = joint_motions[:dof].flatten()
    veloc = joint_motions[dof:2*dof].flatten()
    vecs[0] = joint_local_vel(joint, coord, veloc)
  if order > 2:
    accel = joint_motions[2*dof:3*dof].flatten()
    vecs[1] = joint_local_acc(joint, coord, veloc, accel)
  for i in range(2, order-1):
    vecs[i] = local_tan_vec(joint.select_mat, joint_motions[(i+1)*dof:(i+2)*dof].flatten())
  return CMTM[SE3](frame, vecs)

def joint_rel_frame(joint : JointData, joint_coord : np.ndarray) -> SE3:
  rel_frame =  joint.origin @ joint_local_frame(joint, joint_coord)
  return rel_frame

def joint_rel_cmtm(joint : JointData, joint_motions : np.ndarray, order = 3) -> CMTM:
  if order < 1:
    raise ValueError(f"Invalid order: {order}. Must be over 1.")
  
  dof = joint.dof
  if dof <= 1:
    frame = joint_rel_frame(joint, joint_motions[:dof].flatten())
    vecs = np.zeros((order-1, 6))
    for i in range(order-1):
      vecs[i] = local_tan_vec(joint.select_mat, joint_motions[(i+1)*dof:(i+2)*dof].flatten())
    return CMTM[SE3](frame, vecs)

  frame = joint_rel_frame(joint, joint_motions[:dof].flatten())
  vecs = np.zeros((order-1, 6))
  if order > 1:
    coord = joint_motions[:dof].flatten()
    veloc = joint_motions[dof:2*dof].flatten()
    vecs[0] = joint_local_vel(joint, coord, veloc)
  if order > 2:
    accel = joint_motions[2*dof:3*dof].flatten()
    vecs[1] = joint_local_acc(joint, coord, veloc, accel)
  for i in range(2, order-1):
    vecs[i] = local_tan_vec(joint.select_mat, joint_motions[(i+1)*dof:(i+2)*dof].flatten())

  m = CMTM[SE3](frame, vecs)
  return m

def kinematics(joint : JointData, p_link_frame : SE3, joint_coord : np.ndarray) -> SE3:
  rel_frame = joint_rel_frame(joint, joint_coord)
  frame = p_link_frame @ rel_frame
  return frame

def kinematics_vel(joint : JointData, p_link_vel : np.ndarray, joint_coord : np.ndarray, joint_veloc : np.ndarray) -> np.ndarray:
  rel_frame = joint_rel_frame(joint, joint_coord)
  rel_vel = joint_local_vel(joint, joint_coord, joint_veloc)

  vel = rel_frame.mat_inv_adj() @ p_link_vel  + rel_vel
  return vel

def kinematics_acc(joint : JointData, p_link_vel : np.ndarray, p_link_acc : np.ndarray, joint_coord : np.ndarray, joint_veloc : np.ndarray, joint_accel : np.ndarray) -> np.ndarray:
  rel_frame = joint_rel_frame(joint, joint_coord)
  rel_vel = joint_local_vel(joint, joint_coord, joint_veloc)
  rel_acc = joint_local_acc(joint, joint_coord, joint_veloc, joint_accel)

  acc =  rel_frame.mat_inv_adj() @ p_link_acc + SE3.hat_adj( rel_frame.mat_inv_adj() @ p_link_vel ) @ rel_vel + rel_acc
  return acc

def kinematics_jerk(joint : JointData, 
                    p_link_vel : np.ndarray, p_link_acc : np.ndarray, p_link_jerk : np.ndarray, 
                    joint_coord : np.ndarray, joint_veloc : np.ndarray, joint_accel : np.ndarray, joint_jerk : np.ndarray) -> np.ndarray:
  rel_frame = joint_rel_frame(joint, joint_coord)
  rel_vel = joint_local_vel(joint, joint_coord, joint_veloc)
  rel_acc = joint_local_acc(joint, joint_coord, joint_veloc, joint_accel)
  rel_jerk = joint_local_jerk(joint.select_mat, joint_jerk)
  return (rel_frame.mat_inv_adj() @ p_link_jerk
          + 2 * SE3.hat_adj(rel_frame.mat_inv_adj() @ p_link_acc) @ rel_vel
          + SE3.hat_adj(SE3.hat_adj(rel_frame.mat_inv_adj() @ p_link_vel) @ rel_vel) @ rel_vel
          + SE3.hat_adj(rel_frame.mat_inv_adj() @ p_link_vel) @ rel_acc
          + rel_jerk)

def kinematics_cmtm(joint : JointData, p_link_cmtm : CMTM, joint_motions : np.ndarray, order = 3) -> CMTM:
  if joint.dof * order != len(joint_motions):
    raise ValueError(f"Invalid joint motions: {len(joint_motions)}. Must be {joint.dof * order}.")
  rel_m = joint_rel_cmtm(joint, joint_motions, order)
  m = p_link_cmtm @ rel_m
  return m

def part_link_jacob(joint : JointData, rel_frame : np.ndarray, joint_coord : np.ndarray = None) -> np.ndarray:
  if joint_coord is None:
    tangent_mat = joint.select_mat
  else:
    tangent_mat = local_tangent_mat(joint.select_mat, joint_coord)
  return rel_frame.mat_inv_adj() @ tangent_mat

# specific 3D space (magic number 6)
def part_link_cmtm_tan_jacob(joint : JointData, rel_cmtm : CMTM, joint_cmtm : CMTM) -> np.ndarray:
  '''
  jacobian matrix which map joint space to cmtm space wrt to a link
  Args:
    joint : JointData
    rel_cmtm : CMTM of the relative link
    joint_cmtm : CMTM of the joint
    return : jacobian matrix (joint space -> cmtm tangent space)
  '''

  mat = np.zeros((rel_cmtm._n * 6, rel_cmtm._n * joint.dof))
  # tmp = rel_cmtm.mat_inv_adj() @ joint_cmtm.tan_map() 
  tmp = rel_cmtm.mat_inv_adj() @ joint_cmtm.tangent_mat()

  for i in range(rel_cmtm._n):
    for j in range(i+1):
      mat[i*6:(i+1)*6, j*joint.dof:(j+1)*joint.dof] = (
        tmp[i*6:(i+1)*6, j*6:(j+1)*6] @ joint.select_mat
      )
      
  return mat

def part_link_cmtm_jacob(joint : JointData, rel_cmtm : CMTM, joint_cmtm : CMTM, link_cmtm : CMTM) -> np.ndarray:
  # return link_cmtm.tan_map_inv() @ part_link_cmtm_tan_jacob(joint, rel_cmtm, joint_cmtm)
  return link_cmtm.tangent_mat_inv() @ part_link_cmtm_tan_jacob(joint, rel_cmtm, joint_cmtm)
