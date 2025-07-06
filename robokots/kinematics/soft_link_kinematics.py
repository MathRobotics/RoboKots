#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025.04.05 Created by T.Ishigaki
# kinematics module

import numpy as np
from scipy.linalg import expm

from mathrobo import SE3, CMTM
from mathrobo import gq_integrate

from dataclasses import dataclass

@dataclass
class SoftLinkData:
    origin_coord: np.ndarray
    select_mat: np.ndarray # selection matrix
    length: float = 0.0 # length of the soft link
    dof: int = 0 # degree of freedom
    select_indeces: np.ndarray = None # indeces of the selection matrix
    
def calc_soft_link_strain(soft_link : SoftLinkData, soft_link_coord : np.ndarray) -> np.ndarray:
    if len(soft_link_coord) != 0:
        strain = soft_link.select_mat @ soft_link_coord + soft_link.origin_coord
    else:
        strain = soft_link.origin_coord
    return strain

def calc_local_tan_mat(soft_link : SoftLinkData, 
                   soft_link_motion : np.ndarray, 
                   order : int = 1) -> np.ndarray:
    if order < 1:
        raise ValueError(f"Invalid order: {order}. Must be over 1.")
    
    if order == 1:
        tan_mat = SE3.exp_integ_adj(calc_soft_link_strain(soft_link, soft_link_motion), soft_link.length)
    else:
        motion = soft_link_motion.reshape((order, soft_link.dof))
        strains = (soft_link.select_mat @ motion[i] for i in range(order))
        strains[0] += soft_link.origin_coord
        def integrad(s):
            x = expm(-s * soft_link.length * CMTM.hat_adj(SE3, strains))
            return x
        tan_mat = gq_integrate(integrad, 0, 1)

    return tan_mat

def soft_link_local_frame(soft_link : SoftLinkData, soft_link_coord : np.ndarray) -> SE3:  
  frame = SE3.set_mat(SE3.exp(calc_soft_link_strain(soft_link, soft_link_coord), soft_link.length))
  return frame

def soft_link_local_vel(soft_link : SoftLinkData, 
                        soft_link_coord : np.ndarray, 
                        soft_link_vel : np.ndarray) -> np.ndarray:
  if len(soft_link_vel) != 0:
    v = -SE3.exp_integ_adj(-calc_soft_link_strain(soft_link, soft_link_coord), soft_link.length) @ soft_link.select_mat @ soft_link_vel
  else:
    v = np.zeros(6)
  return v

def soft_link_local_acc(soft_link : SoftLinkData, 
                        soft_link_coord : np.ndarray, 
                        soft_link_vel : np.ndarray, 
                        soft_link_acc : np.ndarray) -> np.ndarray:
  if len(soft_link_acc) != 0:
    def integrad(x):
      return SE3.exp(calc_soft_link_strain(soft_link, soft_link_coord), -x) @ SE3.hat_adj(SE3.exp_integ(calc_soft_link_strain(soft_link, soft_link_coord), -x) @ soft_link.select_mat @ soft_link_vel)
    
    v = -SE3.exp_integ_adj(-calc_soft_link_strain(soft_link, soft_link_coord), soft_link.length) @ soft_link.select_mat @ soft_link_acc + \
        gq_integrate( integrad, 0, soft_link.length) @ soft_link.select_mat @ soft_link_vel
  else:
    v = np.zeros(6)
  return v

def soft_link_local_cmtm(soft_link : SoftLinkData, soft_link_motions : np.ndarray, order = 3) -> CMTM:
  # if order < 1:
  #   raise ValueError(f"Invalid order: {order}. Must be over 1.")

  # dof = soft_link.dof

  # frame = soft_link_local_frame(soft_link, soft_link_motions[:dof].reshape(dof))
  # vecs = np.zeros((order-1, 6))

  # vecs = np.zeros((order-1, 6))
  # if order > 1:
  #   vec = calc_local_tan_mat(soft_link, soft_link_motions[:(order-1)*6], order-1) @ soft_link_motions[6:]
  #   vecs = vec.reshape((order-1, 6))

  # m = CMTM[SE3](frame, vecs)
  
  mot = soft_link_motions.copy()
  mot[:6] += soft_link.origin_coord
  mat = CMTM.set_mat(SE3, expm(soft_link.length * CMTM.hat(SE3, mot.reshape((order, 6)))))

  return mat

def kinematics(soft_link : SoftLinkData, p_link_frame : SE3, soft_link_coord : np.ndarray) -> SE3:
  frame = p_link_frame @ soft_link_local_frame(soft_link, soft_link_coord)
  return frame

def vel_kinematics(soft_link : SoftLinkData, p_link_vel : np.ndarray, soft_link_coord : np.ndarray, soft_link_veloc : np.ndarray) -> np.ndarray:
  rel_frame = soft_link_local_frame(soft_link, soft_link_coord)
  rel_vel = soft_link_local_vel(soft_link, soft_link_coord, soft_link_veloc)

  vel = rel_frame.mat_inv_adj() @ p_link_vel  + rel_vel
  return vel

def acc_kinematics(soft_link : SoftLinkData, p_link_vel : np.ndarray, p_link_acc : np.ndarray, soft_link_coord : np.ndarray, soft_link_veloc : np.ndarray, soft_link_accel : np.ndarray) -> np.ndarray:
  rel_frame = soft_link_local_frame(soft_link, soft_link_coord)
  rel_vel = soft_link_local_vel(soft_link, soft_link_coord, soft_link_veloc)
  rel_acc = soft_link_local_acc(soft_link, soft_link_coord, soft_link_veloc, soft_link_accel)

  acc =  rel_frame.mat_inv_adj() @ p_link_acc + SE3.hat_adj( rel_frame.mat_inv_adj() @ p_link_vel ) @ rel_vel + rel_acc
  return acc

def kinematics_cmtm(soft_link : SoftLinkData, p_link_cmtm : CMTM, soft_link_motions : np.ndarray, order = 3) -> CMTM:
  if soft_link.dof * order != len(soft_link_motions):
    raise ValueError(f"Invalid soft link motions: {len(soft_link_motions)}. Must be {soft_link.dof * order}.")
  rel_m = soft_link_local_cmtm(soft_link, soft_link_motions, order)
  m = p_link_cmtm @ rel_m
  return m

def part_soft_link_jacob(soft_link : SoftLinkData, soft_link_coord : np.ndarray, rel_frame : np.ndarray) -> np.ndarray:
  return ( rel_frame.mat_inv_adj() @ calc_local_tan_mat(soft_link, soft_link_coord) )[:, soft_link.select_indeces]

# specific 3D space (magic number 6)
def part_soft_link_cmtm_tan_jacob(soft_link : SoftLinkData,  soft_link_coord : np.ndarray, rel_cmtm : CMTM, soft_link_cmtm : CMTM) -> np.ndarray:
  '''
  jacobian matrix which map soft link space to cmtm space wrt to a link
  Args:
    soft_link : SoftLinkData
    rel_cmtm : CMTM of the relative link
    soft_link_cmtm : CMTM of the soft link
    return : jacobian matrix (soft link space -> cmtm tangent space)
  '''

  mat = np.zeros((rel_cmtm._n * 6, rel_cmtm._n * soft_link.dof))
  tmp = rel_cmtm.mat_inv_adj() @ calc_local_tan_mat(soft_link, soft_link_coord) @ soft_link_cmtm.tan_map()

  for i in range(rel_cmtm._n):
    for j in range(i+1):
      mat[i*6:(i+1)*6, j*soft_link.dof:(j+1)*soft_link.dof] = (tmp[i*6:(i+1)*6,j*6:(j+1)*6])[:, soft_link.select_indeces]

  return mat

def part_soft_link_cmtm_jacob(soft_link : SoftLinkData,  soft_link_coord : np.ndarray, rel_cmtm : CMTM, soft_link_cmtm : CMTM, link_cmtm : CMTM) -> np.ndarray:
  return link_cmtm.tan_map_inv() @ part_soft_link_cmtm_tan_jacob(soft_link, soft_link_coord, rel_cmtm, soft_link_cmtm)