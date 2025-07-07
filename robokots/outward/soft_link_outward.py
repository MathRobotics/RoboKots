#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki
# outward computation module from motion and robot_model to state

import numpy as np

from mathrobo import SO3, SE3, CMTM, numerical_grad, numerical_difference, build_integrator

from ..basic.robot import RobotStruct, LinkStruct, JointStruct
from ..basic.motion import RobotMotions
from ..kinematics.soft_link_kinematics import *
from ..dynamics.dynamics import *
from ..basic.state_dict import *

from .base import convert_link_to_data

def kinematics(robot : RobotStruct, motions : RobotMotions, order = 3) -> dict:
  '''
  Forward kinematics computation
  Args:
    robot (RobotStruct): robot model
    motions (RobotMotions): robot motion
  Returns:
    dict: state data
  '''
  
  state_data = {}
  state_cmtm = {}

  # Initialize CMTM for the world link
  # The world link is the parent of the first joint
  world_name = robot.links[robot.joints[0].parent_link_id].name
  state_cmtm.update([(world_name, CMTM.eye(SE3, order))])
 
  state = cmtm_to_state_list(state_cmtm[world_name], world_name)
  state_data.update(state)
  
  for joint in robot.joints:
    parent = robot.links[joint.parent_link_id]
    child = robot.links[joint.child_link_id]
    
    link_data = convert_link_to_data(child)
    
    link_motions = motions.link_motions(child.dof, child.dof_index, order)

    p_link_cmtm = state_cmtm[parent.name]

    rel_cmtm = soft_link_local_cmtm(link_data, link_motions, order)
    state = cmtm_to_state_list(rel_cmtm, child.name)
    state_data.update(state)

    link_cmtm = p_link_cmtm @ rel_cmtm
    # Update CMTM for the child link
    state_cmtm.update([(child.name, link_cmtm)])

    state = cmtm_to_state_list(link_cmtm, child.name)
    state_data.update(state)

  return state_data

def __target_part_link_jacob(target_link : LinkStruct, link : LinkStruct, link_coord : np.ndarray, rel_frame : SE3) -> np.ndarray:
  if target_link.id == link.id:
    mat = link.select_mat
  else:
    link_data = convert_link_to_data(link)
    mat = part_soft_link_jacob(link_data, link_coord, rel_frame)  
  return mat

# specific 3d space (magic number 6)
def __target_part_link_cmtm_tan_jacob(target_link : LinkStruct, link : LinkStruct,  link_motion : np.ndarray, rel_cmtm : CMTM, link_cmtm : CMTM) -> np.ndarray:
  '''
  Compute the Jacobian matrix for the target part link in CMTM space.
  Args:
    target_link (LinkStruct): target link
    link (LinkStruct): link structure
    rel_cmtm (CMTM): relative CMTM
    joint_cmtm (CMTM): joint CMTM
  '''
  mat = np.zeros((rel_cmtm._n * 6, rel_cmtm._n * link.dof))
  if target_link.id == link.id:
    tmp = calc_local_tan_mat(link, link_motion) @ link_cmtm.tan_map()
    for i in range(rel_cmtm._n):
      mat[i*6:(i+1)*6, i*link.dof:(i+1)*link.dof] = (tmp[i*6:(i+1)*6, i*6:(i+1)*6])[:, link.select_indeces]
  else:
    link_data = convert_link_to_data(link)
    mat = part_soft_link_cmtm_tan_jacob(link_data, link_motion, rel_cmtm, link_cmtm)
  return mat

def __link_jacobian(robot : RobotStruct, motions: RobotMotions, state : dict, target_link : LinkStruct) -> np.ndarray:
  jacob = np.zeros((6,robot.dof))
  link_route = []
  joint_route = []
  robot.route_target_link(target_link, link_route, joint_route)
  print(f"Link route: {link_route}, Joint route: {joint_route}")
  for l in link_route:
    link = robot.links[l]
    if link.dof < 1:
      continue
    link_motion = motions.link_motions(link.dof, link.dof_index, 1)
    rel_frame = state_dict_to_rel_frame(state, link.name, target_link.name)
    mat = __target_part_link_jacob(target_link, link, link_motion, rel_frame)
    jacob[:,link.dof_index:link.dof_index+link.dof] = mat

  return jacob

def __link_cmtm_tan_jacobian(robot, motions : RobotMotions, state : dict, target_link : LinkStruct, order : int) -> np.ndarray:
  jacob = np.zeros((6*order,robot.dof*order))
  link_route = []
  joint_route = []
  robot.route_target_link(target_link, link_route, joint_route)
  
  for l in link_route:
    link = robot.links[l]
    if link.dof > 0:
      rel_cmtm = state_dict_to_rel_cmtm(state, link.name, target_link.name, order)

      link_cmtm = state_dict_to_cmtm(state, link.name, order)
      mat = __target_part_link_cmtm_tan_jacob(target_link, link, motions.link_motions(link.dof, link.dof_index, order), rel_cmtm, link_cmtm)

      for i in range(order):
        jacob[:,i*robot.dof+link.dof_index:i*robot.dof+link.dof_index+link.dof]  \
          = mat[:,i*link.dof:(i+1)*link.dof]
  return jacob

# specific 3d space (magic number 6)
def link_jacobian(robot : RobotStruct, motions : RobotMotions, state : dict, link_name_list : list[str]) -> np.ndarray:
  links = robot.link_list(link_name_list)
  jacobs = np.zeros((6*len(links),robot.dof))
  for i in range(len(links)):
    jacobs[6*i:6*(i+1),:] = __link_jacobian(robot, motions, state, links[i])
  return jacobs

# specific 3d space (magic number 6)
def link_cmtm_jacobian(robot : RobotStruct, motions : RobotMotions, state : dict, link_name_list : list[str], order = 3) -> np.ndarray:
  links = robot.link_list(link_name_list)
  jacobs = np.zeros((6*order*len(links),robot.dof*order))
  for i in range(len(links)):
    link_cmtm = state_dict_to_cmtm(state, link_name_list[i], order)
    jacobs[6*order*i:6*order*(i+1),:] \
      = link_cmtm.tan_map_inv() @ __link_cmtm_tan_jacobian(robot, motions, state, links[i], order)

  return jacobs

def link_diff_kinematics_numerical(robot : RobotStruct, motions : RobotMotions, link_name_list : list[str],  data_type : str, order = None, \
                                    eps = 1e-8, update_method = None, update_direction = None) -> np.ndarray:
  if data_type not in ["pos", "rot", "vel", "acc", "jerk", "frame", "cmtm"]:
    raise ValueError(f"Invalid data_type: {data_type}. Must be 'pos', 'rot', 'vel', 'acc', 'frame' or 'cmtm'.")

  dof = 6

  if order is None:
    order = 3

  if data_type == "pos" or data_type == "rot":
    dof = 3
  elif data_type == "frame":
    dof = 6
  elif data_type == "vel":
    dof = 6
  elif data_type == "acc":
    dof = 6
  elif data_type == "jerk":
    dof = 6
  elif data_type == "cmtm":
    if order is None:
      dof = 3 * 6
    else:
      dof = order * 6

  diff = np.zeros((len(link_name_list), dof))

  def update_func(x_init, direct, eps):
    x_ = x_init.copy()

    if update_method is None:
      D, d = build_integrator(robot.dof, order, eps, method="poly")
    else:
      D, d = build_integrator(robot.dof, order, eps, method=update_method)

    x_ = D @ x_init + d @ direct
    return x_

  for i in range(len(link_name_list)):
    def kinematics_func(x):
      motions.motions = x
      state = kinematics(robot, motions, order)
      y = extract_dict_link_info(state, data_type, link_name_list[i])
      return y

    if data_type == "rot":
      diff[i] = numerical_difference(motions.motions, kinematics_func, sub_func = SO3.sub_tan_vec, update_func = update_func, direction = update_direction)
    if data_type == "frame":
      diff[i] = numerical_difference(motions.motions, kinematics_func, sub_func = SE3.sub_tan_vec, update_func = update_func, direction = update_direction)
    elif data_type == "cmtm":
      diff[i] = numerical_difference(motions.motions, kinematics_func, sub_func = CMTM.sub_vec, update_func = update_func, direction = update_direction)
    else:
      diff[i] = numerical_difference(motions.motions, kinematics_func, update_func = update_func, direction = update_direction)
  
  return diff

def link_jacobian_numerical(robot : RobotStruct, motions : RobotMotions, link_name_list : list[str], data_type : str, order_ = None) -> np.ndarray:
  if data_type not in ["pos", "rot", "vel", "acc", "frame", "cmtm"]:
    raise ValueError(f"Invalid data_type: {data_type}. Must be 'pos', 'rot', 'vel', 'acc', 'frame' or 'cmtm'.")

  order = 3
  dof = 6
  if data_type == "pos" or data_type == "rot" :
    dof = 3
    order = 1
  elif data_type == "frame":
    order = 1
  elif data_type == "vel":
    order = 2
  elif data_type == "acc":
    order = 3
  elif data_type == "cmtm":
    if order_ is None:
      order = 3
    else:
      order = order_

  if data_type  == "cmtm":
    jacobs = np.zeros((dof*order*len(link_name_list),robot.dof*order))
  else:
    jacobs = np.zeros((dof*len(link_name_list),robot.dof*order))
  motion = motions.motions[:robot.dof*order]

  for i in range(len(link_name_list)):
    def kinematics_func(x):
      motions.motions = x
      state = kinematics(robot, motions, order)
      y = extract_dict_link_info(state, data_type, link_name_list[i])
      return y

    if data_type == "rot":
      jacobs[dof*i:dof*(i+1)] = numerical_grad(motions.motions, kinematics_func, sub_func = SO3.sub_tan_vec)
    elif data_type == "frame":
      jacobs[dof*i:dof*(i+1)] = numerical_grad(motion, kinematics_func, sub_func = SE3.sub_tan_vec)
    elif data_type == "cmtm":
      state = kinematics(robot, motions, order)
      jacobs[(dof*order)*i:(6*order)*(i+1)] = \
        extract_dict_link_info(state, data_type, link_name_list[i]).tan_map_inv() @ numerical_grad(motion, kinematics_func, sub_func = CMTM.sub_tan_vec_var)
    else:
      jacobs[dof*i:dof*(i+1)] = numerical_grad(motion, kinematics_func)

  return jacobs
