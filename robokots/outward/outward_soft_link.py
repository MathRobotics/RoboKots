#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki
# outward computation module from motion and robot_model to state

import numpy as np

from mathrobo import SO3, SE3, CMTM, numerical_difference, build_integrator

from ..basic.robot import RobotStruct, LinkStruct, JointStruct
from ..basic.motion import RobotMotions
from ..kinematics import *
from ..dynamics.dynamics import *
from ..basic.state_dict import *

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

def calc_link_local_point_frame(robot : RobotStruct, motions : RobotMotions, state : dict, link_name : str, point : float) -> SE3:
  link_data = convert_link_to_data(robot.link(link_name))
  link_motion = motions.link_motions(link_data.dof, link_data.dof_index, 1)
  link_frame = state_dict_to_frame(state, link_data.name)
  return link_frame @ SE3.set_mat(SE3.exp(SE3.hat(calc_soft_link_strain(link_data, link_motion[0])), point))

def calc_link_total_point_frame(robot : RobotStruct, motions : RobotMotions, state : dict, point : float) -> SE3:
  base = 0.0
  for l in robot.links:
      if point > base + l.length:
          base += l.length
          continue
      return calc_link_local_point_frame(robot, motions, state, l.name, point - base)

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
