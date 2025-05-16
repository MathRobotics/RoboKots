#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki
# outward computation module from motion and robot_model to state

import numpy as np

from mathrobo import SO3, SE3, CMTM, numerical_grad, numerical_difference, build_integrator

from .basic.robot import RobotStruct, LinkStruct, JointStruct
from .basic.motion import RobotMotions
from .basic.state import RobotState
from .basic.kinematics import *
from .basic.dynamics import *
from .basic.state_dict import *

def convert_joint_to_data(joint: JointStruct) -> JointData:
  '''
  Convert joint data to JointData structure
  Args:
    joint (JointStruct): joint structure
  Returns:
    JointData: JointData structure
  '''
  return  JointData(joint.origin, joint.select_mat, joint.dof, joint.select_indeces)

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
 
  state = cmtm_to_state_dict(state_cmtm[world_name], world_name)
  state_data.update(state)
  
  for joint in robot.joints:
    parent = robot.links[joint.parent_link_id]
    child = robot.links[joint.child_link_id]
    
    joint_data = convert_joint_to_data(joint)

    joint_motions = motions.joint_motions(joint.dof, joint.dof_index, order)

    joint_cmtm = joint_local_cmtm(joint_data, joint_motions, order)
    state = cmtm_to_state_dict(joint_cmtm, joint.name)
    state_data.update(state)

    p_link_cmtm = state_cmtm[parent.name]
    rel_cmtm = link_rel_cmtm(joint_data, joint_motions, order)

    link_cmtm = p_link_cmtm @ rel_cmtm
    # Update CMTM for the child link
    state_cmtm.update([(child.name, link_cmtm)])

    state = cmtm_to_state_dict(link_cmtm, child.name)
    state_data.update(state)

  return state_data

def __target_part_link_jacob(target_link : LinkStruct, joint : JointStruct, rel_frame : SE3) -> np.ndarray:
  if target_link.id == joint.child_link_id:
    mat = joint.select_mat
  else:
    joint_data = convert_joint_to_data(joint)
    mat = part_link_jacob(joint_data, rel_frame)  
  return mat

# specific 3d space (magic number 6)
def __target_part_link_cmtm_tan_jacob(target_link : LinkStruct, joint : JointStruct, rel_cmtm : CMTM, joint_cmtm : CMTM) -> np.ndarray:
  '''
  Compute the Jacobian matrix for the target part link in CMTM space.
  Args:
    target_link (LinkStruct): target link
    joint (JointStruct): joint structure
    rel_cmtm (CMTM): relative CMTM
    joint_cmtm (CMTM): joint CMTM
  '''
  mat = np.zeros((rel_cmtm._n * 6, rel_cmtm._n * joint.dof))
  if target_link.id == joint.child_link_id:
    tmp = joint_cmtm.tan_map()
    for i in range(rel_cmtm._n):
      mat[i*6:(i+1)*6, i*joint.dof:(i+1)*joint.dof] = (tmp[i*6:(i+1)*6, i*6:(i+1)*6])[:, joint.select_indeces]
  else:
    joint_data = convert_joint_to_data(joint)
    mat = part_link_cmtm_tan_jacob(joint_data, rel_cmtm, joint_cmtm)
  return mat

def __link_jacobian(robot, state : dict, target_link : LinkStruct) -> np.ndarray:
  jacob = np.zeros((6,robot.dof))
  link_route = []
  joint_route = []
  robot.route_target_link(target_link, link_route, joint_route)
  
  for j in joint_route:
    joint = robot.joints[j]
    rel_frame = state_dict_to_rel_frame(state, robot.links[joint.child_link_id].name, target_link.name)
    mat = __target_part_link_jacob(target_link, joint, rel_frame)
    jacob[:,joint.dof_index:joint.dof_index+joint.dof] = mat
    
  return jacob

def __link_cmtm_tan_jacobian(robot, state : dict, target_link : LinkStruct, order : int) -> np.ndarray:
  jacob = np.zeros((6*order,robot.dof*order))
  link_route = []
  joint_route = []
  robot.route_target_link(target_link, link_route, joint_route)
  
  for j in joint_route:
    joint = robot.joints[j]
    if joint.dof > 0:
      rel_cmtm = state_dict_to_rel_cmtm(state, robot.links[joint.child_link_id].name, target_link.name, order)

      joint_cmtm = state_dict_to_cmtm(state, joint.name, order)
      mat = __target_part_link_cmtm_tan_jacob(target_link, joint, rel_cmtm, joint_cmtm)

      for i in range(order):
        jacob[:,i*robot.dof+joint.dof_index:i*robot.dof+joint.dof_index+joint.dof]  \
          = mat[:,i*joint.dof:(i+1)*joint.dof]
  return jacob

# specific 3d space (magic number 6)
def link_jacobian(robot : RobotStruct, state : dict, link_name_list : list[str]) -> np.ndarray:
  links = robot.link_list(link_name_list)
  jacobs = np.zeros((6*len(links),robot.dof))
  for i in range(len(links)):
    jacobs[6*i:6*(i+1),:] = __link_jacobian(robot, state, links[i])
  return jacobs

# specific 3d space (magic number 6)
def link_cmtm_jacobian(robot : RobotStruct, state : dict, link_name_list : list[str], order = 3) -> np.ndarray:
  links = robot.link_list(link_name_list)
  jacobs = np.zeros((6*order*len(links),robot.dof*order))
  for i in range(len(links)):
    link_cmtm = state_dict_to_cmtm(state, link_name_list[i], order)
    jacobs[6*order*i:6*order*(i+1),:] \
      = link_cmtm.tan_map_inv() @ __link_cmtm_tan_jacobian(robot, state, links[i], order)
  
  return jacobs

def link_diff_kinematics_numerical(robot : RobotStruct, motions : RobotMotions, link_name_list : list[str],  data_type : str, order = None, \
                                    eps = 1e-8, update_method = None, update_direction = None) -> np.ndarray:
  if data_type not in ["pos", "rot", "vel", "acc", "jark", "frame", "cmtm"]:
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
  elif data_type == "jark":
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
      jacobs[(dof*order)*i:(6*order)*(i+1)] = numerical_grad(motion, kinematics_func, sub_func = CMTM.sub_vec)
    else:
      jacobs[dof*i:dof*(i+1)] = numerical_grad(motion, kinematics_func)

  return jacobs

# specific 3d space (magic number 6)
def dynamics(robot : RobotStruct, motions : RobotMotions) -> dict:
  state_data = {}
  
  state_data = kinematics(robot, motions)

  # world_name = robot.links[robot.joints[0].parent_link_id].name
  # state_data.update([(world_name + "_link_force" , [0.,0.,0.,0.,0.,0.])])

  for joint in reversed(robot.joints):
    child = robot.links[joint.child_link_id]
    joint_data = convert_joint_to_data(joint)
    
    joint_coord = motions.joint_coord(joint.dof, joint.dof_index)

    inertia = spatial_inertia(child.mass, child.inertia, child.cog)

    link_veloc = state_data[child.name + "_vel"]
    link_accel = state_data[child.name + "_acc"]
    
    link_force = link_dynamics(inertia, link_veloc, link_accel)  
    state_data.update([(child.name + "_link_force" , link_force.tolist())])
    
    rel_frame = link_rel_frame(joint_data, joint_coord)

    p_joint_force = np.zeros(6)
    for id in child.child_joint_ids:
      p_joint_force += state_data[robot.joints[id].name + "_joint_force"]

    joint_torque, joint_force = joint_dynamics(joint, rel_frame, p_joint_force, link_force)
    
    state_data.update([(joint.name + "_joint_force" , joint_force.tolist())])
    state_data.update([(joint.name + "_joint_torque" , joint_torque.tolist())])
    
  return state_data

def dynamics_cmtm(robot : RobotStruct, motions : RobotMotions) -> dict:
  state_data = {}
  
  state_data = kinematics(robot, motions)

  world_name = robot.links[robot.joints[0].parent_link_id].name
  state_data.update([(world_name + "_link_force" , [0.,0.,0.,0.,0.,0.])])

  for joint in reversed(robot.joints):
    child = robot.links[joint.child_link_id]
    joint_data = convert_joint_to_data(joint)
    
    joint_coord = motions.joint_coord(joint.dof, joint.dof_index)

    inertia = spatial_inertia(child.mass, child.inertia, child.cog)

    link_veloc = state_data[child.name + "_vel"]
    link_accel = state_data[child.name + "_acc"]
    
    link_force = link_dynamics(inertia, link_veloc, link_accel)  
    state_data.update([(child.name + "_link_force" , link_force.tolist())])
    
    rel_frame = link_rel_frame(joint_data, joint_coord)

    p_joint_force = np.zeros(6)
    for id in child.child_joint_ids:
      p_joint_force += state_data[robot.joints[id].name + "_joint_force"]

    joint_torque, joint_force = joint_dynamics(joint, rel_frame, p_joint_force, link_force)
    
    state_data.update([(joint.name + "_joint_force" , joint_force.tolist())])
    state_data.update([(joint.name + "_joint_torque" , joint_torque.tolist())])
    
  return state_data
