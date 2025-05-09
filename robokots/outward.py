#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki
# outward computation module from motion and robot_model to state

import numpy as np

from mathrobo import SE3, CMTM, numerical_grad

from .robot import RobotStruct, LinkStruct, JointStruct
from .motion import RobotMotions
from .state import RobotState
from .kinematics import *
from .dynamics import *
from .state_dict import *

#specific 3d-CMTM
def f_kinematics(robot : RobotStruct, motions : RobotMotions, order = 3) -> dict:
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
  state_cmtm.update([(world_name, CMTM.eye(SE3, order-1))])
 
  state = cmtm_to_state_dict(state_cmtm[world_name], world_name, order)
  state_data.update(state)
  
  for joint in robot.joints:
    parent = robot.links[joint.parent_link_id]
    child = robot.links[joint.child_link_id]

    joint_motions = motions.joint_motions(joint, order)

    joint_cmtm = joint_local_cmtm(joint, joint_motions, order)
    state = cmtm_to_state_dict(joint_cmtm, joint.name, order)
    state_data.update(state)

    p_link_cmtm = state_cmtm[parent.name]
    rel_cmtm = link_rel_cmtm(joint, joint_motions, order)

    link_cmtm = p_link_cmtm @ rel_cmtm
    # Update CMTM for the child link
    state_cmtm.update([(child.name, link_cmtm)])

    state = cmtm_to_state_dict(link_cmtm, child.name, order)
    state_data.update(state)

  return state_data

def __target_part_link_jacob(target_link : LinkStruct, joint : JointStruct, rel_frame : SE3) -> np.ndarray:
  if target_link.id == joint.child_link_id:
    mat = joint.select_mat
  else:
    mat = part_link_jacob(joint, rel_frame)  
  return mat

# specific 3d space (magic number 6)
def __target_part_link_cmtm_jacob(target_link : LinkStruct, joint : JointStruct, rel_cmtm : CMTM, joint_cmtm : CMTM) -> np.ndarray:
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
    tmp = joint_cmtm.tan_mat_adj()
    for i in range(rel_cmtm._n):
      mat[i*6:(i+1)*6, i*joint.dof:(i+1)*joint.dof] = joint.selector(tmp[i*6:(i+1)*6, i*6:(i+1)*6])
  else:
    mat = part_link_cmtm_jacob(joint, rel_cmtm, joint_cmtm)
  return mat

def __link_jacobian(robot, state : RobotState, target_link : LinkStruct) -> np.ndarray:
  jacob = np.zeros((6,robot.dof))
  link_route = []
  joint_route = []
  robot.route_target_link(target_link, link_route, joint_route)
  
  for j in joint_route:
    joint = robot.joints[j]
    rel_frame = state.link_rel_frame(robot.links[joint.child_link_id].name, target_link.name)
    mat = __target_part_link_jacob(target_link, joint, rel_frame)
    jacob[:,joint.dof_index:joint.dof_index+joint.dof] = mat
    
  return jacob

# specific 3d space (magic number 6)
def __link_cmtm_jacobian(robot, state : RobotState, target_link : LinkStruct, order : int) -> np.ndarray:
  jacob = np.zeros((6*order,robot.dof*order))
  link_route = []
  joint_route = []
  robot.route_target_link(target_link, link_route, joint_route)
  
  for j in joint_route:
    joint = robot.joints[j]
    if joint.dof > 0:
      rel_cmtm = state.link_rel_cmtm(robot.links[joint.child_link_id].name, target_link.name, order)

      joint_cmtm = state.joint_cmtm(joint.name, order)
      mat = __target_part_link_cmtm_jacob(target_link, joint, rel_cmtm, joint_cmtm)

      for i in range(order):
        jacob[:,i*robot.dof+joint.dof_index:i*robot.dof+joint.dof_index+joint.dof]  \
          = mat[:,i*joint.dof:(i+1)*joint.dof]
  return jacob

def f_link_jacobian(robot : RobotStruct, state : RobotState, link_name_list : list[str]) -> np.ndarray:
  links = robot.link_list(link_name_list)
  jacobs = np.zeros((6*len(links),robot.dof))
  for i in range(len(links)):
    jacobs[6*i:6*(i+1),:] = __link_jacobian(robot, state, links[i])
  return jacobs

# specific 3d space (magic number 6)
def f_link_cmtm_jacobian(robot : RobotStruct, state : RobotState, link_name_list : list[str], order = 3) -> np.ndarray:
  links = robot.link_list(link_name_list)
  jacobs = np.zeros((6*order*len(links),robot.dof*order))
  for i in range(len(links)):
    link_cmtm = state.link_cmtm(link_name_list[i], order)
    jacobs[6*order*i:6*order*(i+1),:] \
      = link_cmtm.tan_mat_inv_adj() @ __link_cmtm_jacobian(robot, state, links[i], order)
   
  return jacobs

#specific 3d-CMTM
def f_link_jacobian_numerical(robot : RobotStruct, motions : RobotMotions, link_name_list : list[str], data_type : str) -> np.ndarray:
  if data_type not in ["pos", "rot", "vel", "acc", "frame", "cmtm"]:
    raise ValueError(f"Invalid data_type: {data_type}. Must be 'pos', 'rot', 'vel', 'acc', 'frame' or 'cmtm'.")

  order = 3
  if data_type == "pos" or data_type == "rot" or data_type == "frame":
    order = 1
  elif data_type == "vel":
    order = 2
  elif data_type == "acc":
    order = 3
  elif data_type == "cmtm":
    order = 3

  jacobs = np.zeros((6*len(link_name_list),robot.dof*order))
  motion = motions.motions[:robot.dof*order]

  for i in range(len(link_name_list)):
    def f_kinematics_func(x):
      motions.motions = x
      state = f_kinematics(robot, motions, order)
      y = extract_dict_link_info(state, data_type, link_name_list[i])
      return y

    if data_type == "frame":
      jacobs[6*i:6*(i+1)] = numerical_grad(motion, f_kinematics_func, sub_func = SE3.sub_tan_vec)
    else:
      jacobs[6*i:6*(i+1)] = numerical_grad(motion, f_kinematics_func)

  return jacobs

# specific 3d space (magic number 6)
def f_dynamics(robot : RobotStruct, motions : RobotMotions) -> dict:
  state_data = {}
  
  state_data = f_kinematics(robot, motions)

  # world_name = robot.links[robot.joints[0].parent_link_id].name
  # state_data.update([(world_name + "_link_force" , [0.,0.,0.,0.,0.,0.])])

  for joint in reversed(robot.joints):
    child = robot.links[joint.child_link_id]
    
    joint_coord = motions.joint_coord(joint)

    inertia = spatial_inertia(child.mass, child.inertia, child.cog)

    link_veloc = state_data[child.name + "_vel"]
    link_accel = state_data[child.name + "_acc"]
    
    link_force = link_dynamics(inertia, link_veloc, link_accel)  
    state_data.update([(child.name + "_link_force" , link_force.tolist())])
    
    rel_frame = link_rel_frame(joint, joint_coord)

    p_joint_force = np.zeros(6)
    for id in child.child_joint_ids:
      p_joint_force += state_data[robot.joints[id].name + "_joint_force"]

    joint_torque, joint_force = joint_dynamics(joint, rel_frame, p_joint_force, link_force)
    
    state_data.update([(joint.name + "_joint_force" , joint_force.tolist())])
    state_data.update([(joint.name + "_joint_torque" , joint_torque.tolist())])
    
  return state_data

def f_dynamics_cmtm(robot : RobotStruct, motions : RobotMotions) -> dict:
  state_data = {}
  
  state_data = f_kinematics(robot, motions)

  world_name = robot.links[robot.joints[0].parent_link_id].name
  state_data.update([(world_name + "_link_force" , [0.,0.,0.,0.,0.,0.])])

  for joint in reversed(robot.joints):
    child = robot.links[joint.child_link_id]
    
    joint_coord = motions.joint_coord(joint)

    inertia = spatial_inertia(child.mass, child.inertia, child.cog)

    link_veloc = state_data[child.name + "_vel"]
    link_accel = state_data[child.name + "_acc"]
    
    link_force = link_dynamics(inertia, link_veloc, link_accel)  
    state_data.update([(child.name + "_link_force" , link_force.tolist())])
    
    rel_frame = link_rel_frame(joint, joint_coord)

    p_joint_force = np.zeros(6)
    for id in child.child_joint_ids:
      p_joint_force += state_data[robot.joints[id].name + "_joint_force"]

    joint_torque, joint_force = joint_dynamics(joint, rel_frame, p_joint_force, link_force)
    
    state_data.update([(joint.name + "_joint_force" , joint_force.tolist())])
    state_data.update([(joint.name + "_joint_torque" , joint_torque.tolist())])
    
  return state_data
