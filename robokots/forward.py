#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki
# forward computation module from motion and robot_model to state

import numpy as np

from mathrobo import SE3, CMTM

from .robot import RobotStruct, LinkStruct, JointStruct
from .motion import RobotMotions
from .state import RobotState
from .kinematics import *
from .dynamics import *

#specific 3d-CMTM
def cmtm_to_state(cmtm : CMTM, name : str, order = 3) -> dict:
  '''
  Convert CMTM to state data
  Args:
    cmtm (CMTM): CMTM object
    name (str): name of the link
  Returns:
    dict: state data
  '''
  if order < 1 and order > 3:
    raise ValueError("order must be 1, 2 or 3")

  state = []

  if 1:
    mat = cmtm.elem_mat()
    pos = mat[:3,3]
    rot_vec = mat[:3,:3].ravel()
    state.append((name+"_pos" , pos.tolist()))
    state.append((name+"_rot" , rot_vec.tolist()))
    if order > 1:
      veloc = cmtm.elem_vecs(0)
      state.append((name+"_vel" , veloc.tolist()))
    if order > 2:
      accel = cmtm.elem_vecs(1)
      state.append((name+"_acc" , accel.tolist()))
  else:
    mat = cmtm.elem_mat()
    veloc = cmtm.elem_vecs(0)
    accel = cmtm.elem_vecs(1)
    
    pos = mat[:3,3]
    rot_vec = mat[:3,:3].ravel()

    state = [
        (name+"_pos" , pos.tolist()),
        (name+"_rot" , rot_vec.tolist()),
        (name+"_vel" , veloc.tolist()),
        (name+"_acc" , accel.tolist())
    ]
    
  return state

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
 
  state = cmtm_to_state(state_cmtm[world_name], world_name, order)
  state_data.update(state)
  
  for joint in robot.joints:    
    parent = robot.links[joint.parent_link_id]
    child = robot.links[joint.child_link_id]

    joint_motions = motions.joint_motions(joint)

    p_link_cmtm = state_cmtm[parent.name]
    rel_cmtm = link_rel_cmtm(joint, joint_motions, order)

    link_cmtm = p_link_cmtm @ rel_cmtm
    # Update CMTM for the child link
    state_cmtm.update([(child.name, link_cmtm)])

    state = cmtm_to_state(link_cmtm, child.name, order)
    state_data.update(state)
    
  return state_data

def __target_part_link_jacob(target_link : LinkStruct, joint : JointStruct, rel_frame : SE3) -> np.ndarray:
  if target_link.id == joint.child_link_id:
    mat = joint.select_mat
  else:
    mat = part_link_jacob(joint, rel_frame)  
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
  
def f_link_jacobian(robot : RobotStruct, state : RobotState, link_name_list : list[str]) -> np.ndarray:
  links = robot.link_list(link_name_list)
  jacobs = np.zeros((6*len(links),robot.dof))
  for i in range(len(links)):
    jacobs[6*i:6*(i+1),:] = __link_jacobian(robot, state, links[i])
  return jacobs

# specific 3d space (magic number 6)
def __target_part_link_cmtm_jacob(target_link : LinkStruct, joint : JointStruct, rel_cmtm : CMTM) -> np.ndarray:
  mat = np.zeros((rel_cmtm._n * 6, rel_cmtm._n * joint.dof))
  if target_link.id == joint.child_link_id:
    tmp = np.eye(rel_cmtm.mat_adj().shape[0], rel_cmtm.mat_adj().shape[1])
    for i in range(rel_cmtm._n):
      mat[i*6:(i+1)*6, i*joint.dof:(i+1)*joint.dof] = joint.selector(tmp[i*6:(i+1)*6, i*6:(i+1)*6])
  else:
    mat = part_link_cmtm_jacob(joint, rel_cmtm)
  return mat

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
      mat = __target_part_link_cmtm_jacob(target_link, joint, rel_cmtm)
      for i in range(order):
        jacob[:,i*robot.dof+joint.dof_index:i*robot.dof+joint.dof_index+joint.dof]  \
          = mat[:,i*joint.dof:(i+1)*joint.dof]
  return jacob

# specific 3d space (magic number 6)
def f_link_cmtm_jacobian(robot : RobotStruct, state : RobotState, link_name_list : list[str], order = 3) -> np.ndarray:
  links = robot.link_list(link_name_list)
  jacobs = np.zeros((6*order*len(links),robot.dof*order))
  for i in range(len(links)):
    jacobs[6*order*i:6*order*(i+1),:] = __link_cmtm_jacobian(robot, state, links[i], order)
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
