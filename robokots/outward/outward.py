#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki
# outward computation module from motion and robot_model to state

import numpy as np

from mathrobo import SO3, SE3, CMTM, numerical_difference, build_integrator

from ..basic.robot import RobotStruct
from ..basic.motion import RobotMotions
from ..basic.state_dict import state_dict_to_cmtm, extract_dict_link_info, vecs_to_state_dict, cmtm_to_state_list, state_dict_to_frame, state_dict_to_force_vecs
from ..basic.state import data_type_to_sub_func, data_type_dof

from ..kinematics.base import convert_joint_to_data, convert_link_to_data
from ..kinematics.kinematics import joint_local_cmtm, joint_rel_cmtm, joint_rel_frame
from ..kinematics.kinematics_soft_link import soft_link_local_cmtm, calc_link_local_point_frame

from ..dynamics.base import spatial_inertia
from ..dynamics.dynamics import link_dynamics, joint_dynamics, link_momentum_cmtm, link_force_cmtm, link_dynamics_cmtm, joint_dynamics_cmtm

def kinematics(robot : RobotStruct, motions, order = 3) -> dict:
  '''
  Forward kinematics computation
  Args:
    robot (RobotStruct): robot model
    motions : robot motion
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
    
    joint_data = convert_joint_to_data(joint)
    link_data = convert_link_to_data(child)

    joint_motions = motions[joint.dof_index*order:joint.dof_index*order+joint.dof*order]
    link_motions = motions[child.dof_index*order:child.dof_index*order+child.dof*order]

    p_link_cmtm = state_cmtm[parent.name]
    joint_cmtm = joint_rel_cmtm(joint_data, joint_motions, order)
    link_cmtm = soft_link_local_cmtm(link_data, link_motions, order)

    link_cmtm = p_link_cmtm @ joint_cmtm @ link_cmtm
    # Update CMTM for the child link
    state_cmtm.update([(child.name, link_cmtm)])

    state = cmtm_to_state_list(link_cmtm, child.name)
    state_data.update(state)
    
    #---for pre-computation
    joint_cmtm = joint_local_cmtm(joint_data, joint_motions, order)
    state = cmtm_to_state_list(joint_cmtm, joint.name)
    state_data.update(state)

  return state_data

def calc_link_total_point_frame(robot : RobotStruct, motions : RobotMotions, state : dict, point : float) -> SE3:
  base = 0.0
  p_link = robot.links[0]
  for l in robot.links:
      if point > base + l.length:
          base += l.length
          p_link = l
          continue
      p_link_frame = state_dict_to_frame(state, p_link.name)
      coord = motions.link_motions(l.dof, l.dof_index, 1)[0]
      return calc_link_local_point_frame(l, coord, p_link_frame, point - base)
  
def link_diff_kinematics_numerical(robot : RobotStruct, motions, link_name_list : list[str],  data_type : str, order = 3, \
                                    eps = 1e-8, update_method = None, update_direction = None) -> np.ndarray:
  if data_type not in ["pos", "rot", "vel", "acc", "jerk", "snap", "frame", "cmtm"]:
    raise ValueError(f"Invalid data_type: {data_type}. Must be 'pos', 'rot', 'vel', 'acc', 'frame' or 'cmtm'.")

  dof = data_type_dof(data_type, order, dim=3)

  diff = np.zeros((len(link_name_list), dof))

  def update_func(x_init, direct, eps):
    x_ = x_init.copy()
    
    if update_method is None:
      D, d = build_integrator(1, order, eps, method="poly")
    else:
      D, d = build_integrator(1, order, eps, method=update_method)

    x_ = np.kron(np.eye(robot.dof), D) @ x_init + np.kron(np.eye(robot.dof), d) @ direct
    return x_

  for i in range(len(link_name_list)):
    def kinematics_func(x):
      state = kinematics(robot, x, order)
      y = extract_dict_link_info(state, data_type, link_name_list[i])
      return y
    
    sub_func = data_type_to_sub_func(data_type)
    # diff[i] = numerical_difference(motions.motions, kinematics_func, sub_func = sub_func, update_func = update_func, direction = update_direction, eps=eps)
    diff[i] = numerical_difference(motions, kinematics_func, sub_func = sub_func, update_func = update_func, direction = update_direction, eps=eps)

  return diff

# specific 3d space (magic number 6)
def dynamics(robot : RobotStruct, joint_motions) -> dict:
  state_data = {}
  
  state_data = kinematics(robot, joint_motions, 3)

  world_name = robot.links[robot.joints[0].parent_link_id].name
  state_data.update([(world_name + "_link_force" , [0.,0.,0.,0.,0.,0.])])

  for joint in reversed(robot.joints):
    child = robot.links[joint.child_link_id]
    joint_data = convert_joint_to_data(joint)
    
    joint_coord = joint_motions[joint.dof_index:joint.dof_index+joint.dof]

    inertia = spatial_inertia(child.mass, child.inertia, child.cog)

    link_veloc = np.array(state_data[child.name + "_vel"])
    link_accel = np.array(state_data[child.name + "_acc"])
    
    link_force = link_dynamics(inertia, link_veloc, link_accel)  
    state_data.update([(child.name + "_link_force" , link_force.tolist())])
    
    joint_frame = joint_rel_frame(joint_data, joint_coord)

    p_joint_force = np.zeros(6)
    for id in child.child_joint_ids:
      p_joint_force += state_data[robot.joints[id].name + "_joint_force"]

    joint_torque, joint_force = joint_dynamics(joint.select_mat, joint_frame, p_joint_force, link_force)
    
    state_data.update([(joint.name + "_joint_force" , joint_force.tolist())])
    state_data.update([(joint.name + "_joint_torque" , joint_torque.tolist())])
    
  return state_data

def dynamics_cmtm(robot : RobotStruct, joint_motions, dynamics_order = 1) -> dict:
  state_data = {}
  
  state_data = kinematics(robot, joint_motions, dynamics_order + 2)

  world_name = robot.links[robot.joints[0].parent_link_id].name
  state_data.update([(world_name + "_link_force" , [0.,0.,0.,0.,0.,0.])])

  for joint in reversed(robot.joints):
    child = robot.links[joint.child_link_id]
    joint_data = convert_joint_to_data(joint)

    joint_motion = joint_motions[joint.dof_index*(dynamics_order+2):joint.dof_index*(dynamics_order+2) + joint.dof*(dynamics_order+2)]
    inertia = spatial_inertia(child.mass, child.inertia, child.cog)

    link_cmtm = state_dict_to_cmtm(state_data, child.name, dynamics_order + 2)

    link_momentums = link_momentum_cmtm(inertia, link_cmtm.vecs())
    state = vecs_to_state_dict(link_momentums, child.name, "link_momentum", dynamics_order+1)
    state_data.update(state)
    
    link_forces = link_force_cmtm(link_cmtm.vecs(-1), link_momentums)
    state = vecs_to_state_dict(link_forces, child.name, "link_force", dynamics_order)
    state_data.update(state)
    
    joint_cmtm = joint_rel_cmtm(joint_data, joint_motion, dynamics_order)

    p_joint_force = np.zeros(6*dynamics_order)
    for id in child.child_joint_ids:
      p_joint_force += state_dict_to_force_vecs(state_data, robot.joints[id].name, "joint_force")

    joint_torques, joint_forces = joint_dynamics_cmtm(joint, joint_cmtm, p_joint_force, link_forces)

    state = vecs_to_state_dict(joint_forces, joint.name, "joint_force", dynamics_order)
    state_data.update(state)
    state = vecs_to_state_dict(joint_torques, joint.name, "joint_torque", dynamics_order)
    state_data.update(state)
    
  return state_data
