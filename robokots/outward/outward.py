#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki
# outward computation module from motion and robot_model to state

import numpy as np

from mathrobo import CMVector
from mathrobo import SO3, SE3, CMTM, SE3wrench, numerical_difference, build_integrator

from ..basic.robot import RobotStruct
from ..basic.motion import RobotMotions
from ..basic.state_dict import state_dict_to_cmtm, extract_dict_link_info, vecs_to_state_dict, cmtm_to_state_list, state_dict_to_frame, state_dict_to_vecs
from ..basic.state import data_type_to_sub_func, data_type_dof

from ..kinematics.base import convert_joint_to_data, convert_link_to_data
from ..kinematics.kinematics import joint_local_cmtm, joint_rel_cmtm, joint_rel_frame
from ..kinematics.kinematics_matrix import joint_select_diag_mat
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
  
  state_dict = {}
  cmtm_dict = {}

  # Initialize CMTM for the world link
  # The world link is the parent of the first joint
  world_name = robot.links[robot.joints[0].parent_link_id].name
  cmtm_dict.update([(world_name, CMTM.eye(SE3, order))])

  state = cmtm_to_state_list(cmtm_dict[world_name], world_name)
  state_dict.update(state)

  for joint in robot.joints:
    parent = robot.links[joint.parent_link_id]
    child = robot.links[joint.child_link_id]
    
    joint_data = convert_joint_to_data(joint)
    link_data = convert_link_to_data(child)

    joint_motions = motions[joint.dof_index*order:joint.dof_index*order+joint.dof*order]
    link_motions = motions[child.dof_index*order:child.dof_index*order+child.dof*order]

    p_link_cmtm = cmtm_dict[parent.name]
    joint_cmtm = joint_rel_cmtm(joint_data, joint_motions, order)
    link_cmtm = soft_link_local_cmtm(link_data, link_motions, order)

    link_cmtm = p_link_cmtm @ joint_cmtm @ link_cmtm
    # Update CMTM for the child link
    cmtm_dict.update([(child.name, link_cmtm)])

    state = cmtm_to_state_list(link_cmtm, child.name)
    state_dict.update(state)
    
    #---for pre-computation
    joint_cmtm = joint_local_cmtm(joint_data, joint_motions, order)
    state = cmtm_to_state_list(joint_cmtm, joint.name)
    state_dict.update(state)

  return state_dict

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
  state_dict = kinematics(robot, joint_motions, 3)

  world_name = robot.links[robot.joints[0].parent_link_id].name
  state_dict.update([(world_name + "_link_force" , [0.,0.,0.,0.,0.,0.])])

  for joint in reversed(robot.joints):
    child = robot.links[joint.child_link_id]
    joint_data = convert_joint_to_data(joint)
    
    joint_coord = joint_motions[joint.dof_index:joint.dof_index+joint.dof]

    inertia = spatial_inertia(child.mass, child.inertia, child.cog)

    link_veloc = np.array(state_dict[child.name + "_vel"])
    link_accel = np.array(state_dict[child.name + "_acc"])
    
    link_force = link_dynamics(inertia, link_veloc, link_accel)  
    state_dict.update([(child.name + "_link_force" , link_force.tolist())])
    
    joint_frame = joint_rel_frame(joint_data, joint_coord)

    p_joint_force = np.zeros(6)
    for id in child.child_joint_ids:
      p_joint_force += state_dict[robot.joints[id].name + "_joint_force"]

    joint_torque, joint_force = joint_dynamics(joint.select_mat, joint_frame, p_joint_force, link_force)
    
    state_dict.update([(joint.name + "_joint_force" , joint_force.tolist())])
    state_dict.update([(joint.name + "_joint_torque" , joint_torque.tolist())])
    
  return state_dict

def dynamics_cmtm(robot : RobotStruct, motions, dynamics_order = 1) -> dict:
  state_dict = kinematics(robot, motions, dynamics_order + 2)
  momentum_dict = {}

  for joint in reversed(robot.joints):
    child = robot.links[joint.child_link_id]
    child_joint_ids = child.child_joint_ids

    motion = motions[joint.dof_index*(dynamics_order+2):joint.dof_index*(dynamics_order+2) + joint.dof*(dynamics_order+2)]
    inertia = spatial_inertia(child.mass, child.inertia, child.cog)

    link_cmtm = state_dict_to_cmtm(state_dict, child.name, dynamics_order + 2)

    link_momentum = link_momentum_cmtm(inertia, link_cmtm.cmvecs())
    state = vecs_to_state_dict(link_momentum.vecs(), child.name, "link_momentum", dynamics_order+1)
    state_dict.update(state)
    
    link_force = link_force_cmtm(link_cmtm.cmvecs(), link_momentum)
    state = vecs_to_state_dict(link_force.vecs(), child.name, "link_force", dynamics_order)
    state_dict.update(state)

    joint_momentums = link_momentum.vec()
    for c_id in child_joint_ids:
      c_joint = robot.joints[c_id]
      c_joint_data = convert_joint_to_data(c_joint)

      c_joint_cmtm = joint_rel_cmtm(c_joint_data, motion, dynamics_order + 1)
      c_joint_cmtm_wrench = CMTM.change_elemclass(c_joint_cmtm, SE3wrench)

      c_joint_momentums = state_dict_to_vecs(momentum_dict, c_joint.name, "joint_momentums")

      joint_momentums += c_joint_cmtm_wrench.mat_adj() @ c_joint_momentums

    momentum_dict.update([(joint.name+'_joint_momentums', joint_momentums)])

    state = vecs_to_state_dict(joint_momentums, joint.name, "joint_momentum", dynamics_order+1)
    state_dict.update(state)

    link_cmtm = state_dict_to_cmtm(state_dict, child.name, dynamics_order + 2)
    joint_momentum = CMVector(joint_momentums.reshape(-1,6))
    joint_force = link_force_cmtm(link_cmtm.cmvecs(), joint_momentum)
    state = vecs_to_state_dict(joint_force.vecs(), joint.name, "joint_force", dynamics_order)
    state_dict.update(state)

    if joint.dof == 0:
      continue
    joint_torque = joint_select_diag_mat(joint.select_mat, dynamics_order).T @ link_force.vec()
    state = vecs_to_state_dict(joint_torque.reshape(-1, joint.dof), joint.name, "joint_torque", dynamics_order)
    state_dict.update(state)

  # Compute for the world link
  world_link = robot.links[robot.joints[0].parent_link_id]
  inertia = spatial_inertia(world_link.mass, world_link.inertia, world_link.cog)
  link_cmtm = state_dict_to_cmtm(state_dict, world_link.name, dynamics_order + 2)

  link_vel = CMVector(link_cmtm.vecs())
  link_momentum = link_momentum_cmtm(inertia, link_vel)
  state = vecs_to_state_dict(link_momentum.vecs(), world_link.name, "link_momentum", dynamics_order+1)
  state_dict.update(state)

  link_force = link_force_cmtm(link_vel, link_momentum)
  state = vecs_to_state_dict(link_force.vecs(), world_link.name, "link_force", dynamics_order)
  state_dict.update(state)
    
  return state_dict
