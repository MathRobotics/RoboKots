#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki
# outward computation module from motion and robot_model to state

import numpy as np

from mathrobo import SO3, SE3, CMTM, numerical_grad, numerical_difference, build_integrator

from ..basic.robot import RobotStruct, LinkStruct, JointStruct
from ..basic.motion import RobotMotions
from ..kinematics.kinematics import *
from ..dynamics import *
from ..basic.state_dict import *

from .base import convert_joint_to_data

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
    
    joint_data = convert_joint_to_data(joint)

    joint_motions = motions.joint_motions(joint.dof, joint.dof_index, order)

    joint_cmtm = joint_local_cmtm(joint_data, joint_motions, order)
    state = cmtm_to_state_list(joint_cmtm, joint.name)
    state_data.update(state)

    p_link_cmtm = state_cmtm[parent.name]
    rel_cmtm = link_rel_cmtm(joint_data, joint_motions, order)

    link_cmtm = p_link_cmtm @ rel_cmtm
    # Update CMTM for the child link
    state_cmtm.update([(child.name, link_cmtm)])

    state = cmtm_to_state_list(link_cmtm, child.name)
    state_data.update(state)

  return state_data

# specific 3d space (magic number 6)
def dynamics(robot : RobotStruct, motions : RobotMotions) -> dict:
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

def dynamics_cmtm(robot : RobotStruct, motions : RobotMotions, dynamics_order = 1) -> dict:
  state_data = {}
  
  state_data = kinematics(robot, motions, dynamics_order + 2)

  world_name = robot.links[robot.joints[0].parent_link_id].name
  state_data.update([(world_name + "_link_force" , [0.,0.,0.,0.,0.,0.])])

  for joint in reversed(robot.joints):
    child = robot.links[joint.child_link_id]
    joint_data = convert_joint_to_data(joint)
    
    joint_motion = motions.joint_motions(joint.dof, joint.dof_index, dynamics_order + 2)  
  
    inertia = spatial_inertia(child.mass, child.inertia, child.cog)

    link_cmtm = state_dict_to_cmtm(state_data, child.name, dynamics_order + 2)
    
    link_forces = link_dynamics_cmtm(inertia, link_cmtm.vecs())  

    state = vecs_to_state_dict(link_forces, child.name, "link_force", dynamics_order)
    state_data.update(state)
    
    rel_cmtm = link_rel_cmtm(joint_data, joint_motion, dynamics_order)

    p_joint_force = np.zeros(6*dynamics_order)
    for id in child.child_joint_ids:
      p_joint_force += state_dict_to_force_vecs(state_data, robot.joints[id].name, "joint_force")

    joint_torques, joint_forces = joint_dynamics_cmtm(joint, rel_cmtm, p_joint_force, link_forces)

    state = vecs_to_state_dict(joint_forces, joint.name, "joint_force", dynamics_order)
    state_data.update(state)
    state = vecs_to_state_dict(joint_torques, joint.name, "joint_torque", dynamics_order)
    state_data.update(state)
    
  return state_data
