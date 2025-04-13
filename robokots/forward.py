#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki
# forward computation module from motion and robot_model to state

import numpy as np

from mathrobo import SE3, CMTM

from .robot_model import *
from .motion import *
from .state import *
from .kinematics import *
from .dynamics import *

def f_kinematics(robot, motions):
  state_data = {}
  
  world_name = robot.links[robot.joints[0].parent_link_id].name
  state_data.update([(world_name + "_pos" , [0.,0.,0.])])
  state_data.update([(world_name + "_rot" , [1.,0.,0.,0.,1.,0.,0.,0.,1.])])
  state_data.update([(world_name + "_vel" , [0.,0.,0.,0.,0.,0.])])
  state_data.update([(world_name + "_acc" , [0.,0.,0.,0.,0.,0.])])
  
  for joint in robot.joints:    
    parent = robot.links[joint.parent_link_id]
    child = robot.links[joint.child_link_id]
    
    joint_coord = motions.joint_coord(joint)
    joint_veloc = motions.joint_veloc(joint)
    joint_accel = motions.joint_accel(joint)
    
    rot = np.array(state_data[parent.name + "_rot"]).reshape((3,3))
    p_link_frame = SE3(rot, state_data[parent.name + "_pos"])
    p_link_vel = state_data[parent.name + "_vel"]  
    p_link_acc = state_data[parent.name + "_acc"]  
    
    p_link_cmtm = CMTM[SE3](p_link_frame, np.array((p_link_vel, p_link_acc)))
    rel_cmtm = link_rel_cmtm(joint, joint_coord, joint_veloc, joint_accel)
    
    link_cmtm = p_link_cmtm @ rel_cmtm
    
    frame = link_cmtm.mat()
    veloc = link_cmtm.elem_vecs(0)
    accel = link_cmtm.elem_vecs(1)     
    
    pos = frame[:3,3]
    rot_vec = frame[:3,:3].ravel()
    
    state_data.update([
        (child.name + "_pos" , pos.tolist()),
        (child.name + "_rot" , rot_vec.tolist()),
        (child.name + "_vel" , veloc.tolist()),
        (child.name + "_acc" , accel.tolist())
    ])
    
  return state_data

def __target_part_link_jacob(target_link, joint, rel_frame):
  if target_link.id == joint.child_link_id:
    mat = joint.origin.mat_inv_adj() @ joint.joint_select_mat
  else:
    mat = part_link_jacob(joint, rel_frame)  
  return mat

def __link_jacobian(robot, state, target_link):
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
  
def f_link_jacobian(robot, state, link_name_list):
  links = robot.link_list(link_name_list)
  jacobs = np.zeros((6*len(links),robot.dof))
  for i in range(len(links)):
    jacobs[6*i:6*(i+1),:] = __link_jacobian(robot, state, links[i])
  return jacobs

def f_dynamics(robot, motions):
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
