#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki
# forward computation module from motion and robot_model to state

import numpy as np
from .robot_model import *
from .motion import *
from .state import *
from .kinematics import *

from mathrobo import *

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

    frame = kinematics(joint, p_link_frame, joint_coord)  
    veloc = vel_kinematics(joint, p_link_vel, joint_coord, joint_veloc)  
    accel = acc_kinematics(joint, p_link_vel, p_link_acc, joint_coord, joint_veloc, joint_accel)       
    
    pos = frame.pos()
    rot_vec = frame.rot().ravel()

    state_data.update([(child.name + "_pos" , pos.tolist())])
    state_data.update([(child.name + "_rot" , rot_vec.tolist())])
    state_data.update([(child.name + "_vel" , veloc.tolist())])
    state_data.update([(child.name + "_acc" , accel.tolist())])
    
  return state_data

def __link_jacobian(robot, state, target_link):
  jacob = np.zeros((6,robot.dof))
  link_route = []
  joint_route = []
  robot.route_target_link(target_link, link_route, joint_route)
  
  for j in joint_route:
    joint = robot.joints[j]
    mat = target_part_link_jacob(target_link, joint, state.link_rel_frame(robot.links[joint.parent_link_id], target_link))
    jacob[:,joint.dof_index:joint.dof_index+joint.dof] = mat
    
  return jacob
  
def f_link_jacobian(robot, state, link_name_list):
  links = robot.link_list(link_name_list)
  jacobs = np.zeros((6*len(links),robot.dof))
  for i in range(len(links)):
    jacobs[6*i:6*(i+1),:] = __link_jacobian(robot, state, links[i])
  return jacobs