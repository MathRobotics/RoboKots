#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki

import numpy as np

class RobotMotions:
  motions : np.ndarray = np.array([])

  def __init__(self, robot, aliases_ = ["coord", "veloc", "accel", "force"]):
    self.aliases = aliases_
    self.dof = robot.dof
    self.motion_num = len(self.aliases) 
    self.motions = np.zeros(self.dof * self.motion_num)
    
  def set_motion(self, vecs):
    self.motions = vecs
    
  def motion_index(self, name):
    for i in range(len(self.aliases)):
      if name == self.aliases[i]:
        return i
    return None
  
  def gen_value(self, joint, name):
    m_index = self.motion_index(name)
    offset = self.dof * m_index + joint.dof_index
    return self.motions[offset : offset + joint.dof]
  
  def joint_coord(self, joint):
    return self.gen_value(joint, "coord")
  
  def joint_veloc(self, joint):
    return self.gen_value(joint, "veloc")
  
  def joint_accel(self, joint):
    return self.gen_value(joint, "accel")
  
  def joint_force(self, joint):
    return self.gen_value(joint, "force")

  def link_coord(self, link):
    return self.gen_value(link, "coord")
  
  def link_veloc(self, link):
    return self.gen_value(link, "veloc")
  
  def link_accel(self, link):
    return self.gen_value(link, "accel")
  
  def link_force(self, link):
    return self.gen_value(link, "force")