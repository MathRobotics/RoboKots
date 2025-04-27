#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki

import numpy as np

from .robot import RobotStruct, JointStruct, LinkStruct

class RobotMotions:
  motions : np.ndarray = np.array([])
  ALLOWED_ALIASES = frozenset(["coord", "veloc", "accel", "accel_diff1", "accel_diff2", "accel_diff3"])

  def __init__(self, robot : RobotStruct, aliases_ = ["coord", "veloc", "accel"]):
    if not set(aliases_).issubset(self.ALLOWED_ALIASES):
      raise ValueError(f"Invalid alias: {set(aliases_) - self.ALLOWED_ALIASES}")
    self.aliases = aliases_
    self.dof = robot.dof
    self.motion_num = len(self.aliases) 
    self.motions = np.zeros(self.dof * self.motion_num)
    
  def set_aliases(self, aliases_ = ["coord", "veloc", "accel"]):
    if not set(aliases_).issubset(self.ALLOWED_ALIASES):
      raise ValueError(f"Invalid alias: {set(aliases_) - self.ALLOWED_ALIASES}")
    self.aliases = aliases_
    
  def set_motion(self, vecs):
    self.motions = vecs
    
  def motion_index(self, name : str) -> int:
    if name not in self.aliases:
      raise ValueError(f"Invalid alias: {name}")
    for i in range(len(self.aliases)):
      if name == self.aliases[i]:
        return i
  
  def gen_values(self, name : str):
    m_index = self.motion_index(name)
    offset = self.dof * m_index
    return self.motions[offset : offset + self.dof]

  def coord(self):
    return self.gen_values("coord")

  def veloc(self):
    return self.gen_values("veloc")
    
  def accel(self):
    return self.gen_values("accel")
    
  def gen_value(self, joint : JointStruct, name : str):
    m_index = self.motion_index(name)
    offset = self.dof * m_index + joint.dof_index
    return self.motions[offset : offset + joint.dof]
  
  def joint_coord(self, joint : JointStruct):
    return self.gen_value(joint, "coord")
  
  def joint_veloc(self, joint : JointStruct):
    return self.gen_value(joint, "veloc")
  
  def joint_accel(self, joint : JointStruct):
    return self.gen_value(joint, "accel")

  def link_coord(self, link : LinkStruct):
    return self.gen_value(link, "coord")
  
  def link_veloc(self, link : LinkStruct):
    return self.gen_value(link, "veloc")
  
  def link_accel(self, link : LinkStruct):
    return self.gen_value(link, "accel")

  def joint_motions(self, joint : JointStruct):
    values = ()
    for a in self.aliases:
      values += (self.gen_value(joint, a),)
    return np.array(values)
