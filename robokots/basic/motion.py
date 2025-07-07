#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki

import numpy as np

class RobotMotions:
  motions : np.ndarray = np.array([])
  ALLOWED_ALIASES = frozenset(["coord", "veloc", "accel", "accel_diff1", "accel_diff2", "accel_diff3"])

  def __init__(self, robot_dof : int, aliases_ = ["coord", "veloc", "accel"]):
    if not set(aliases_).issubset(self.ALLOWED_ALIASES):
      raise ValueError(f"Invalid alias: {set(aliases_) - self.ALLOWED_ALIASES}")
    self.aliases = aliases_
    self.dof = robot_dof
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
    
  def gen_value(self, dof : int, dof_index : int, name : str):
    m_index = self.motion_index(name)
    offset = self.dof * m_index + dof_index
    return self.motions[offset : offset + dof]
  
  def joint_coord(self, dof : int, dof_index : int):
    return self.gen_value(dof, dof_index, "coord")
  
  def joint_veloc(self, dof : int, dof_index : int):
    return self.gen_value(dof, dof_index, "veloc")
  
  def joint_accel(self, dof : int, dof_index : int):
    return self.gen_value(dof, dof_index, "accel")

  def link_coord(self, dof : int, dof_index : int):
    return self.gen_value(dof, dof_index, "coord")
  
  def link_veloc(self, dof : int, dof_index : int):
    return self.gen_value(dof, dof_index, "veloc")
  
  def link_accel(self, dof : int, dof_index : int):
    return self.gen_value(dof, dof_index, "accel")

  def joint_motions(self, joint_dof : int, joint_dof_index : int, order = None):
    if order is None:
      order = self.motion_num
    values = ()
    for i in range(order):
      values += (self.gen_value(joint_dof, joint_dof_index, self.aliases[i]),)

    return np.array(values)

  def link_motions(self, link_dof : int, link_dof_index : int, order = None):
    if order is None:
      order = self.motion_num
    values = ()
    for i in range(order):
      values += (self.gen_value(link_dof, link_dof_index, self.aliases[i]),)

    return np.array(values)