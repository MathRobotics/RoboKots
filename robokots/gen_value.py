#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.08.14 Created by T.Ishigaki

import numpy as np
import warnings

from .df import *

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
  

class RobotGenValue:
  df : RobotDF
  coord : np.ndarray = np.array([])
  veloc : np.ndarray = np.array([])
  accel : np.ndarray = np.array([])
  force : np.ndarray = np.array([])
  
  def __init__(self, robot, aliases = ["coord", "veloc", "accel", "force"], separator = "_"):
    state_names = robot.joint_names + robot.link_names
    self.df = RobotDF(state_names, aliases, separator)
  
  def to_dict(self, index):
    gen_value_row_tuple = self.df.df.row(index)
    gen_value_columns = self.df.df.columns
    gen_value = dict(zip(gen_value_columns, gen_value_row_tuple))
    return gen_value
    
  def export_vec(self, robot, name):
    vec = np.zeros(robot.dof)
    
    for j in robot.joints:
      vec[j.dof_index : j.dof_index + j.dof] = self.df()[j.name + "_" + name][-1].to_numpy()
    
    for l in robot.links:
      vec[l.dof_index : l.dof_index + l.dof] = self.df()[l.name + "_" + name][-1].to_numpy()
      
    return vec

  def export_coord(self, robot):
    self.coord = self.export_vec(robot, "coord")
    return self.coord
  
  def export_veloc(self, robot):
    self.veloc = self.export_vec(robot, "veloc")
    return self.veloc
  
  def export_accel(self, robot):
    self.accel = self.export_vec(robot, "accel")
    return self.accel
  
  def export_force(self, robot):
    self.force = self.export_vec(robot, "force")
    return self.force
  
  def _vec_to_gen_value(self, dof, index, vec):
    if(dof > 0):
      return vec[index:index+dof]
    else:
      return []
  
  def import_vecs(self, robot, vecs):
    data = {}
    for j in robot.joints:
      for i in range(len(vecs)):
        vec = self._vec_to_gen_value(j.dof, j.dof_index, vecs[i])
        data.update([(j.name + "_" + self.df.aliases[i] , vec)])
    for l in robot.links:
      for i in range(len(vecs)):  
        vec = self._vec_to_gen_value(l.dof, robot.joint_dof+l.dof_index, vecs[i])
        data.update([(l.name + "_" + self.df.aliases[i] , vec)])
    self.df.add_row(data)
    
  def joint_gen_value(self, joint, name):
    return self.df.df[joint.name + "_" + name][-1].to_numpy()

  def joint_coord(self, joint):
    return self.link_gen_value(joint, "coord")
  
  def joint_veloc(self, joint):
    return self.link_gen_value(joint, "veloc")
  
  def joint_accel(self, joint):
    return self.link_gen_value(joint, "accel")
  
  def joint_force(self, joint):
    return self.link_gen_value(joint, "force")
    
  def link_gen_value(self, link, name):
    return self.df.df[link.name + "_" + name][-1].to_numpy()

  def link_coord(self, link):
    return self.link_gen_value(link, "coord")
  
  def link_veloc(self, link):
    return self.link_gen_value(link, "veloc")
  
  def link_accel(self, link):
    return self.link_gen_value(link, "accel")
  
  def link_force(self, link):
    return self.link_gen_value(link, "force")