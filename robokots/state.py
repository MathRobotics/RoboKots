#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki

import polars as pl
import numpy as np

from mathrobo import SE3

class RobotDF:
  def __init__(self, names_, aliases_, separator_ = "_"):
    self.names = names_
    self.aliases = aliases_
    self.separator = separator_

    self.df = pl.DataFrame()
    self.set_df()
    
  def add_row(self, data):
    new_row = pl.DataFrame([data])
    self.df = self.df.vstack(new_row)
    
  def set_df(self):
    for name in self.names:
      for a in self.aliases:
        alias_name = name + self.separator + a
        self.df = self.df.with_columns([pl.Series(name=alias_name, dtype=pl.List(pl.Float64))])

class RobotState:
  state_df : RobotDF
  
  def __init__(self, robot, aliases = ["pos", "rot", "vel", "acc"], separator = "_"):
    state_names = robot.link_names
    self.state_df = RobotDF(state_names, aliases, separator)
    
  def df(self):
    return self.state_df.df
    
  @staticmethod
  def link_state_vec(df, link_name, type):
    return df[link_name+"_"+type][-1].to_numpy()
  
  @staticmethod
  def link_state_mat(df, link_name, type):
    mat_vec = df[link_name+"_"+type][-1].to_numpy()
    mat = mat_vec.reshape((3,3))
    return mat
  
  def all_state_vec(self, robot, type):
    labels = []
    for l in robot.links:
      labels.append(l.name+"_"+type) 
    mat = [self.df()[label][-1].to_list() for label in labels]
    return np.array(mat)
  
  def link_pos(self, link_name):
    return RobotState.link_state_vec(self.df(), link_name, "pos")
  
  def all_link_pos(self, robot):
    return self.all_state_vec(robot, "pos")
  
  def link_rot(self, link_name):
    return RobotState.link_state_mat(self.df(), link_name, "rot")

  def link_vel(self, link_name):
    return RobotState.link_state_vec(self.df(), link_name, "vel")

  def link_acc(self, link_name):
    return RobotState.link_state_vec(self.df(), link_name, "acc")
    
  def link_frame(self, link_name):
    h = SE3(self.link_rot(link_name), self.link_pos(link_name))
    return h

  def link_rel_frame(self, base_link_name, target_link_name):
    h = SE3(self.link_rot(base_link_name), self.link_pos(base_link_name)).inv() \
        @ SE3(self.link_rot(target_link_name), self.link_pos(target_link_name))
    return h

  def extract_link_info(self, type, link_name, frame = "dummy", rel_frame = 'dummy'):
    if type == "pos":
      return self.link_pos(link_name)
    elif type == "rot":
      return self.link_rot(link_name)
    elif type == "vel":
      return self.link_vel(link_name)
    elif type == "acc":
      return self.link_acc(link_name)
    elif type == "frame":
      return self.link_frame(link_name)
    else:
      raise ValueError(f"Invalid type: {set(type)}")
    
  def extract_joint_info(self, type, name, frame = "dummy", rel_frame = 'dummy'):
    'dummy'
    
  def extract_total_info(self, type, name, frame = "dummy", rel_frame = 'dummy'):
    'dummy'
  
  def extract_info(self, group, type, name, frame = "dummy", rel_frame = 'dummy'):
    if group == "link":
      return self.extract_link_info(type, name, frame, rel_frame)
    elif group == "joint":
      return self.extract_link_info(type, name, frame, rel_frame)
    elif group == "total":
      return self.extract_link_info(type, name, frame, rel_frame)
    else:
      raise ValueError(f"Invalid group: {set(group)}")

  def import_state(self, data):
    self.state_df.add_row(data)