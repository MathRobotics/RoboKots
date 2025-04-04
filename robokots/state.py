#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki

import polars as pl
import numpy as np

from mathrobo import *

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
  def link_state_vec(df, link, name):
    return df[link.name+"_"+name][-1].to_numpy()
  
  @staticmethod
  def link_state_mat(df, link, name):
    mat_vec = df[link.name+"_"+name][-1].to_numpy()
    mat = mat_vec.reshape((3,3))
    return mat
  
  def all_state_vec(self, robot, name):
    labels = []
    for l in robot.links:
      labels.append(l.name+"_"+name) 
    mat = [self.df()[label][-1].to_list() for label in labels]
    return np.array(mat)
  
  def link_pos(self, link):
    return RobotState.link_state_vec(self.df(), link, "pos")
  
  def all_link_pos(self, robot):
    return self.all_state_vec(robot, "pos")
  
  def link_rot(self, link):
    return RobotState.link_state_mat(self.df(), link, "rot")

  def link_vel(self, link):
    return RobotState.link_state_vec(self.df(), link, "vel")

  def link_acc(self, link):
    return RobotState.link_state_vec(self.df(), link, "acc")
    
  def link_frame(self, link):
    h = SE3(self.link_rot(link), self.link_pos(link))
    return h

  def link_rel_frame(self, base_link, target_link):
    h = SE3(self.link_rot(target_link), self.link_pos(target_link)).inv() \
        @ SE3(self.link_rot(base_link), self.link_pos(base_link))
    return h

  def import_state(self, data):
    self.state_df.add_row(data)