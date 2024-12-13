#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki

import polars as pl

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
    
  @staticmethod
  def link_state_vec(df, link, name):
    return df[link.name+"_"+name][-1].to_numpy()
  
#   @staticmethod
#   def vec_to_mat(mat_vec):
#     nn = len(mat_vec)
#     n = int(np.sqrt(nn))

#     mat = np.zeros((n,n))
#     for i in range(n):
#       mat[i,0:n] = mat_vec[n*i:n*i+n]
#     return mat   
  
#   def mat_to_vec(mat):
#     n = len(mat)

#     mat_vec = np.zeros(n*n)
#     for i in range(n):
#       mat_vec[n*i:n*i+n] = mat[i,0:n]
#     return mat_vec   
    
  @staticmethod
  def link_state_mat(df, link, name):
    mat_vec = df[link.name+"_"+name][-1].to_numpy()
    mat = RobotState.vec_to_mat(mat_vec)
    return mat
  
  def all_state_vec(self, robot, name):
    labels = []
    for l in robot.links:
      labels.append(l.name+"_"+name) 
    mat = [self.state_df.df[label][-1].to_list() for label in labels]
    return np.array(mat)
  
  def link_pos(self, link):
    return RobotState.link_state_vec(self.state_df(), link, "pos")
  
  def all_link_pos(self, robot):
    return self.all_state_vec(robot, "pos")
  
  def link_rot(self, link):
    return RobotState.link_state_mat(self.state_df(), link, "rot")

  def link_vel(self, link):
    return RobotState.link_state_vec(self.state_df(), link, "vel")

  def link_acc(self, link):
    return RobotState.link_state_vec(self.state_df(), link, "acc")
    
  def link_frame(self, link):
    h = SE3(self.link_rot(link), self.link_pos(link))
    return h.mat()

  def link_adj_frame(self, link):
    a = SE3(self.link_rot(link), self.link_pos(link))
    return a.adjoint()

  def import_state(self, data):
    self.state_df.add_row(data)