#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.08.14 Created by T.Ishigaki

from .df import *

class RobotState:
  df : RobotDF
  
  def __init__(self, robot, aliases = ["pos", "rot", "vel", "acc"], separator = "_"):
    state_names = robot.link_names
    self.df = RobotDF(state_names, aliases, separator)
    
  @staticmethod
  def link_state_vec(df, link, name):
    return df[link.name+"_"+name][-1].to_numpy()
  
  @staticmethod
  def vec_to_mat(mat_vec):
    nn = len(mat_vec)
    n = int(np.sqrt(nn))

    mat = np.zeros((n,n))
    for i in range(n):
      mat[i,0:n] = mat_vec[n*i:n*i+n]
    return mat   
  
  def mat_to_vec(mat):
    n = len(mat)

    mat_vec = np.zeros(n*n)
    for i in range(n):
      mat_vec[n*i:n*i+n] = mat[i,0:n]
    return mat_vec   
    
  @staticmethod
  def link_state_mat(df, link, name):
    mat_vec = df[link.name+"_"+name][-1].to_numpy()
    mat = RobotState.vec_to_mat(mat_vec)
    return mat
  
  def all_state_vec(self, robot, name):
    labels = []
    for l in robot.links:
      labels.append(l.name+"_"+name) 
    mat = [self.df.df[label][-1].to_list() for label in labels]
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
    return h.mat()

  def link_adj_frame(self, link):
    a = SE3(self.link_rot(link), self.link_pos(link))
    return a.adjoint()

  def import_state(self, data):
    self.df.add_row(data)