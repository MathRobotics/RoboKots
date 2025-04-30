#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki

import polars as pl
import numpy as np

from mathrobo import SE3, CMTM

from .robot import RobotStruct

class RobotDF:
  df : pl.DataFrame
  names : list
  def __init__(self, names_ : list):
    self.names = names_

    self.df = pl.DataFrame()
    self.set_df()
    
  def add_row(self, data : dict):
    new_row = pl.DataFrame([data], schema=self.df.schema)
    self.df = self.df.vstack(new_row)
    
  def set_df(self):
    for name in self.names:
      self.df = self.df.with_columns([pl.Series(name=name, dtype=pl.List(pl.Float64))])

class RobotState:
  state_df : RobotDF
  
  def __init__(self, robot : RobotStruct, l_aliases = ["pos", "rot", "vel", "acc"], j_aliases = [], separator = "_"):
    names = []
    self.l_aliases = l_aliases
    self.j_aliases = j_aliases
    self.separator = separator  
    if len(l_aliases) != 0:
      for l_name in robot.link_names:
        for al in l_aliases:
          names.append(l_name + separator + al)
    
    if len(j_aliases) != 0:
      for j_name in robot.joint_names:
        for al in j_aliases:
          names.append(j_name + separator + al)

    self.state_df = RobotDF(names)
    
  def df(self) -> pl.DataFrame:
    if self.state_df.df.is_empty():
      raise ValueError("DataFrame is empty. Please add data first.")
    return self.state_df.df
    
  @staticmethod
  def link_state_vec(df, link_name : str, type : str) -> np.ndarray:
    return df[link_name+"_"+type][-1].to_numpy()
  
  @staticmethod
  def link_state_mat(df, link_name : str, type : str) -> np.ndarray:
    mat_vec = df[link_name+"_"+type][-1].to_numpy()
    mat = mat_vec.reshape((3,3))
    return mat
  
  def all_state_vec(self, robot : RobotStruct, type : str) -> np.ndarray:
    labels = []
    for l in robot.links:
      labels.append(l.name+"_"+type) 
    mat = [self.df()[label][-1].to_list() for label in labels]
    return np.array(mat)
  
  def link_pos(self, link_name : str) -> np.ndarray:
    return RobotState.link_state_vec(self.df(), link_name, "pos")
  
  def all_link_pos(self, robot : RobotStruct) -> np.ndarray:
    return self.all_state_vec(robot, "pos")
  
  def link_rot(self, link_name : str) -> np.ndarray:
    return RobotState.link_state_mat(self.df(), link_name, "rot")

  def link_vel(self, link_name : str) -> np.ndarray:
    return RobotState.link_state_vec(self.df(), link_name, "vel")

  def link_acc(self, link_name : str) -> np.ndarray:
    return RobotState.link_state_vec(self.df(), link_name, "acc")
    
  def link_frame(self, link_name : str) -> SE3:
    h = SE3(self.link_rot(link_name), self.link_pos(link_name))
    return h
  
  def link_values(self, link_name : str, order : int) -> dict:
    if order < 1:
      raise ValueError(f"Invalid order: {order}. Must be over 1.")
    
    d = []
    d.append(self.link_frame(link_name))
    if order > 1:
      d.append(self.link_vel(link_name))
    if order > 2:
      d.append(self.link_acc(link_name))
    if order > 3:
      for i in range(order-3):
        d.append(RobotState.link_state_vec(self.df(), link_name, "acc_diff"+str(i+1)))
    return d
  
  def link_cmtm(self, link_name : str, order = 3) -> CMTM:
    vec = np.zeros((order-1, 6))
    state = self.link_values(link_name, order)
    h = state[0]
    for i in range(1, order):
      vec[i-1] = state[i]
    cmtm = CMTM[SE3](h, vec)
    return cmtm

  def link_rel_frame(self, base_link_name : str, target_link_name : str) -> SE3:
    h = self.link_frame(base_link_name).inv() @ self.link_frame(target_link_name)
    return h
  
  def link_rel_cmtm(self, base_link_name : str, target_link_name : str, order : int) -> CMTM:
    x = self.link_cmtm(base_link_name, order).inv() @ self.link_cmtm(target_link_name, order)
    return x
  
  @staticmethod
  def joint_state_vec(df, joint_name : str, type : str) -> np.ndarray:
    return df[joint_name+"_"+type][-1].to_numpy()
  
  def joint_coord(self, joint_name : str) -> np.ndarray:
    return RobotState.joint_state_vec(self.df(), joint_name, "coord")
  
  def joint_veloc(self, joint_name : str) -> np.ndarray:
    return RobotState.joint_state_vec(self.df(), joint_name, "veloc")
  
  def joint_accel(self, joint_name : str) -> np.ndarray:
    return RobotState.joint_state_vec(self.df(), joint_name, "accel")

  def joint_values(self, joint_name : str, order : int) -> np.ndarray:
    if order < 1:
      raise ValueError(f"Invalid order: {order}. Must be over 1.")
    
    # magic number 1 dof joint
    d = np.zeros((order, 1))
    d[0] = self.joint_coord(joint_name)
    if order > 1:
      d[1] = self.joint_veloc(joint_name)
    if order > 2:
      d[2] = self.joint_accel(joint_name)
    if order > 3:
      for i in range(order-3):
        d[i+3] = RobotState.link_state_vec(self.df(), joint_name, "accel_diff"+str(i+1))
    return d
  
  def joint_cmtm(self, joint_name : str, order = 3) -> CMTM:
    vec = np.zeros((order-1, 6))
    state = self.joint_values(joint_name, order)
    h = SE3(state[0])
    for i in range(1, order):
      vec[i-1] = state[i]
    cmtm = CMTM[SE3](h, vec)
    return cmtm

  #specific 3d-CMTM
  def extract_link_info(self, type : str, link_name : str, frame = "dummy", rel_frame = 'dummy'):
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
    elif type == "cmtm":
      return self.link_cmtm(link_name)
    else:
      raise ValueError(f"Invalid type: {set(type)}")
    
  def extract_joint_info(self, type : str, name : str, frame = "dummy", rel_frame = 'dummy'):
    'dummy'
    
  def extract_total_info(self, type : str, name : str, frame = "dummy", rel_frame = 'dummy'):
    'dummy'
  
  def extract_info(self, group : str, type : str, name : str, frame = "dummy", rel_frame = 'dummy'):
    '''
    group : str
      link
      joint
      total
    type : str
      pos
      rot
      vel
      acc
      frame
    name : str
      link name or joint name
    '''
    if group == "link":
      return self.extract_link_info(type, name, frame, rel_frame)
    elif group == "joint":
      return self.extract_joint_info(type, name, frame, rel_frame)
    elif group == "total":
      return self.extract_total_info(type, name, frame, rel_frame)
    else:
      raise ValueError(f"Invalid group: {set(group)}")

  def import_state(self, data : dict):
    self.state_df.add_row(data)