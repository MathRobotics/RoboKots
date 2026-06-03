#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki

import numpy as np
try:
  import polars as pl
except ImportError as e:  # pragma: no cover
  raise ImportError(
    "robokots.contrib.polars requires the optional `polars` dependency. "
    "Install RoboKots with `pip install 'robokots[table]'`."
  ) from e

from mathrobo import SE3, CMTM
from robokots.core.state import state_storage_key
from robokots.core.state_dict import (
  extract_dict_link_info,
  extract_dict_joint_info,
  is_state_payload_key,
)

class RobotDF:
  df : pl.DataFrame
  names : list
  def __init__(self, names_ : list):
    self.names = names_

    self.df = pl.DataFrame()
    self.set_df()
    
  def add_row(self, data : dict):
    row = {
      key: value
      for key, value in data.items()
      if is_state_payload_key(key) and key in self.df.schema
    }
    new_row = pl.DataFrame([row], schema=self.df.schema)
    self.df = self.df.vstack(new_row)
    
  def set_df(self):
    for name in self.names:
      self.df = self.df.with_columns([pl.Series(name=name, dtype=pl.List(pl.Float64))])

class RobotState:
  state_df : RobotDF
  
  def __init__(self, link_names : list, joint_names : list, l_aliases = ["pos", "rot", "vel", "acc"], j_aliases = [], separator = "_"):
    names = []
    self.l_aliases = l_aliases
    self.j_aliases = j_aliases
    self.separator = separator  
    if len(l_aliases) != 0:
      for l_name in link_names:
        for al in l_aliases:
          names.append(l_name + separator + "link" + separator + al)
    
    if len(j_aliases) != 0:
      for j_name in joint_names:
        for al in j_aliases:
          names.append(j_name + separator + "joint" + separator + al)

    self.state_df = RobotDF(names)
    
  def df(self) -> pl.DataFrame:
    if self.state_df.df.is_empty():
      raise ValueError("DataFrame is empty. Please add data first.")
    return self.state_df.df
    
  @staticmethod
  def state_vec(df, owner_name : str, owner_type : str, data_type : str) -> np.ndarray:
    return df[state_storage_key(owner_type, owner_name, data_type)][-1].to_numpy()
  
  @staticmethod
  def state_vecs(df, owner_name_list : str, owner_type : str, data_type : str) -> np.ndarray:
    vecs = []
    for name in owner_name_list:
      vecs.append(df[state_storage_key(owner_type, name, data_type)][-1])
    return np.array(vecs)
  
  @staticmethod
  def state_vec_traj(df, owner_name : str, owner_type : str, data_type : str) -> np.ndarray:
    return df[state_storage_key(owner_type, owner_name, data_type)].to_numpy()

  @staticmethod
  def state_vecs_traj(df, owner_name_list : list, owner_type : str, data_type : str) -> np.ndarray:
    length = df[state_storage_key(owner_type, owner_name_list[0], data_type)].shape[0]
    vecs = np.zeros((len(owner_name_list), length, 3))
    for i in range(len(owner_name_list)):
      vecs[i] = np.array(df[state_storage_key(owner_type, owner_name_list[i], data_type)].to_list())
    return vecs
  
  @staticmethod
  def state_mat(df, owner_name : str, owner_type : str, data_type : str) -> np.ndarray:
    mat_vec = df[state_storage_key(owner_type, owner_name, data_type)][-1].to_numpy()
    mat = mat_vec.reshape((3,3))
    return mat

  @staticmethod
  def state_mats_traj(df, owner_name_list : list, owner_type : str, data_type : str) -> np.ndarray:
    length = df[state_storage_key(owner_type, owner_name_list[0], data_type)].shape[0]
    mats = np.zeros((len(owner_name_list), length, 3, 3))
    for i in range(len(owner_name_list)):
      mat_vecs = np.array(df[state_storage_key(owner_type, owner_name_list[i], data_type)].to_list())
      mats[i] = mat_vecs.reshape((length, 3, 3))
    return mats

  def state_dict(self, index : int = -1) -> dict:
    return {
      key: np.asarray(value)
      for key, value in self.df().row(index, named=True).items()
    }
  
  def link_values(self, link_name : str, order : int) -> dict:
    if order < 1:
      raise ValueError(f"Invalid order: {order}. Must be over 1.")
    
    d = []
    d.append(self.extract_link_info("frame", link_name))
    if order > 1:
      d.append(self.extract_link_info("vel", link_name))
    if order > 2:
      d.append(self.extract_link_info("acc", link_name))
    if order > 3:
      for i in range(order-3):
        d.append(RobotState.state_vec(self.df(), link_name, "link", "acc_diff"+str(i+1)))
    return d
  
  def link_cmtm(self, link_name : str, order = 3) -> CMTM:
    vec = np.zeros((order-1, 6))
    state = self.link_values(link_name, order)
    h = state[0]
    for i in range(1, order):
      vec[i-1] = state[i]
    return CMTM[SE3](h, vec)

  def joint_values(self, joint_name : str, order : int) -> dict:
    if order < 1:
      raise ValueError(f"Invalid order: {order}. Must be over 1.")
    
    d = []
    d.append(self.extract_joint_info("frame", joint_name))
    if order > 1:
      d.append(self.extract_joint_info("vel", joint_name))
    if order > 2:
      d.append(self.extract_joint_info("acc", joint_name))
    if order > 3:
      for i in range(order-3):
        d.append(RobotState.state_vec(self.df(), joint_name, "joint", "acc_diff"+str(i+1)))
    return d
  
  def joint_cmtm(self, joint_name : str, order = 3) -> CMTM:
    vec = np.zeros((order-1, 6))
    state = self.joint_values(joint_name, order)
    h = state[0]
    for i in range(1, order):
      vec[i-1] = state[i]
    return CMTM[SE3](h, vec)

  #specific 3d-CMTM
  def extract_link_info(self, type : str, link_name : str, frame = "dummy", rel_frame = 'dummy'):
    frame = None if frame == "dummy" else frame
    return extract_dict_link_info(self.state_dict(), type, link_name, frame, rel_frame)
    
  def extract_joint_info(self, type : str, joint_name : str, frame = "dummy", rel_frame = 'dummy'):
    frame = None if frame == "dummy" else frame
    return extract_dict_joint_info(self.state_dict(), type, joint_name, frame, rel_frame)
    
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
    
  def extract_links_info_traj(self, type : str, link_names : list, frame = "dummy", rel_frame = 'dummy'):
    '''
    type : str
      pos
      rot
      vel
      acc
      frame
    names : list of str
      link names
    '''
    if type == "pos":
      return RobotState.state_vecs_traj(self.df(), link_names, "link", "pos")
    elif type == "rot":
      return RobotState.state_mats_traj(self.df(), link_names, "link", "rot")
    elif type == "vel":
      return RobotState.state_vecs_traj(self.df(), link_names, "link", "vel")
    elif type == "acc":
      return RobotState.state_vecs_traj(self.df(), link_names, "link", "acc")
    elif type == "frame":
      return SE3(RobotState.state_mats_traj(self.df(), link_names, "link", "rot"),
             RobotState.state_vecs_traj(self.df(), link_names, "link", "pos"))
    elif type == "cmtm":
      return self.link_cmtm(link_names)
    else:
      raise ValueError(f"Invalid type: {set(type)}")

  def import_state(self, data : dict):
    self.state_df.add_row(data)

  def clear_state(self) -> dict:
    return self.state_df.df.clear()
