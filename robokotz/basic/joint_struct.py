#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.07.27 Created by T.Ishigaki

import numpy as np
import warnings

from dataclasses import dataclass

@dataclass
class JointStruct_:
  '''
    joint primitive imformation
  '''
  name: str = 'name'
  joint_type: str = "revolution"
  is_root: bool = False
  
class JointStruct(JointStruct_):
  id : int = 0
  dof : int = 0
  dof_index : int = 0
  
  connect_link: np.ndarray = np.array([])

  joint_select_mat : np.ndarray = np.array([])

  def init(self):
    self.joint_select_mat = self._joint_select_mat(self.joint_type)
    self.set_dof()
    
  def set_dof(self):
    self.dof = self._joint_dof(self.joint_type)

  @staticmethod
  def _joint_dof(type):
    if type == "revolution":
      return 1
    elif type == "fix":
      return 0
    else:
      warnings.warn('Not applicable joint type', DeprecationWarning)
      return 0
    
  @staticmethod
  def _joint_select_mat(joint_type):
    if joint_type == 'fix':
      return np.zeros((6,1))
    elif joint_type == 'revolution':
      mat = np.zeros((6,1))
      mat[2,0] = 1
      return mat
    else:
      warnings.warn('Not applicable joint type', DeprecationWarning)
      return np.zeros((6,1))