#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025.02.20 Created by T.Ishigaki

from typing import List, Dict

import numpy as np
from mathrobo import SE3

from .state import StateType, keys_time_order
class Target:
  def __init__(self, state_type: StateType):
    self._state_type = state_type
    self._index = -1
    self._target_name = None

  def set_index(self, i : int):
    self._index = i

  def __repr__(self):
     return f"Target(\n  state type: {self._state_type}\n)"
class TargetList:
  def __init__(self, targets: List["Target"]):
    self._targets = targets
    self._target_num = len(self._targets)  
    self._max_order = max([keys_time_order[t._state_type.data_type] for t in self._targets])
    
    index = 0
    for t in self._targets:
      t.set_index(index)
      index += 1
    
  @staticmethod
  def from_dict(data: Dict) -> "TargetList":  
    targets = []
    
    for target in data["targets"]:
      if isinstance(target.get("data_type"), str):
        data_types = [target.get("data_type")]
      else:
        data_types = list(target.get("data_type"))
      for dt in data_types:
        t = Target(
          StateType(
            target["owner_type"],
            target["owner_name"],
            dt,
            target.get("frame_name")
          )
        )
        targets.append(t)

    return TargetList(targets)

  def __repr__(self):
     return f"TargetList(\n  targets: {self._targets}\n)"