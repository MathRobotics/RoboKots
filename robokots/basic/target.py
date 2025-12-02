#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025.02.20 Created by T.Ishigaki

from typing import List, Dict
from .state import StateType, keys_time_order

class TargetList:
  def __init__(self, targets: List["StateType"]):
    self._targets = targets
    self._target_num = len(self._targets)  
    self._max_order = max([keys_time_order[t.data_type] for t in self._targets])

  @staticmethod
  def from_dict(data: Dict) -> "TargetList":  
    state_types = []
    
    for target in data["targets"]:
      if isinstance(target.get("data_type"), str):
        data_types = [target.get("data_type")]
      else:
        data_types = list(target.get("data_type"))
      for dt in data_types:
        st = StateType(
          target["owner_type"],
          target["owner_name"],
          dt,
          target.get("frame_name")
        )
        state_types.append(st)

    return TargetList(state_types)

  def __repr__(self):
    rep = "TargetList(\n"
    rep += f"  target num: {self._target_num}\n"
    rep += f"  max order: {self._max_order}\n"
    for st in self._targets:
        rep += f"  {st}\n"
    rep += ")"
    return rep