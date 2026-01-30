#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025.02.20 Created by T.Ishigaki

import dataclasses
from typing import List, Dict
from .state import StateType, keys_time_order

@dataclasses.dataclass
class RobotNames:
  joint_names: List[str]
  link_names: List[str]
  
class TargetList:
  def __init__(self, targets: List["StateType"]):
    self._targets = targets
    self._target_num = len(self._targets)  
    self._max_order = max([keys_time_order[t.data_type] for t in self._targets])

  @staticmethod
  def from_dict(data: Dict, robot : RobotNames) -> "TargetList":  
    if not isinstance(data, dict):
      raise ValueError("Input data must be a dictionary.")
    
    state_types = []
    
    for target in data["targets"]:
      if isinstance(target.get("data_type"), str):
        data_types = [target.get("data_type")]
      else:
        data_types = list(target.get("data_type"))
      owner_names = []
      if target["owner_type"] == "joint":
        if target["owner_name"] not in robot.joint_names:
          raise ValueError(f"TargetList.from_dict: joint name '{target['owner_name']}' is not found in robot.joint_names")
        else:
          owner_names.append(target["owner_name"])
      elif target["owner_type"] == "link":
        if target["owner_name"] not in robot.link_names:
          raise ValueError(f"TargetList.from_dict: link name '{target['owner_name']}' is not found in robot.link_names")
        else:
          owner_names.append(target["owner_name"])
      elif target["owner_type"] == "total_link":
        owner_names.append(robot.link_names)
      elif target["owner_type"] == "total_joint":
        owner_names.append(robot.joint_names)

      for dt in data_types:
        for owner_name in owner_names:
          st = StateType(
            owner_type=target["owner_type"],
            owner_name=owner_name,
            data_type=dt,
            frame_name=target.get("frame_name")
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