#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025.02.20 Created by T.Ishigaki

from typing import List, Dict

import numpy as np
from mathrobo import SE3
class Target:
  def __init__(self, data_type: str, owner_type, owner_name: str, frame_name: str = None, frame: SE3 = SE3.eye()):
    self.data_type = data_type
    self.owner_type = owner_type
    self.owner_name = owner_name
    self.frame_name = frame_name
    self.frame = frame
    
  def set_index(self, i : int):
    self.index = i    
class TargetList:
  def __init__(self, targets_: List["Target"]):
    self.targets = targets_
    self.target_num = len(self.targets)  
    self.target_owner_types = [t.owner_type for t in self.targets]
    self.target_owner_names = [t.owner_name for t in self.targets]
    self.target_data_types = [t.data_type for t in self.targets]
    self.target_frame_names = [t.frame_name for t in self.targets]
    self.target_frames = [t.frame for t in self.targets]
    
    index = 0
    for t in self.targets:
      t.set_index(index)
      index += 1
    
  @staticmethod
  def from_dict(data: Dict) -> "TargetList":  
    targets = []

    def normalize_types(ts):
        return [ts] if isinstance(ts, str) else list(ts)
    
    targets = [Target(
        data_type=normalize_types(target["data_type"]),
        owner_type=target["owner_type"],
        owner_name=target["owner_name"],
        frame_name=target["frame_name"],
        frame=np.array(target.get("frame", SE3.eye()))
    ) for target in data["targets"]]

    return TargetList(targets)
  
  def print(self):
      print(f"Target Number: {self.target_num}")
      print("\nTargets:")
      for t in self.targets:
          print(f"  Owner Type: {t.owner_type}")
          print(f"  Owner Name: {t.owner_name}")
          print(f"  Data Type: {t.data_type}")
          print(f"  Index: {t.index}")
          print(f"  Frame Name: {t.frame_name}")
          print(f"  Frame: {t.frame}\n")