#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025.02.20 Created by T.Ishigaki

from typing import List, Dict

import numpy as np

class Target:
  def __init__(self, type: str, owner_type, owner_name: str, pos: np.ndarray):
    self.type = type
    self.owner_type = owner_type
    self.owner_name = owner_name
    self.pos = pos
    
  def set_index(self, i : int):
    self.index = i    

class TargetList:
  def __init__(self, targets_: List["Target"]):
    self.targets = targets_
    self.target_num = len(self.targets)  
    self.target_owner_types = [t.owner_type for t in self.targets]
    self.target_owner_names = [t.owner_name for t in self.targets]
    self.target_types = [t.type for t in self.targets]
    self.target_positions = [t.pos for t in self.targets]
    
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
        type=normalize_types(target["type"]),
        owner_type=target["owner_type"],
        owner_name=target["owner_name"],
        pos=np.array(target.get("pos", [0., 0., 0.]))
    ) for target in data["targets"]]

    return TargetList(targets)
  
  def print(self):
      print(f"Target Number: {self.target_num}")
      print("\nTargets:")
      for t in self.targets:
          print(f"  Type: {t.type}")
          print(f"  Owner Type: {t.owner_type}")
          print(f"  Owner Name: {t.owner_name}")
          print(f"  Index: {t.index}")
          print(f"  Pos: {t.pos}\n")