#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025.02.20 Created by T.Ishigaki

from typing import List, Dict

import numpy as np

from mathrobo import *

class Target:
  def __init__(self, type: str, link_name: str, pos: np.ndarray):
    self.type = type
    self.link_name = link_name
    self.pos = pos
    
  def set_index(self, i):
    self.index = i    

class TargetList:
  def __init__(self, targets_: List["Target"]):
    self.targets = targets_
    self.target_num = len(self.targets)  
    self.target_names = [t.link_name for t in self.targets]
    self.target_types = [t.type for t in self.targets]
    self.target_positions = [t.pos for t in self.targets]
    
    index = 0
    for t in self.targets:
      t.set_index(index)
      index += 1
    
  @staticmethod
  def from_dict(data: Dict):  
    targets = []
    
    targets = [Target(
        type=target["type"],
        link_name=target["link"],
        pos=np.array(target.get("pos", [0., 0., 0.]))
    ) for target in data["targets"]]

    return TargetList(targets)
  
  def print(self):
      print(f"Target Number: {self.target_num}")
      print("\nTargets:")
      for t in self.targets:
          print(f"  Type: {t.type}, Link: {t.link_name}, Index: {t.index}")
          print(f"  Pos: {t.pos}\n")