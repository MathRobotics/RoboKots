#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.11 Created by T.Ishigaki

import json
from typing import Dict

from .robot_model import *
from .target import *

class RobotIO():
  def load_json(file_path: str) -> Dict:
      try:
          with open(file_path, 'r', encoding='utf-8') as file:
              return json.load(file)
      except FileNotFoundError:
          raise FileNotFoundError(f"File {file_path} not found.")
      except json.JSONDecodeError as e:
          raise ValueError(f"Invalid JSON format: {e}")

  def save_json(data: Dict, file_path: str):
      try:
          with open(file_path, 'w', encoding='utf-8') as file:
              json.dump(data, file, indent=4)
      except Exception as e:
          raise IOError(f"Failed to write JSON file: {e}")
      
  @staticmethod
  def from_json_file(file_path: str) -> "RobotStruct":
      data = RobotIO.load_json(file_path)
      return RobotStruct.from_dict(data)

  def print_structure(robot : RobotStruct):
      robot.print()
          
  @staticmethod
  def from_target_json(file_path: str) -> "TargetList":
      data = RobotIO.load_json(file_path)
      return TargetList.from_dict(data)
  
  def print_targets(t_list : TargetList):
      t_list.print()