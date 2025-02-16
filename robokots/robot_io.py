#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.11 Created by T.Ishigaki

import json
from typing import Dict

from .robot_model import *

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
      print(f"Robot DOF: {robot.dof}")
      print("\nLinks:")
      for link in robot.links:
          print(f"  ID: {link.id}, Name: {link.name}, Type: {link.type}")
          print(f"    COG: {link.cog}, Mass: {link.mass}")
          print(f"    Inertia: {link.inertia}, DOF: {link.dof}")
          print(f"    Connect parent joint: {link.parent_joint_ids}")
          print(f"    Connect child joint: {link.child_joint_ids}")
          print(f"    DOF index: {link.dof_index}")

      print("\nJoints:")
      for joint in robot.joints:
          print(f"  ID: {joint.id}, Name: {joint.name}, Type: {joint.type}")
          print(f"    Axis: {joint.axis}, Parent Link: {joint.parent_link_id}, Child Link: {joint.child_link_id}")
          print(f"    DOF: {joint.dof}")
          print(f"    Origin: {joint.origin.pos()}")
          print(f"{joint.origin.rot()}")
          print(f"    DOF index: {joint.dof_index}")
