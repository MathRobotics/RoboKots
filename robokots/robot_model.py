#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.11 Created by T.Ishigaki

import numpy as np

import json

import warnings
from typing import List, Dict

warnings.simplefilter("always", UserWarning)

class RobotStruct:
  def __init__(self, links_: List["LinkStruct"], joints_: List["JointStruct"]):
    self.joints = joints_
    self.links = links_
    self.joint_num: int = 0
    self.link_num: int = 0
    self.dof: int = 0
    self.joint_dof: int = 0
    self.link_dof: int = 0
    self.link_names: List[str] = []
    self.joint_names: List[str] = []
    self.robot_init()

  def robot_init(self):
    self.joint_num = len(self.joints)  
    self.link_num = len(self.links)  

    self.dof = 0
    self.joint_dof = 0
    self.link_dof = 0
    
    for j in self.joints:
      self.joint_dof += j.dof
      
    for l in self.links:
      self.link_dof += l.dof
      
    self.dof = self.joint_dof + self.link_dof
    
    self.link_names = [l.name for l in self.links]
    self.joint_names = [j.name for j in self.joints]
    
  @staticmethod
  def from_json_file(file_path: str) -> "RobotStruct":
      with open(file_path, 'r', encoding='utf-8') as file:
          data = json.load(file)
      return RobotStruct.from_dict(data)
  
  @staticmethod
  def from_dict(data: Dict):  
    joints = []
    links = []
    
    links = [LinkStruct(
        id=link["id"],
        name=link["name"],
        cog=np.array(link.get("cog", [0., 0., 0.])),
        mass=float(link.get("mass", 0.)),
        inertia=np.array(link.get("inertia", [1.0, 1.0, 1.0, 0.0, 0.0, 0.0])),
        type=link.get("type", "rigid")
    ) for link in data["links"]]

    joints = [JointStruct(
        id=joint["id"],
        name=joint["name"],
        type=joint["type"],
        axis=np.array(joint["axis"]),
        parent_link=joint["parent_link_id"],
        child_link=joint["child_link_id"]
    ) for joint in data["joints"]]

    return RobotStruct(links, joints)

  def print_structure(self):
      print(f"Robot DOF: {self.dof}")
      print("\nLinks:")
      for link in self.links:
          print(f"  ID: {link.id}, Name: {link.name}, Type: {link.type}")
          print(f"    COG: {link.cog}, Mass: {link.mass}")
          print(f"    Inertia: {link.inertia}, DOF: {link.dof}")

      print("\nJoints:")
      for joint in self.joints:
          print(f"  ID: {joint.id}, Name: {joint.name}, Type: {joint.type}")
          print(f"    Axis: {joint.axis}, Parent Link: {joint.parent_link}, Child Link: {joint.child_link}")
          print(f"    DOF: {joint.dof}")

class LinkStruct:
  def __init__(self, id: int, name: str, cog: np.ndarray, mass: float, inertia: np.ndarray, type: str = "rigid"):
    self.id = id
    self.name = name
    self.type = type
    self.cog = cog
    self.mass = mass
    self.inertia = inertia
    self.dof = self._link_dof(self.type)
  
  @staticmethod
  def _link_dof(type):
    if type == "rigid":
      return 0

class JointStruct:
    def __init__(self, id: int, name: str, type: str, axis: np.ndarray, parent_link: int, child_link: int):
        self.id = id
        self.name = name
        self.type = type
        self.axis = axis if np.linalg.norm(axis) > 0 else np.array([1, 0, 0])
        self.parent_link = parent_link
        self.child_link = child_link
        self.dof = self._joint_dof(self.type)
        self.joint_select_mat = self._joint_select_mat(self.type, self.axis)

    @staticmethod
    def _joint_dof(type: str) -> int:
        if type == "revolution":
            return 1
        elif type == "fix":
            return 0
        else:
            warnings.warn(f"Unsupported joint type: {type}", UserWarning)
            return 0

    @staticmethod
    def _joint_select_mat(type: str, axis: np.ndarray) -> np.ndarray:
        mat = np.zeros((6, 1))
        if type == "fix":
            return mat
        elif type == "revolution":
            mat[0:3, 0] = axis
            return mat
        else:
            raise warnings.warn(f"Unsupported joint type: {type}", UserWarning)