#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.11 Created by T.Ishigaki

import numpy as np

import warnings
from typing import List, Dict

from mathrobo import SE3

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
    
  def link(self, name : str) -> "LinkStruct":
    for l in self.links:
      if name == l.name:
        return l
    ValueError(f"Invalid link name: {name}")
    return None
  
  def link_list(self, name_list : list[str]) -> List["LinkStruct"]:
    link_list = []
    for name in name_list:
      link_list.append(self.link(name))
    return link_list

  def joint(self, name : str) -> "JointStruct":
    for l in self.joints:
      if name == l.name:
        return l
    ValueError(f"Invalid joint name: {name}")
    return None
  
  def joint_list(self, name_list : list[str]) -> List["JointStruct"]:
    joint_list = []
    for name in name_list:
      joint_list.append(self.joint(name))
    return joint_list

  def robot_init(self):
    self.joint_num = len(self.joints)  
    self.link_num = len(self.links)  

    self.dof = 0
    self.joint_dof = 0
    self.link_dof = 0
    
    dof_index = 0
    
    for l in self.links:
      l.set_dof_index(dof_index)
      dof_index += l.dof
      self.link_dof += l.dof
    
    for j in self.joints:
      j.set_dof_index(dof_index)
      dof_index += j.dof
      self.joint_dof += j.dof
      
      self.links[j.parent_link_id].child_joint_ids.append(j.id)
      self.links[j.child_link_id].parent_joint_ids.append(j.id)
      
    self.dof = self.joint_dof + self.link_dof
    
    self.link_names = [l.name for l in self.links]
    self.joint_names = [j.name for j in self.joints]
    
  def route_target_link(self, target_link : "LinkStruct", link_route : List, joint_route : List):
    link_route.append(target_link.id)
    for joint_id in target_link.parent_joint_ids:
      self.route_target_joint(self.joints[joint_id], link_route, joint_route)
  
  def route_target_joint(self, target_joint : "JointStruct", link_route : List, joint_route : List):
    joint_route.append(target_joint.id)
    self.route_target_link(self.links[target_joint.parent_link_id], link_route, joint_route)
    
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
        axis=np.array(joint.get("axis", [0., 0., 0.])),
        parent_link_id=joint["parent_link_id"],
        child_link_id=joint["child_link_id"],
        origin=SE3.set_pos_quaternion(
          joint.get("origin", {}).get("position", [0., 0., 0.]),
          joint.get("origin", {}).get("orientation", [1., 0., 0., 0.])
        )
    ) for joint in data["joints"]]

    return RobotStruct(links, joints)
  
  def to_dict(self) -> Dict:
    links_array = []
    for link in self.links:
        link_dict = {}
        link_dict["id"] = link.id
        link_dict["name"] = link.name
        link_dict["type"] = link.type
        
        link_dict["mass"] = float(link.mass)
        link_dict["cog"] = link.cog.tolist() if link.cog is not None else [0.0, 0.0, 0.0]

        if link.inertia is not None and link.inertia.shape == (6,):
            inertia_list = link.inertia.tolist()
        else:
            inertia_list = [1,1,1,0,0,0]
        link_dict["inertia"] = inertia_list

        link_dict["geometry"] = None

        links_array.append(link_dict)

    joints_array = []
    for joint in self.joints:
        joint_dict = {}
        joint_dict["id"] = joint.id
        joint_dict["name"] = joint.name
        joint_dict["type"] = joint.type

        joint_dict["axis"] = joint.axis.tolist()

        joint_dict["parent_link_id"] = joint.parent_link_id
        joint_dict["child_link_id"] = joint.child_link_id

        pos, quat = joint.origin.pos_quaternion()
        origin_dict = {
            "position": pos,
            "orientation": quat
        }
        joint_dict["origin"] = origin_dict

        joints_array.append(joint_dict)

    return {
        "links": links_array,
        "joints": joints_array
    }
  
  def print(self):
      print(f"Robot DOF: {self.dof}")
      print("\nLinks:")
      for link in self.links:
          print(f"  ID: {link.id}, Name: {link.name}, Type: {link.type}")
          print(f"    COG: {link.cog}, Mass: {link.mass}")
          print(f"    Inertia: {link.inertia}, DOF: {link.dof}")
          print(f"    Connect parent joint: {link.parent_joint_ids}")
          print(f"    Connect child joint: {link.child_joint_ids}")
          print(f"    DOF index: {link.dof_index}")

      print("\nJoints:")
      for joint in self.joints:
          print(f"  ID: {joint.id}, Name: {joint.name}, Type: {joint.type}")
          print(f"    Axis: {joint.axis}, Parent Link: {joint.parent_link_id}, Child Link: {joint.child_link_id}")
          print(f"    DOF: {joint.dof}")
          print(f"    Origin: {joint.origin.pos()}")
          print(f"{joint.origin.rot()}")
          print(f"    DOF index: {joint.dof_index}\n")

class LinkStruct:
  dof_index : int = 0
  def __init__(self, id: int, name: str, cog: np.ndarray, mass: float, inertia: np.ndarray, type: str = "rigid"):
    self.id = id
    self.name = name
    self.type = type
    self.cog = cog
    self.mass = mass
    self.inertia = inertia
    self.dof = self._link_dof(self.type)
    self.child_joint_ids = []
    self.parent_joint_ids = []
    
  def set_dof_index(self, n : int):
    if n < 0:
      raise ValueError(f"Invalid DOF index: {n}")
    self.dof_index = n
  
  @staticmethod
  def _link_dof(type) -> int:
    if type == "rigid":
      return 0

class JointStruct:
    dof_index : int = 0
    def __init__(self, id: int, name: str, type: str, axis: np.ndarray, parent_link_id: int, child_link_id: int, origin: SE3):
        self.id = id
        self.name = name
        self.type = type
        self.axis = axis if np.linalg.norm(axis) > 0 else np.array([1, 0, 0])
        self.parent_link_id = parent_link_id
        self.child_link_id = child_link_id
        self.dof = self._joint_dof(self.type)
        self.select_mat = self._select_mat(self.type, self.axis)
        self.origin = origin
        
    def set_dof_index(self, n : int):
      if n < 0:
        raise ValueError(f"Invalid DOF index: {n}")
      self.dof_index = n

    @staticmethod
    def _joint_dof(type: str) -> int:
        if type == "revolute":
            return 1
        elif type == "fix":
            return 0
        else:
            warnings.warn(f"Unsupported joint type: {type}", UserWarning)
            return 0

    @staticmethod
    def _select_mat(type: str, axis: np.ndarray) -> np.ndarray:
        mat = np.zeros((6, 1))
        if type == "fix":
            return mat
        elif type == "revolute":
            mat[0:3, 0] = axis
            return mat
        else:
            raise warnings.warn(f"Unsupported joint type: {type}", UserWarning)