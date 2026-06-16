#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.11 Created by T.Ishigaki

from __future__ import annotations

import numpy as np

import warnings
from dataclasses import dataclass
from typing import List, Dict

warnings.simplefilter("always", UserWarning)

MODEL_SCHEMA_VERSION = "0.0.2"
SUPPORTED_LINK_TYPES = frozenset({"rigid", "soft"})
SUPPORTED_JOINT_TYPES = frozenset({"fixed", "revolute", "prismatic"})
INERTIA_KEYS = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")


def _array_module(lib: str):
  if lib == "numpy":
    return np
  if lib == "jax":
    import jax.numpy as jnp
    return jnp
  raise ValueError(f"Unsupported library: {lib}. Use 'jax' or 'numpy'.")


def _require_list3(value, field_name: str) -> list[float]:
  if not isinstance(value, (list, tuple)) or len(value) != 3:
    raise ValueError(f"{field_name} must be a list of 3 numbers.")
  try:
    result = [float(v) for v in value]
  except (TypeError, ValueError) as exc:
    raise ValueError(f"{field_name} must be a list of 3 numbers.") from exc
  if not np.all(np.isfinite(result)):
    raise ValueError(f"{field_name} must contain only finite numbers.")
  return result


def _require_list4(value, field_name: str) -> list[float]:
  if not isinstance(value, (list, tuple)) or len(value) != 4:
    raise ValueError(f"{field_name} must be a list of 4 numbers in [w, x, y, z] order.")
  try:
    result = [float(v) for v in value]
  except (TypeError, ValueError) as exc:
    raise ValueError(f"{field_name} must be a list of 4 numbers in [w, x, y, z] order.") from exc
  if not np.all(np.isfinite(result)):
    raise ValueError(f"{field_name} must contain only finite numbers.")
  if np.linalg.norm(result) <= 0.0:
    raise ValueError(f"{field_name} must be non-zero.")
  return result


def inertia_dict_to_vector(inertia: dict | None) -> list[float]:
  if inertia is None:
    return [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
  if not isinstance(inertia, dict):
    raise ValueError("inertia must be a dictionary with keys ixx, ixy, ixz, iyy, iyz, izz.")
  missing = [key for key in INERTIA_KEYS if key not in inertia]
  extra = [key for key in inertia if key not in INERTIA_KEYS]
  if missing:
    raise ValueError("inertia is missing required keys: " + ", ".join(missing))
  if extra:
    raise ValueError("inertia contains unsupported keys: " + ", ".join(extra))
  try:
    values = {key: float(inertia[key]) for key in INERTIA_KEYS}
  except (TypeError, ValueError) as exc:
    raise ValueError("inertia values must be numbers.") from exc
  if not np.all(np.isfinite(list(values.values()))):
    raise ValueError("inertia values must be finite numbers.")
  return [values["ixx"], values["iyy"], values["izz"], values["ixy"], values["ixz"], values["iyz"]]


def inertia_vector_to_dict(inertia) -> dict[str, float]:
  if inertia is None:
    values = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
  else:
    values = inertia.tolist() if hasattr(inertia, "tolist") else list(inertia)
    if len(values) != 6:
      values = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
  return {
    "ixx": float(values[0]),
    "ixy": float(values[3]),
    "ixz": float(values[4]),
    "iyy": float(values[1]),
    "iyz": float(values[5]),
    "izz": float(values[2]),
  }


def _validate_id_set(items: list[dict], item_type: str) -> None:
  ids: list[int] = []
  for i, item in enumerate(items):
    if not isinstance(item, dict):
      raise ValueError(f"{item_type}[{i}] must be a dictionary.")
    item_id = item.get("id")
    if not isinstance(item_id, int):
      raise ValueError(f"{item_type}[{i}].id must be an integer.")
    ids.append(item_id)
  expected = list(range(len(items)))
  if sorted(ids) != expected:
    raise ValueError(f"{item_type}.id values must be unique contiguous integers 0..{len(items) - 1}.")


def _validate_unique_names(items: list[dict], item_type: str) -> None:
  seen: set[str] = set()
  for i, item in enumerate(items):
    name = item.get("name")
    if not isinstance(name, str) or name == "":
      raise ValueError(f"{item_type}[{i}].name must be a non-empty string.")
    if name in seen:
      raise ValueError(f"Duplicate {item_type} name: '{name}'.")
    seen.add(name)


def _validate_joint_link_references(links: list[dict], joints: list[dict]) -> None:
  link_num = len(links)
  for joint in joints:
    joint_name = joint["name"]
    parent = joint.get("parent_link_id")
    child = joint.get("child_link_id")
    if not isinstance(parent, int) or not 0 <= parent < link_num:
      raise ValueError(f"Joint '{joint_name}' has invalid parent_link_id: {parent}.")
    if not isinstance(child, int) or not 0 <= child < link_num:
      raise ValueError(f"Joint '{joint_name}' has invalid child_link_id: {child}.")
    if parent == child:
      raise ValueError(f"Joint '{joint_name}' cannot connect a link to itself.")


def _validate_robotstruct_supported_topology(links: list[dict], joints: list[dict]) -> None:
  link_num = len(links)
  children_by_link = [[] for _ in range(link_num)]
  parent_joint_by_child: dict[int, str] = {}

  for joint in joints:
    joint_name = joint["name"]
    parent = joint["parent_link_id"]
    child = joint["child_link_id"]
    if child in parent_joint_by_child:
      raise NotImplementedError(
        "RobotStruct currently supports only tree topology; "
        f"link id {child} is attached by both '{parent_joint_by_child[child]}' and '{joint_name}'."
      )
    parent_joint_by_child[child] = joint_name
    children_by_link[parent].append(child)

  world_ids = [link["id"] for link in links if link["name"] == "world"]
  if world_ids:
    roots = world_ids
  else:
    roots = [link["id"] for link in links if link["id"] not in parent_joint_by_child]
    if len(roots) != 1:
      raise NotImplementedError("RobotStruct currently supports only tree topology with exactly one root link.")

  visiting: set[int] = set()
  visited: set[int] = set()

  def visit(link_id: int) -> None:
    if link_id in visiting:
      raise NotImplementedError("RobotStruct currently supports only acyclic tree topology.")
    if link_id in visited:
      return
    visiting.add(link_id)
    for child_id in children_by_link[link_id]:
      visit(child_id)
    visiting.remove(link_id)
    visited.add(link_id)

  for root in roots:
    visit(root)

  if len(visited) != link_num:
    unreachable = [link["name"] for link in links if link["id"] not in visited]
    if world_ids:
      raise NotImplementedError(
        "RobotStruct currently supports only topologies reachable from the 'world' link; "
        "unreachable links: " + ", ".join(unreachable)
      )
    raise NotImplementedError(
      "RobotStruct currently supports only topologies reachable from the root link; "
      "unreachable links: " + ", ".join(unreachable)
    )


def validate_model_data(data: Dict) -> None:
  """Validate canonical RoboKots model JSON data."""
  schema_version = data.get("schema_version")
  if schema_version != MODEL_SCHEMA_VERSION:
    raise ValueError(
      f"model_data.schema_version must be '{MODEL_SCHEMA_VERSION}', got {schema_version!r}."
    )

  links = data.get("links")
  joints = data.get("joints")
  if not isinstance(links, list) or len(links) == 0:
    raise ValueError("model_data.links must be a non-empty list.")
  if not isinstance(joints, list):
    raise ValueError("model_data.joints must be a list.")

  _validate_id_set(links, "links")
  _validate_id_set(joints, "joints")
  _validate_unique_names(links, "link")
  _validate_unique_names(joints, "joint")

  for link in links:
    link_type = link.get("type", "rigid")
    if link_type not in SUPPORTED_LINK_TYPES:
      raise ValueError(f"Unsupported link type '{link_type}' for link '{link['name']}'.")
    if "cog" in link:
      _require_list3(link["cog"], f"link '{link['name']}'.cog")
    if "inertia" in link:
      try:
        inertia_dict_to_vector(link["inertia"])
      except ValueError as exc:
        raise ValueError(f"link '{link['name']}'.{exc}") from exc

  for joint in joints:
    joint_type = joint.get("type")
    joint_name = joint["name"]
    if joint_type == "fix":
      raise ValueError(f"Unsupported joint type 'fix' for joint '{joint_name}'. Use 'fixed' instead.")
    if joint_type not in SUPPORTED_JOINT_TYPES:
      raise ValueError(f"Unsupported joint type '{joint_type}' for joint '{joint_name}'.")
    if joint_type in ("revolute", "prismatic"):
      if "axis" not in joint:
        raise ValueError(f"joint '{joint_name}'.axis is required for {joint_type} joints.")
      axis = _require_list3(joint["axis"], f"joint '{joint_name}'.axis")
      if np.linalg.norm(axis) <= 0.0:
        raise ValueError(f"joint '{joint_name}'.axis must be non-zero.")
    origin = joint.get("origin", {})
    if origin is not None:
      if not isinstance(origin, dict):
        raise ValueError(f"joint '{joint_name}'.origin must be a dictionary.")
      if "position" in origin:
        _require_list3(origin["position"], f"joint '{joint_name}'.origin.position")
      if "orientation" in origin:
        _require_list4(origin["orientation"], f"joint '{joint_name}'.origin.orientation")

  _validate_joint_link_references(links, joints)


@dataclass(frozen=True)
class MotionOwner:
  dof: int
  dof_index: int


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
    self._links_by_name: Dict[str, "LinkStruct"] = {}
    self._joints_by_name: Dict[str, "JointStruct"] = {}
    self.robot_init()
    
  def link(self, name : str) -> "LinkStruct":
    return self._links_by_name.get(name)
  
  def link_list(self, name_list : list[str]) -> List["LinkStruct"]:
    return [self._links_by_name.get(name) for name in name_list]
  
  def is_link(self, name : str) -> bool:
    return name in self._links_by_name

  def joint(self, name : str) -> "JointStruct":
    return self._joints_by_name.get(name)
  
  def joint_list(self, name_list : list[str]) -> List["JointStruct"]:
    return [self._joints_by_name.get(name) for name in name_list]
  
  def is_joint(self, name : str) -> bool:
    return name in self._joints_by_name

  def motion_owners(self) -> tuple[MotionOwner, ...]:
    owners = [owner for owner in (*self.links, *self.joints) if owner.dof > 0]
    owners.sort(key=lambda owner: owner.dof_index)
    return tuple(MotionOwner(owner.dof, owner.dof_index) for owner in owners)

  def motion_owner_dofs(self) -> list[int]:
    return [owner.dof for owner in self.motion_owners()]

  def robot_init(self):
    self.joint_num = len(self.joints)  
    self.link_num = len(self.links)  

    self.dof = 0
    self.joint_dof = 0
    self.link_dof = 0
    
    dof_index = 0
    
    for l in self.links:
      l.child_joint_ids = []
      l.parent_joint_ids = []
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
    self._links_by_name = {l.name: l for l in self.links}
    self._joints_by_name = {j.name: j for j in self.joints}
    
  def route_target_link(self, target_link : "LinkStruct", link_route : List, joint_route : List):
    link_route.append(target_link.id)
    for joint_id in target_link.parent_joint_ids:
      self.route_target_joint(self.joints[joint_id], link_route, joint_route)
  
  def route_target_joint(self, target_joint : "JointStruct", link_route : List, joint_route : List):
    joint_route.append(target_joint.id)
    self.route_target_link(self.links[target_joint.parent_link_id], link_route, joint_route)

  def route_end_links(self, target_link: "LinkStruct", link_route: List, joint_route: List):
    link_route.append(target_link.id)
    for joint_id in target_link.child_joint_ids:
      self.route_end_joints(self.joints[joint_id], link_route, joint_route)

  def route_end_joints(self, target_joint: "JointStruct", link_route: List, joint_route: List):
    joint_route.append(target_joint.id)
    self.route_end_links(self.links[target_joint.child_link_id], link_route, joint_route)
    
  @staticmethod
  def from_dict(data: Dict, lib: str = "numpy") -> "RobotStruct":  
    from mathrobo import SE3

    if not isinstance(data, dict):
        raise ValueError("Input data must be a dictionary.")
    validate_model_data(data)
    _validate_robotstruct_supported_topology(data["links"], data["joints"])
    sorted_links = sorted(data["links"], key=lambda link: link["id"])
    sorted_joints = sorted(data["joints"], key=lambda joint: joint["id"])
    
    joints = []
    links = []

    xp = _array_module(lib)
    
    links = [LinkStruct(
        id=link["id"],
        name=link["name"],
        cog=xp.array(link.get("cog", [0., 0., 0.])),
        mass=float(link.get("mass", 0.)),
        inertia=xp.array(inertia_dict_to_vector(link.get("inertia"))),
        type=link.get("type", "rigid"),
        length=float(link.get("length", 0.0)),
        lib=lib
    ) for link in sorted_links]

    joints = [JointStruct(
        id=joint["id"],
        name=joint["name"],
        type=joint["type"],
        axis=xp.array(joint.get("axis", [0., 0., 0.])),
        parent_link_id=joint["parent_link_id"],
        child_link_id=joint["child_link_id"],
        origin=SE3.set_pos_quaternion(
          xp.array(joint.get("origin", {}).get("position", [0., 0., 0.])),
          xp.array(joint.get("origin", {}).get("orientation", [1., 0., 0., 0.])),
          LIB=lib
        ),
        lib=lib
    ) for joint in sorted_joints]

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

        link_dict["inertia"] = inertia_vector_to_dict(link.inertia)
        link_dict["length"] = float(link.length) if link.length is not None else 0.0

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
        "schema_version": MODEL_SCHEMA_VERSION,
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
          print(f"    DOF:{link.dof}")
          print(f"    DOF index: {link.dof_index}")
          print(f"    Length: {link.length}\n")

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
  def __init__(self, id: int, name: str, cog: np.ndarray, mass: float, inertia: np.ndarray, type: str = "rigid", length: float = None, lib: str = "numpy"):
    self.lib = lib
    self.id = id
    self.name = name
    self.type = type
    self.cog = cog
    self.mass = mass
    self.inertia = inertia
    self.length = length if length is not None else 1.0  # Default length if not specified
    self.dof = self._link_dof(self.type)
    self.select_mat = self._select_mat(self.type, lib)
    self.select_indeces = np.argmax(self.select_mat, axis=0)
    self.origin_coord = np.array([0., 0., 0., 0., 0., 1.])
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
    elif type == "soft":
      return 6
    else:
        warnings.warn(f"Unsupported link type: {type}", UserWarning)
        return 0

  #specific for rigid link or soft link
  @staticmethod
  def _select_mat(type: str, lib: str = "numpy") -> np.ndarray:
      xp = _array_module(lib)
      mat = xp.zeros((6, 1))
      if type == "rigid":
          return mat
      elif type == "soft":
          mat = xp.eye(6)
          return mat
      else:
          warnings.warn(f"Unsupported link type: {type}", UserWarning)

class JointStruct:
    dof_index : int = 0
    def __init__(self, id: int, name: str, type: str, axis: np.ndarray, parent_link_id: int, child_link_id: int, origin: SE3, lib: str = "numpy"):
        xp = _array_module(lib)
        if type == "fix":
            raise ValueError(
                f"Unsupported joint type 'fix' for joint '{name}'. "
                "Use 'fixed' instead."
            )
        self.id = id
        self.name = name
        self.type = type
        self.axis = axis if xp.linalg.norm(axis) > 0 else xp.array([1, 0, 0])
        self.parent_link_id = parent_link_id
        self.child_link_id = child_link_id
        self.dof = self._joint_dof(self.type)
        self.select_mat = self._select_mat(self.type, self.axis, lib)
        self.select_indeces = xp.argmax(self.select_mat, axis=0)
        self.origin = origin
        
    def set_dof_index(self, n : int):
      if n < 0:
        raise ValueError(f"Invalid DOF index: {n}")
      self.dof_index = n

    @staticmethod
    def _joint_dof(type: str) -> int:
        if type == "revolute":
            return 1
        elif type == "prismatic":
            return 1
        elif type == "fixed":
            return 0
        else:
            warnings.warn(f"Unsupported joint type: {type}", UserWarning)
            return 0

    #specific for 1 DOF joint or fixed joint
    @staticmethod
    def _select_mat(type: str, axis: np.ndarray, lib: str = "numpy") -> np.ndarray:
        xp = _array_module(lib)
        mat = xp.zeros((6, 1))

        if type == "fixed":
            return mat
        elif type == "revolute":
            if lib == "jax":
                mat = mat.at[0:3, 0].set(axis)
            elif lib == "numpy":
                mat[0:3, 0] = axis
            return mat
        elif type == "prismatic":
            if lib == "jax":
                mat = mat.at[3:6, 0].set(axis)
            elif lib == "numpy":
                mat[3:6, 0] = axis
            return mat
        else:
            raise warnings.warn(f"Unsupported joint type: {type}", UserWarning)
        
    def selector(self, mat: np.ndarray) -> np.ndarray:
        return mat[:, self.select_indeces]
    
    #specific for 3D space (magic number 6)
    def scatter(self, mat: np.ndarray) -> np.ndarray:
        result = np.zeros((6, mat.shape[1]))
        if mat.shape[1] != self.dof:
            raise ValueError(f"Invalid input vector length: {len(mat)}")
        for i in range(self.dof):
            row = self.select_indeces[i]
            print(f"row: {row}")
            result[row] += mat[i]
        return result

        
