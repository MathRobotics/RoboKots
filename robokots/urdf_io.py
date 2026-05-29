#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2026.02.13 Created by Codex

import xml.etree.ElementTree as ET
import heapq
from math import cos, sin
from typing import Dict, List


def _parse_xyz(text: str | None, default: List[float]) -> List[float]:
    if text is None or text.strip() == "":
        return list(default)
    vals = [float(v) for v in text.strip().split()]
    if len(vals) != 3:
        raise ValueError(f"Expected 3 values, got {len(vals)}: '{text}'")
    return vals


def _rpy_to_quaternion_wxyz(rpy: List[float]) -> List[float]:
    roll, pitch, yaw = rpy
    cr = cos(roll * 0.5)
    sr = sin(roll * 0.5)
    cp = cos(pitch * 0.5)
    sp = sin(pitch * 0.5)
    cy = cos(yaw * 0.5)
    sy = sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [w, x, y, z]


def _topologically_order_joints(joints: List[Dict]) -> List[Dict]:
    if not joints:
        return []

    child_link_to_joint_idx: Dict[int, int] = {}
    for idx, joint in enumerate(joints):
        child_link_id = int(joint["child_link_id"])
        if child_link_id in child_link_to_joint_idx:
            prev_joint = joints[child_link_to_joint_idx[child_link_id]]["name"]
            raise ValueError(
                f"Invalid URDF tree: child link id {child_link_id} is attached by both "
                f"'{prev_joint}' and '{joint['name']}'."
            )
        child_link_to_joint_idx[child_link_id] = idx

    children: List[List[int]] = [[] for _ in joints]
    indegree = [0] * len(joints)
    for idx, joint in enumerate(joints):
        parent_link_id = int(joint["parent_link_id"])
        parent_joint_idx = child_link_to_joint_idx.get(parent_link_id)
        if parent_joint_idx is None:
            continue
        indegree[idx] = 1
        children[parent_joint_idx].append(idx)

    ready = [idx for idx, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)

    ordered: List[Dict] = []
    while ready:
        idx = heapq.heappop(ready)
        ordered.append(joints[idx])
        for child_idx in children[idx]:
            indegree[child_idx] -= 1
            if indegree[child_idx] == 0:
                heapq.heappush(ready, child_idx)

    if len(ordered) != len(joints):
        raise ValueError("Invalid URDF tree: joint graph must be acyclic and connected through parent links.")

    return ordered


def urdf_root_to_model_data(root: ET.Element, add_world_link: bool = True) -> Dict:
    if root.tag != "robot":
        raise ValueError(f"URDF root tag must be 'robot', got '{root.tag}'")

    link_elems = root.findall("link")
    if len(link_elems) == 0:
        raise ValueError("URDF must contain at least one <link> element.")
    urdf_link_names = [link_elem.attrib.get("name") for link_elem in link_elems]
    if add_world_link and "world" in urdf_link_names:
        raise ValueError(
            "URDF already contains link name 'world'. "
            "Set add_world_link=False to avoid duplicate world links."
        )

    links: List[Dict] = []
    link_name_to_id: Dict[str, int] = {}

    if add_world_link:
        links.append({"id": 0, "name": "world"})

    base_link_id = len(links)
    for i, link_elem in enumerate(link_elems):
        name = link_elem.attrib.get("name")
        if not name:
            raise ValueError("Each <link> must have a non-empty 'name' attribute.")
        if name in link_name_to_id:
            raise ValueError(f"Duplicate link name in URDF: '{name}'")

        link_id = base_link_id + i
        link_name_to_id[name] = link_id
        link_data: Dict = {"id": link_id, "name": name}

        inertial = link_elem.find("inertial")
        if inertial is not None:
            mass_elem = inertial.find("mass")
            if mass_elem is not None and "value" in mass_elem.attrib:
                link_data["mass"] = float(mass_elem.attrib["value"])

            origin_elem = inertial.find("origin")
            if origin_elem is not None:
                link_data["cog"] = _parse_xyz(origin_elem.attrib.get("xyz"), [0.0, 0.0, 0.0])

            inertia_elem = inertial.find("inertia")
            if inertia_elem is not None:
                ixx = float(inertia_elem.attrib.get("ixx", 1.0))
                ixy = float(inertia_elem.attrib.get("ixy", 0.0))
                ixz = float(inertia_elem.attrib.get("ixz", 0.0))
                iyy = float(inertia_elem.attrib.get("iyy", 1.0))
                iyz = float(inertia_elem.attrib.get("iyz", 0.0))
                izz = float(inertia_elem.attrib.get("izz", 1.0))
                # RoboKots inertia order: [ixx, iyy, izz, ixy, ixz, iyz]
                link_data["inertia"] = [ixx, iyy, izz, ixy, ixz, iyz]

        links.append(link_data)

    parsed_joints: List[Dict] = []
    urdf_type_to_kots = {
        "revolute": "revolute",
        "continuous": "revolute",
        "fixed": "fix",
        "prismatic": "prismatic",
    }
    child_link_names = set()

    for joint_elem in root.findall("joint"):
        name = joint_elem.attrib.get("name")
        joint_type = joint_elem.attrib.get("type")
        if not name:
            raise ValueError("Each <joint> must have a non-empty 'name' attribute.")
        if not joint_type:
            raise ValueError(f"Joint '{name}' is missing the 'type' attribute.")
        if joint_type not in urdf_type_to_kots:
            raise ValueError(
                f"Unsupported URDF joint type '{joint_type}' for joint '{name}'. "
                "Supported types: revolute, continuous, fixed, prismatic."
            )

        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")
        parent_name = None if parent_elem is None else parent_elem.attrib.get("link")
        child_name = None if child_elem is None else child_elem.attrib.get("link")
        if parent_name is None or child_name is None:
            raise ValueError(f"Joint '{name}' must have both <parent link='...'> and <child link='...'>.")
        if parent_name not in link_name_to_id:
            raise ValueError(f"Joint '{name}' references unknown parent link '{parent_name}'.")
        if child_name not in link_name_to_id:
            raise ValueError(f"Joint '{name}' references unknown child link '{child_name}'.")

        child_link_names.add(child_name)

        origin_elem = joint_elem.find("origin")
        pos = [0.0, 0.0, 0.0]
        quat = [1.0, 0.0, 0.0, 0.0]
        if origin_elem is not None:
            pos = _parse_xyz(origin_elem.attrib.get("xyz"), [0.0, 0.0, 0.0])
            rpy = _parse_xyz(origin_elem.attrib.get("rpy"), [0.0, 0.0, 0.0])
            quat = _rpy_to_quaternion_wxyz(rpy)

        kots_joint_type = urdf_type_to_kots[joint_type]
        joint_data: Dict = {
            "name": name,
            "type": kots_joint_type,
            "parent_link_id": link_name_to_id[parent_name],
            "child_link_id": link_name_to_id[child_name],
            "origin": {"position": pos, "orientation": quat},
        }

        if kots_joint_type in ("revolute", "prismatic"):
            axis_elem = joint_elem.find("axis")
            axis = _parse_xyz(
                None if axis_elem is None else axis_elem.attrib.get("xyz"),
                [1.0, 0.0, 0.0],
            )
            joint_data["axis"] = axis

        parsed_joints.append(joint_data)

    world_joints: List[Dict] = []
    if add_world_link:
        used_joint_names = {joint["name"] for joint in parsed_joints}
        root_links = [link for link in links if link["name"] != "world" and link["name"] not in child_link_names]
        for root_link in root_links:
            base_name = f"world_to_{root_link['name']}"
            candidate = base_name
            suffix = 1
            while candidate in used_joint_names:
                candidate = f"{base_name}_{suffix}"
                suffix += 1

            used_joint_names.add(candidate)
            world_joints.append(
                {
                    "name": candidate,
                    "type": "fix",
                    "parent_link_id": 0,
                    "child_link_id": int(root_link["id"]),
                    "origin": {
                        "position": [0.0, 0.0, 0.0],
                        "orientation": [1.0, 0.0, 0.0, 0.0],
                    },
                }
            )

    # RoboKots uses joint order as part of its internal joint-base representation.
    # Normalize URDF/XML order into a parent-before-child order while preserving
    # the relative order of siblings from the input file.
    joints = _topologically_order_joints(world_joints + parsed_joints)
    for idx, joint in enumerate(joints):
        joint["id"] = idx

    return {"links": links, "joints": joints}


def urdf_xml_to_model_data(urdf_xml: str, add_world_link: bool = True) -> Dict:
    try:
        root = ET.fromstring(urdf_xml)
    except ET.ParseError as e:
        raise ValueError(f"Invalid URDF format: {e}")
    return urdf_root_to_model_data(root, add_world_link=add_world_link)


def load_urdf_file(file_path: str, add_world_link: bool = True) -> Dict:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            urdf_xml = file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found.")

    return urdf_xml_to_model_data(urdf_xml, add_world_link=add_world_link)
