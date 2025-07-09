#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.11 Created by T.Ishigaki

import json
from typing import Dict

from .basic.robot import RobotStruct
from .basic.target import TargetList

def load_json_file(file_path: str) -> Dict:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

def save_json_file(data: Dict, file_path: str):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        raise IOError(f"Failed to write JSON file: {e}")

def load_robot_json(data : Dict) -> "RobotStruct":
    if not isinstance(data, dict):
        raise ValueError("Input data must be a dictionary.")
    return RobotStruct.from_dict(data)

def load_robot_json_file(file_path: str) -> "RobotStruct":
    data = load_json_file(file_path)
    return load_robot_json(data)

def print_robot_structure(robot : RobotStruct):
    robot.print()

def load_target_json(data: Dict) -> "TargetList":
    if not isinstance(data, dict):
        raise ValueError("Input data must be a dictionary.")
    return TargetList.from_dict(data)
        
def load_target_json_file(file_path: str) -> "TargetList":
    data = load_json_file(file_path)
    return load_target_json(data)

def print_target_list(t_list : TargetList):
    t_list.print()