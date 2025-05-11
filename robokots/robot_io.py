#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.11 Created by T.Ishigaki

import json
from typing import Dict

from .basic.robot import *
from .basic.target import *

def io_load_json(file_path: str) -> Dict:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

def io_save_json(data: Dict, file_path: str):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        raise IOError(f"Failed to write JSON file: {e}")

def io_from_json_file(file_path: str) -> "RobotStruct":
    data = io_load_json(file_path)
    return RobotStruct.from_dict(data)

def io_print_structure(robot : RobotStruct):
    robot.print()
        
def io_from_target_json(file_path: str) -> "TargetList":
    data = io_load_json(file_path)
    return TargetList.from_dict(data)

def io_print_targets(t_list : TargetList):
    t_list.print()