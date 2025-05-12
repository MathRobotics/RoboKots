# -*- coding: utf-8 -*-

import re
import numpy as np
from mathrobo import SE3, CMTM, SO3

def extract_state_keys(state: dict, prefix: str = "") -> list:
    """
    Extracts the suffix part of keys from a dictionary that match known physical quantities.

    Args:
        state (dict): Dictionary with string keys.
        prefix (str): Common prefix to strip from keys (e.g., "link").

    Returns:
        list: Suffixes such as "pos", "rot", "vel", etc.
    """
    keywords = ("pos", "rot", "vel", "acc", "acc_diff", "jerk", "snap")
    result = []

    for k in state.keys():
        # If prefix is given, remove it (e.g., "link_" from "link_pos")
        if prefix and k.startswith(prefix + "_"):
            suffix = k[len(prefix) + 1:]
        else:
            suffix = k

        # Check if the suffix matches any keyword
        if any(kw in suffix for kw in keywords):
            result.append(suffix)

    return result

def count_dict_order(state: dict) -> int:
    """
    Determines the highest derivative order present in the state's keys.

    Recognized keywords and their corresponding derivative orders:
        - "pos", "rot"      → 1st order (position, orientation)
        - "vel"             → 2nd order (velocity)
        - "acc"             → 3rd order (acceleration)
        - "jerk"            → 4th order (jerk)
        - "snap"            → 5th order (snap)
        - "acc_diffN"       → (3 + N)th order

    Args:
        state (dict): Dictionary containing state variable keys.

    Returns:
        int: The maximum derivative order detected in the keys.
    """
    # Mapping from keyword to corresponding derivative order
    keyword_order_map = {
        "pos": 1,
        "rot": 1,
        "vel": 2,
        "acc": 3,
        "jerk": 4,
        "snap": 5
    }

    keys = extract_state_keys(state)
    max_order = 0

    for k in keys:
        # Handle special case like "acc_diff1", "acc_diff2", etc.
        match = re.search(r"acc_diff(\d+)", k)
        if match:
            diff_order = int(match.group(1))
            max_order = max(max_order, 3 + diff_order)
            continue  # Skip other checks if acc_diff matched

        # Match against known keywords
        for keyword, order in keyword_order_map.items():
            if keyword in k:
                max_order = max(max_order, order)
                break  # Stop checking once a match is found

    return max_order

def cmtm_to_state_dict(cmtm : CMTM, name : str) -> dict:
  '''
  Convert CMTM to state data
  Args:
    cmtm (CMTM): CMTM object
    name (str): name of the link
  Returns:
    dict: state data
  '''
  state = []

  order = cmtm._n

  mat = cmtm.elem_mat()
  pos = mat[:3,3]
  rot_vec = mat[:3,:3].ravel()
  state.append((name+"_pos" , pos.tolist()))
  state.append((name+"_rot" , rot_vec.tolist()))
  if order > 1:
    veloc = cmtm.elem_vecs(0)
    state.append((name+"_vel" , veloc.tolist()))
  if order > 2:
    accel = cmtm.elem_vecs(1)
    state.append((name+"_acc" , accel.tolist()))
  if order > 3:
    for i in range(order-3):
      vec = cmtm.elem_vecs(i+2)
      state.append((name+"_acc_diff"+str(i+1) , vec.tolist()))
  
  return state

def state_dict_to_rot(state : dict, name : str) -> np.ndarray:
    '''
    Convert state data to rotation matrix
    Args:
        state (dict): state data
        name (str): name of the link
    Returns:
        np.ndarray: rotation matrix
    '''
    rot_vec = np.array(state[name+"_rot"])
    rot = rot_vec.reshape(3,3)

    return rot

def state_dict_to_frame(state : dict, name : str) -> SE3:
    '''
    Convert state data to SE3
    Args:
        state (dict): state data
        name (str): name of the link
    Returns:
        SE3: SE3 object
    '''
    pos = np.array(state[name+"_pos"])
    rot = state_dict_to_rot(state, name)
    mat = SE3(rot, pos)

    return mat

def state_dict_to_cmtm(state : dict, name : str) -> CMTM:
    '''
    Convert state data to CMTM
    Args:
        state (dict): state data
        name (str): name of the link
    Returns:
        CMTM: CMTM object
    '''

    order = count_dict_order(state)
    
    if order < 1:
        raise ValueError("order must be over 1")

    vec = np.zeros((order-1, 6))

    if order > 0:
        mat = state_dict_to_frame(state, name)
    if order > 1:
        vec[0] = np.array(state[name+"_vel"])
    if order > 2:
        vec[1] = np.array(state[name+"_acc"])
    if order > 3:
        for i in range(order-3):
            vec[i+2] = np.array(state[name+"_acc_diff"+str(i+1)])
    
    cmtm = CMTM[SE3](mat, vec)

    return cmtm

def extract_dict_link_info(state : dict, type : str, link_name : str, frame = "dummy", rel_frame = 'dummy'):
    if type == "pos":
      return np.array(state[link_name+"_pos"])
    elif type == "rot":
      return state_dict_to_rot(state, link_name)
    elif type == "vel":
      return np.array(state[link_name+"_vel"])
    elif type == "acc":
      return np.array(state[link_name+"_acc"])
    elif type == "frame":
      return state_dict_to_frame(state, link_name)
    elif type == "cmtm":
      return state_dict_to_cmtm(state, link_name)
    else:
      raise ValueError(f"Invalid type: {set(type)}")
    
def extract_dict_joint_info(data : dict, type : str, joint_name : str, frame = "dummy", rel_frame = 'dummy'):
    'dummy'
    
def extract_dict_total_info(data : dict, type : str, name : str, frame = "dummy", rel_frame = 'dummy'):
    'dummy'

def extract_dict_info(data : dict, data_type : str, group : str, name : str, frame = "dummy", rel_frame = 'dummy'):
    '''
    group : str
      link
      joint
      total
    type : str
      pos
      rot
      vel
      acc
      frame
    name : str
      link name or joint name
    '''

    if group == "link":
      return extract_dict_link_info(data, data_type, name, frame, rel_frame)
    elif group == "joint":
      return extract_dict_joint_info(data, data_type, name, frame, rel_frame)
    elif group == "total":
      return extract_dict_total_info(data, data_type, name, frame, rel_frame)
    else:
      raise ValueError(f"Invalid group: {set(group)}")

def sub_state_dict_vec(data0 : dict, data1 : dict, type : str, name : str) -> dict:
    '''
    data0 : dict
        state data 0
    data1 : dict
        state data 1
    type : str
        pos
        rot
        vel
        acc
        frame
        cmtm
    name : str
        link name or joint name
    '''

    if type == "pos":
        return data1[type] - data0[type]
    elif type == "rot":
        return SO3.sub_tan_vec(data0[type], data1[type], "bframe")
    elif type == "vel":
        return data1[type] - data0[type]
    elif type == "acc":
        return data1[type] - data0[type]
    elif type == "frame":
        return SE3.sub_tan_vec(data0[type], data1[type], "bframe")
    elif type == "cmtm":
        return CMTM.sub_vec(data0[type], data1[type], "bframe")
    else:
        raise ValueError(f"Invalid type: {set(type)}")