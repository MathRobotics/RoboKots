# -*- coding: utf-8 -*-

import re
import numpy as np
from mathrobo import CMVector
from mathrobo import SE3, SE3wrench, CMTM, CMTM, SO3

from .state import keys, keys_order, keys_time_order, keys_name

def extract_state_keys(state: dict, prefix: str = "") -> list:
    """
    Extracts the suffix part of keys from a dictionary that match known physical quantities.

    Args:
        state (dict): Dictionary with string keys.
        prefix (str): Common prefix to strip from keys (e.g., "link").

    Returns:
        list: Suffixes such as "pos", "rot", "vel", etc.
    """
  
    result = []

    for k in state.keys():
        # If prefix is given, remove it (e.g., "link_" from "link_pos")
        if prefix and k.startswith(prefix + "_"):
            suffix = k[len(prefix) + 1:]
        else:
            suffix = k

        # Check if the suffix matches any keyword
        if any(kw in suffix for kw in keys):
            result.append(suffix)

    return result

def count_dict_time_order(state: dict) -> int:
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

    keys = extract_state_keys(state)
    max_order = 0

    for k in keys:
        # Handle special case like "acc_diff1", "acc_diff2", etc.
        match = re.search(r"acc_diff(\d+)", k)
        if match:
            diff_order = int(match.group(1))
            max_order = max(max_order, keys_time_order["acc"] + diff_order)
            continue  # Skip other checks if acc_diff matched

        # Match against known keywords
        for keyword, order in keys_time_order.items():
            if keyword in k:
                max_order = max(max_order, order)
                break  # Stop checking once a match is found

    return max_order

def cmtm_to_state_list(cmtm : CMTM, name : str) -> list:
  '''
  Convert CMTM to state data
  Args:
    cmtm (CMTM): CMTM object
    name (str): name of the link
  Returns:
    list: state data
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
    for i in range(order-keys_order["acc"]):
      vec = cmtm.elem_vecs(i+2)
      state.append((name+"_acc_diff"+str(i+1) , vec.tolist()))
  
  return state

def vecs_to_state_dict(vec : np.ndarray, name : str, type_name : str, order : int) -> list:
    '''
    Convert vector data to state data
    Args:
        vecs (np.ndarray): vector data
        name (str): name of the link or joint
        type_name (str): type of the vector (e.g., "link", "joint")
        vec_dof (int): dimension of the vector
    Returns:
        list: state data
    '''
    state = []

    if order > 0:
        vec_dof = vec.size // order
    else:
        raise ValueError("order must be greater than 0")

    if vec.size == 0 or vec_dof == 0:
        return [
            (
                f"{name}_{type_name}" if i == 0 else f"{name}_{type_name}_diff{i}", []
            )
            for i in range(order)
        ]

    vecs = vec.reshape(-1, vec_dof)

    state = [
        (
            f"{name}_{type_name}" if i == 0 else f"{name}_{type_name}_diff{i}",
            row.tolist(),
        )
        for i, row in enumerate(vecs)
    ]

    return state

def state_dict_to_link_pos(state : dict, name : str) -> np.ndarray:
    '''
    Convert state data to link position
    Args:
        state (dict): state data
        name (str): name of the link
    Returns:
        np.ndarray: position vector
    '''
    pos = np.array(state[name+"_pos"])

    return pos

def state_dict_to_links_pos(state : dict, link_names : list) -> np.ndarray:
    '''
    Convert state data to link positions
    Args:
        state (dict): state data
        link_names (list): list of link names
    Returns:
        np.ndarray: array of positions
    '''
    pos_list = []
    for name in link_names:
        pos = state_dict_to_link_pos(state, name)
        pos_list.append(pos)
    
    return np.array(pos_list)

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

def state_dict_to_frame_wrench(state : dict, name : str) -> SE3wrench:
    '''
    Convert state data to SE3wrench
    Args:
        state (dict): state data
        name (str): name of the link
    Returns:
        SE3wrench: SE3wrench object
    '''
    pos = np.array(state[name+"_pos"])
    rot = state_dict_to_rot(state, name)
    mat = SE3wrench(rot, pos)

    return mat

def __state_dict_to_cmtm_values(state : dict, name : str, order = None) -> tuple[SE3, np.ndarray]:
    '''
    Convert state data to CMTM values
    Args:
        state (dict): state data
        name (str): name of the link
    Returns:
        tuple: (SE3, np.ndarray)
    '''
    if order is None:
        order = count_dict_time_order(state)
    
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
        for i in range(order-keys_order["acc"]):
            vec[i+2] = np.array(state[name+"_acc_diff"+str(i+1)])
    
    return mat, vec

def state_dict_to_cmtm(state : dict, name : str, order = None) -> CMTM:
    '''
    Convert state data to CMTM
    Args:
        state (dict): state data
        name (str): name of the link
    Returns:
        CMTM: CMTM object
    '''
    mat, vec = __state_dict_to_cmtm_values(state, name, order)

    cmtm = CMTM[SE3](mat, vec)

    return cmtm

def state_dict_to_cmtm_wrench(state : dict, name : str, order = None) -> CMTM:
    '''
    Convert state data to CMTM
    Args:
        state (dict): state data
        name (str): name of the link
    Returns:
        CMTM: CMTM object
    '''
    mat, vec = __state_dict_to_cmtm_values(state, name, order)

    cmtm = CMTM[SE3wrench](mat, vec)

    return cmtm


def state_dict_to_rel_frame(state : dict, base_name : str, target_name : str) -> SE3:
    '''
    Convert state data to relative SE3
    Args:
        state (dict): state data
        base_name (str): name of the base link
        target_name (str): name of the target link
    Returns:
        SE3: SE3 object
    '''
    base_frame = state_dict_to_frame(state, base_name)
    target_frame = state_dict_to_frame(state, target_name)
    rel_frame = base_frame.inv() @ target_frame

    return rel_frame

def state_dict_to_rel_cmtm(state : dict, base_name : str, target_name : str, order = None) -> CMTM:
    '''
    Convert state data to relative CMTM
    Args:
        state (dict): state data
        base_name (str): name of the base link
        target_name (str): name of the target link
    Returns:
        CMTM: CMTM object
    '''
    base_cmtm = state_dict_to_cmtm(state, base_name, order)
    target_cmtm = state_dict_to_cmtm(state, target_name, order)
    rel_cmtm = base_cmtm.inv() @ target_cmtm

    return rel_cmtm

def state_dict_to_rel_cmtm_wrench(state : dict, base_name : str, target_name : str, order = None) -> CMTM:
    '''
    Convert state data to relative CMTM
    Args:
        state (dict): state data
        base_name (str): name of the base link
        target_name (str): name of the target link
    Returns:
        CMTM: CMTM object
    '''
    base_cmtm = state_dict_to_cmtm_wrench(state, base_name, order)
    target_cmtm = state_dict_to_cmtm_wrench(state, target_name, order)
    rel_cmtm = base_cmtm.inv() @ target_cmtm

    return rel_cmtm

def state_dict_to_vecs(state : dict, name : str, type_name : str) -> np.ndarray:
    '''
    Convert state data to vector
    Args:
        state (dict): state data
        name (str): name of the link or joint
        type_name (str): type of the vector (e.g., "link", "joint")
    Returns:
        np.ndarray: vector
    '''
    vecs = []
    
    for k in state.keys():
        if k.startswith(name + "_") and k.endswith("_" + type_name):
            vecs.append(np.array(state[k]))
        elif re.match(rf"{name}_{type_name}_diff\d+", k):
            vecs.append(np.array(state[k]))
    if len(vecs) == 0:
        raise ValueError(f"Invalid name: {name} or type_name: {type_name}.")

    return np.concatenate(vecs)

def state_dict_to_cmvec(state : dict, name : str, type_name : str, order : int) -> CMVector:
    '''
    Convert state data to vector
    Args:
        state (dict): state data
        name (str): name of the link or joint
        type_name (str): type of the vector (e.g., "link", "joint")
    Returns:
        CMVector: vector
    '''
    vecs = []

    l = 0
    for k in state.keys():
        if k.startswith(name + "_") and k.endswith("_" + type_name):
            vecs.append(np.array(state[k]))
            l += 1
        elif re.match(rf"{name}_{type_name}_diff\d+", k):
            vecs.append(np.array(state[k]))
            l += 1
        if l >= order:
            break
    if len(vecs) == 0:
        raise ValueError(f"Invalid name: {name} or type_name: {type_name}.")

    return CMVector(np.stack(vecs))

def extract_dict_link_info(state : dict, data_type : str, link_name : str, frame = "dummy", rel_frame = 'dummy'):
    if data_type == "rot":
      return state_dict_to_rot(state, link_name)
    elif data_type == "frame":
        return state_dict_to_frame(state, link_name)
    elif data_type == "cmtm":
        return state_dict_to_cmtm(state, link_name)
    elif data_type == "force":
        return np.array(state[link_name+"_link_force"])    
    elif "force_diff" in data_type:
        return np.array(state[link_name+"_link_"+data_type])
    elif data_type == "momentum":
        return np.array(state[link_name+"_link_momentum"])
    elif "momentum_diff" in data_type:
        return np.array(state[link_name+"_link_"+data_type])
    else:
        return np.array(state[link_name+"_"+keys_name[data_type]])
    
def extract_dict_joint_info(state : dict, data_type : str, joint_name : str, frame = "dummy", rel_frame = 'dummy'):
    if data_type == "frame":
        return state_dict_to_frame(state, joint_name)
    elif data_type == "cmtm":
        return state_dict_to_cmtm(state, joint_name)
    elif data_type == "force":
        return np.array(state[joint_name+"_joint_force"])
    elif "force_diff" in data_type:
        return np.array(state[joint_name+"_joint_"+data_type])
    elif data_type == "momentum":
        return np.array(state[joint_name+"_joint_momentum"])
    elif "momentum_diff" in data_type:
        return np.array(state[joint_name+"_joint_"+data_type])
    elif data_type == "torque":
        return np.array(state[joint_name+"_joint_torque"])
    elif "torque_diff" in data_type:
        return np.array(state[joint_name+"_joint_"+data_type])
    else:
        return np.array(state[joint_name+"_"+data_type])
    
def extract_dict_total_link(state : dict, link_name_list : str, data_type : str, order : int) -> CMVector:
    for i, link_name in enumerate(link_name_list):
        vec = extract_dict_link_info(state, link_name, data_type, order)  
        if i == 0:
            total_vec = np.zeros((len(link_name_list), vec._len)) 
    return total_vec.flatten()

def extract_dict_total_link_cmvec(state : dict, link_name_list : str, data_type : str, order : int) -> CMVector:
    for i, link_name in enumerate(link_name_list):
        vec = state_dict_to_cmvec(state, link_name, data_type, order)  
        if i == 0:
            total_vec = np.zeros((len(link_name_list), vec._len)) 
        total_vec[i] = vec.cm_vec()
    return total_vec.flatten()

def extract_dict_total_info(data : dict, data_type : str, name : str, frame = "dummy", rel_frame = 'dummy'):
    pass

def extract_dict_info(data : dict, data_type : str, group : str, name : str, frame = "dummy", rel_frame = 'dummy'):

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
        return CMTM.sub_tan_vec_var(data0[type], data1[type], "bframe")
    else:
        raise ValueError(f"Invalid type: {set(type)}")
    
def print_state_dict(state : dict):
    for k in state.keys():
        print(f"{k}: {state[k]}")