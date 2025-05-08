# -*- coding: utf-8 -*-


import numpy as np
from mathrobo import SE3, CMTM, SO3

#specific 3d-CMTM
def cmtm_to_state_dict(cmtm : CMTM, name : str, order = 3) -> dict:
  '''
  Convert CMTM to state data
  Args:
    cmtm (CMTM): CMTM object
    name (str): name of the link
  Returns:
    dict: state data
  '''
  if order < 1 and order > 3:
    raise ValueError("order must be 1, 2 or 3")

  state = []

  if 1:
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
  else:
    mat = cmtm.elem_mat()
    veloc = cmtm.elem_vecs(0)
    accel = cmtm.elem_vecs(1)
    
    pos = mat[:3,3]
    rot_vec = mat[:3,:3].ravel()

    state = [
        (name+"_pos" , pos.tolist()),
        (name+"_rot" , rot_vec.tolist()),
        (name+"_vel" , veloc.tolist()),
        (name+"_acc" , accel.tolist())
    ]
    
  return state

def extract_dict_link_info(self, type : str, link_name : str, frame = "dummy", rel_frame = 'dummy'):
    if type == "pos":
      return self.link_pos(link_name)
    elif type == "rot":
      return self.link_rot(link_name)
    elif type == "vel":
      return self.link_vel(link_name)
    elif type == "acc":
      return self.link_acc(link_name)
    elif type == "frame":
      return self.link_frame(link_name)
    elif type == "cmtm":
      return self.link_cmtm(link_name)
    else:
      raise ValueError(f"Invalid type: {set(type)}")
    
def extract_dict_joint_info(self, type : str, joint_name : str, frame = "dummy", rel_frame = 'dummy'):
    if type == "coord":
      return self.joint_coord(joint_name)
    elif type == "veloc":
      return self.joint_veloc(joint_name)
    elif type == "accel":
      return self.joint_accel(joint_name)
    elif type == "cmtm":
      return self.joint_cmtm(joint_name)
    else:
      raise ValueError(f"Invalid type: {set(type)}")
    
def extract_dict_total_info(self, type : str, name : str, frame = "dummy", rel_frame = 'dummy'):
    'dummy'

def extract_dict_info(data : dict, type : str, name : str, frame = "dummy", rel_frame = 'dummy'):
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
      return extract_link_info(type, name, frame, rel_frame)
    elif group == "joint":
      return extract_joint_info(type, name, frame, rel_frame)
    elif group == "total":
      return extract_total_info(type, name, frame, rel_frame)
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