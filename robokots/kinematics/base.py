import numpy as np

from mathrobo import SE3, CMTM

from ..basic.robot import JointStruct

from dataclasses import dataclass

@dataclass
class JointData:
    origin: SE3 # origin frame
    select_mat: np.ndarray # selection matrix
    dof: int = 0 # degree of freedom
    select_indeces: np.ndarray = None # indeces of the selection matrix

def convert_joint_to_data(joint: JointStruct) -> JointData:
  '''
  Convert joint data to JointData structure
  Args:
    joint (JointStruct): joint structure
  Returns:
    JointData: JointData structure
  '''
  return  JointData(joint.origin, joint.select_mat, joint.dof, joint.select_indeces)