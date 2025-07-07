from ..basic.robot import JointStruct
from ..kinematics.base import JointData

def convert_joint_to_data(joint: JointStruct) -> JointData:
  '''
  Convert joint data to JointData structure
  Args:
    joint (JointStruct): joint structure
  Returns:
    JointData: JointData structure
  '''
  return  JointData(joint.origin, joint.select_mat, joint.dof, joint.select_indeces)