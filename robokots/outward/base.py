from ..basic.robot import JointStruct, LinkStruct
from ..kinematics.base import JointData, SoftLinkData

def convert_joint_to_data(joint: JointStruct) -> JointData:
  '''
  Convert joint data to JointData structure
  Args:
    joint (JointStruct): joint structure
  Returns:
    JointData: JointData structure
  '''
  return  JointData(joint.origin, joint.select_mat, joint.dof, joint.select_indeces)

def convert_link_to_data(link: LinkStruct) -> SoftLinkData:
  '''
  Convert link data to SoftLinkData structure
  Args:
    link (LinkStruct): link structure
  Returns:
    SoftLinkData: SoftLinkData structure
  '''
  return  SoftLinkData(link.select_mat, link.dof, link.select_indeces)