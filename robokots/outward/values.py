
import numpy as np

from ..core.robot import RobotStruct
from ..core.state import StateType
from .state import get_value

from .state import build_kinematics_state, build_dynamics_cmtm_state

def compute_outward_value(robot : RobotStruct, motions, state_type : StateType, input_order = None) -> dict:
  motion = np.zeros(robot.dof * state_type.time_order)

  if input_order is None:
    motion = motions
  else:
    time_order = state_type.time_order
    for joint in robot.joints:
        m = motions[joint.dof_index*input_order:joint.dof_index*input_order+joint.dof*time_order]
        motion[joint.dof_index*time_order:joint.dof_index*time_order+joint.dof*time_order] = m.flatten()

    for link in robot.links:
        m = motions[link.dof_index*input_order:link.dof_index*input_order+link.dof*time_order]
        motion[link.dof_index*time_order:link.dof_index*time_order+link.dof*time_order] = m.flatten()

  if state_type.is_dynamics:
    state_dict = build_dynamics_cmtm_state(robot, motion, max(state_type.time_order-2,0))
  else:
    state_dict = build_kinematics_state(robot, motion, state_type.time_order)
  return get_value(robot, state_dict, state_type)