
import numpy as np

from ..core.robot import RobotStruct
from ..core.state import StateType
from ..core.state_cache import StateCache
from .state import get_value
from .state import build_kinematics_state, build_dynamics_cmtm_state


def compute_outward_value(robot : RobotStruct, motions : np.ndarray, state_type : StateType, input_order = None) -> dict:
  if input_order is None or input_order == state_type.time_order:
    motion = motions
  else:
    motion = np.zeros(robot.dof * state_type.time_order)
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

def update_outward_state(robot : RobotStruct, motion_pack, state_cache : StateCache, is_dynamics : bool, order = 3) -> dict:
  if state_cache is None:
    if not is_dynamics:
      state_cache = StateCache(
        build_state=lambda x_all, time=None, required=None: build_kinematics_state(robot, x_all, order)
      )
    else:
      state_cache = StateCache(
        build_state=lambda x_all, time=None, required=None: build_dynamics_cmtm_state(robot, x_all, order-2)
      )

  state_cache.update_if_needed(motion_pack)

  return state_cache.state
