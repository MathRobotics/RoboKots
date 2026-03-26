
import numpy as np

from ..core.robot import RobotStruct
from ..core.motion import RobotMotions
from ..core.state import StateType
from ..core.state_cache import StateCache
from .state import get_value
from .state import build_kinematics_state, build_dynamics_cmtm_state


def _project_motion_order(robot: RobotStruct, motions: np.ndarray, input_order: int, output_order: int) -> np.ndarray:
  if input_order < output_order:
    raise ValueError(
      f"Invalid order conversion: input_order={input_order} must be >= output_order={output_order}."
    )

  vec = np.asarray(motions, dtype=float).reshape(-1)
  required = robot.dof * input_order
  if vec.size < required:
    raise ValueError(f"Invalid motion length: {vec.size}. Must be at least {required}.")

  projected = np.zeros(robot.dof * output_order, dtype=vec.dtype)
  for owner in (*robot.joints, *robot.links):
    src = RobotMotions.owner_vec_index(owner.dof, owner.dof_index, input_order, output_order)
    dst = RobotMotions.owner_vec_index(owner.dof, owner.dof_index, output_order)
    projected[dst] = vec[src]

  return projected


def compute_outward_value(robot : RobotStruct, motions : np.ndarray, state_type : StateType, input_order = None) -> dict:
  if input_order is None:
    motion = np.asarray(motions, dtype=float).reshape(-1)
  elif input_order == state_type.time_order:
    motion = np.asarray(motions, dtype=float).reshape(-1)
  else:
    motion = _project_motion_order(robot, motions, input_order, state_type.time_order)

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
