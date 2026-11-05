
import numpy as np

from ..core.robot import RobotStruct
from ..core.motion import RobotMotions
from ..core.state.spec import StateType
from .state import get_value
from .state import build_kinematics_outward_state, build_dynamics_outward_state


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


def compute_outward_value(
  robot : RobotStruct,
  motions : np.ndarray,
  state_type : StateType,
  input_order = None,
  gravity=(0.0, 0.0, 0.0),
):
  if input_order is None:
    motion = np.asarray(motions, dtype=float).reshape(-1)
  elif input_order == state_type.time_order:
    motion = np.asarray(motions, dtype=float).reshape(-1)
  else:
    motion = _project_motion_order(robot, motions, input_order, state_type.time_order)

  if state_type.is_dynamics:
    state = build_dynamics_outward_state(
      robot, motion, max(state_type.time_order-2,0), gravity=gravity
    )
  else:
    state = build_kinematics_outward_state(robot, motion, state_type.time_order)
  return get_value(robot, state, state_type)
