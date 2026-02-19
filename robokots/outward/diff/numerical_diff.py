import numpy as np
from mathrobo import numerical_difference, build_integrator

from robokots.core import RobotStruct
from robokots.core.motion import RobotMotions
from robokots.core.state import StateType, data_type_dof, data_type_to_sub_func
from robokots.core.state_dict import extract_dict_link_info

from robokots.outward.state import build_kinematics_state
from robokots.outward.values import compute_outward_value


def _make_lifted_update_func(robot_dof: int, order: int, update_method=None):
  method = "poly" if update_method is None else update_method
  eye = np.eye(robot_dof)
  lifted_cache = {}

  def update_func(x_init, direct, offset):
    key = float(offset)
    lifted = lifted_cache.get(key)
    if lifted is None:
      D, d = build_integrator(1, order, key, method=method)
      lifted = (np.kron(eye, D), np.kron(eye, d))
      lifted_cache[key] = lifted

    kron_D, kron_d = lifted
    return kron_D @ x_init + kron_d @ direct

  return update_func


def link_diff_kinematics_numerical(robot : RobotStruct, motions, link_name_list : list[str],  data_type : str, order = 3, \
                                    eps = 1e-8, update_method = None, update_direction = None) -> np.ndarray:
  if data_type not in ["pos", "rot", "vel", "acc", "jerk", "snap", "frame", "cmtm"]:
    raise ValueError(f"Invalid data_type: {data_type}. Must be 'pos', 'rot', 'vel', 'acc', 'frame' or 'cmtm'.")

  dof = data_type_dof(data_type, order, dim=3)

  diff = np.zeros((len(link_name_list), dof))

  update_func = _make_lifted_update_func(robot.dof, order, update_method=update_method)
  if update_direction is None:
    update_direction = np.ones(robot.dof)

  for i in range(len(link_name_list)):
    def kinematics_func(x):
      state = build_kinematics_state(robot, x, order)
      y = extract_dict_link_info(state, data_type, link_name_list[i])
      return y

    sub_func = data_type_to_sub_func(data_type)

    diff[i] = numerical_difference(motions, kinematics_func, sub_func = sub_func, update_func = update_func, direction = update_direction, eps=eps)

  return diff

def diff_outward_numerical(robot : RobotStruct, motions, state_type : StateType, order = None, eps = 1e-8, update_method = None, update_direction = None) -> np.ndarray:
  if order is None:
    order = state_type.time_order

  if state_type.time_order > order:
    return ValueError(f"Invalid order: {order}. Must be equal or larger than state_type.time_order {state_type.time_order}.")

  update_func = _make_lifted_update_func(robot.dof, order, update_method=update_method)

  def func(x):
    m = np.zeros(robot.dof * state_type.time_order)
    for joint in robot.joints:
      m[joint.dof_index*state_type.time_order:joint.dof_index*state_type.time_order+joint.dof*state_type.time_order] = \
        x[RobotMotions.owner_vec_index(joint.dof, joint.dof_index, order, state_type.time_order)]
    return compute_outward_value(robot, m, state_type)

  sub_func = data_type_to_sub_func(state_type.data_type)

  if update_direction is None:
    update_direction = np.ones(robot.dof)

  return numerical_difference(motions, func, sub_func = sub_func, update_func = update_func, direction = update_direction, eps=eps)
