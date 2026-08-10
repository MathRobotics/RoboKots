from __future__ import annotations

import numpy as np

from ...core import batch as batch_api
from ...core.outward_state import ArrayOutwardState
from ...core.robot import RobotStruct
from .model import _rust_compiled_robot

def build_kinematics_outward_state_rust(
  robot: RobotStruct,
  motions,
  order: int = 3,
  compiled_robot=None,
) -> ArrayOutwardState:
  if order < 1:
    raise ValueError("order must be >= 1")
  rust_robot = compiled_robot if compiled_robot is not None else _rust_compiled_robot(robot)
  motion = np.asarray(motions, dtype=float)
  if motion.ndim == 1:
    _validate_motion_length(robot, motion, order)
    arrays = rust_robot.kinematics_cmtm(motion, order)
    batch_shape = ()
  else:
    flat_motion, batch_shape = batch_api.flatten_feature_batch(motion)
    if flat_motion.shape[-1] != robot.dof * order:
      raise ValueError(f"Invalid motion length: {flat_motion.shape[-1]}. Must be {robot.dof * order}.")
    arrays = rust_robot.kinematics_cmtm_batch(flat_motion, order)
    arrays = _reshape_batch_arrays(arrays, batch_shape)

  return _kinematics_outward_state_from_arrays(robot, arrays, order)


def build_dynamics_outward_state_rust(
  robot: RobotStruct,
  motions,
  dynamics_order: int = 1,
  compiled_robot=None,
  gravity=(0.0, 0.0, 0.0),
) -> ArrayOutwardState:
  if dynamics_order < 0:
    raise ValueError("dynamics_order must be >= 0")
  rust_robot = compiled_robot if compiled_robot is not None else _rust_compiled_robot(robot)
  gravity = np.asarray(gravity, dtype=float)
  if gravity.shape != (3,):
    raise ValueError(f"gravity must have shape (3,), got {gravity.shape}.")
  if not np.all(np.isfinite(gravity)):
    raise ValueError("gravity must contain only finite values.")
  gravity = np.ascontiguousarray(gravity)
  kin_order = dynamics_order + 2
  motion = np.asarray(motions, dtype=float)
  if motion.ndim == 1:
    _validate_motion_length(robot, motion, kin_order)
    arrays = rust_robot.dynamics_outward_cmtm(motion, dynamics_order, gravity)
  else:
    flat_motion, batch_shape = batch_api.flatten_feature_batch(motion)
    if flat_motion.shape[-1] != robot.dof * kin_order:
      raise ValueError(f"Invalid motion length: {flat_motion.shape[-1]}. Must be {robot.dof * kin_order}.")
    arrays = rust_robot.dynamics_outward_cmtm_batch(flat_motion, dynamics_order, gravity)
    arrays = _reshape_batch_arrays(arrays, batch_shape)

  return _dynamics_outward_state_from_arrays(robot, arrays, kin_order)


def _validate_motion_length(robot: RobotStruct, motion: np.ndarray, order: int) -> None:
  expected = robot.dof * order
  if motion.shape[-1] != expected:
    raise ValueError(f"Invalid motion length: {motion.shape[-1]}. Must be {expected}.")


def _reshape_batch_arrays(arrays: tuple[np.ndarray, ...], batch_shape: tuple[int, ...]) -> tuple[np.ndarray, ...]:
  reshaped = []
  for arr in arrays:
    arr = np.asarray(arr)
    reshaped.append(arr.reshape(batch_shape + arr.shape[1:]))
  return tuple(reshaped)


def _kinematics_outward_state_from_arrays(
  robot: RobotStruct,
  arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
  order: int,
) -> ArrayOutwardState:
  link_mat, link_vecs, joint_mat, joint_vecs = arrays
  return ArrayOutwardState(
    order=order,
    link_names=tuple(link.name for link in robot.links),
    joint_names=tuple(joint.name for joint in robot.joints),
    joint_dofs=tuple(joint.dof for joint in robot.joints),
    link_mat=link_mat,
    link_vecs=link_vecs,
    joint_mat=joint_mat,
    joint_vecs=joint_vecs,
  )


def _dynamics_outward_state_from_arrays(
  robot: RobotStruct,
  arrays: tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
  ],
  order: int,
) -> ArrayOutwardState:
  (
    link_mat,
    link_vecs,
    joint_mat,
    joint_vecs,
    link_momentum,
    link_force,
    joint_momentum,
    joint_force,
    joint_torque,
  ) = arrays
  return ArrayOutwardState(
    order=order,
    link_names=tuple(link.name for link in robot.links),
    joint_names=tuple(joint.name for joint in robot.joints),
    joint_dofs=tuple(joint.dof for joint in robot.joints),
    link_mat=link_mat,
    link_vecs=link_vecs,
    joint_mat=joint_mat,
    joint_vecs=joint_vecs,
    link_momentum_array=link_momentum,
    link_force_array=link_force,
    joint_momentum_array=joint_momentum,
    joint_force_array=joint_force,
    joint_torque_array=joint_torque,
  )
