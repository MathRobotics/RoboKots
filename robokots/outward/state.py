#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki
# outward computation module from motion and robot_model to state

import numpy as np

from mathrobo import CMVector, CMTM, Factorial, SE3, SE3wrench

from ..core.robot import RobotStruct
from ..core.motion import RobotMotions
from ..core.outward_state import OutwardState
from ..core.state_dict import (
    state_dict_to_cmtm,
    state_dict_to_cmtm_wrench,
    state_dict_to_cmvec,
    state_dict_to_frame,
)
from ..core.state import data_type_dof, StateType, state_dict_key

from ..core.models.kinematics.base import convert_joint_to_data, convert_link_to_data
from ..core.models.kinematics.kinematics import joint_local_cmtm, joint_rel_cmtm, joint_rel_frame
from ..core.models.kinematics.kinematics_matrix import joint_select_diag_mat
from ..core.models.kinematics.kinematics_soft_link import soft_link_local_cmtm, calc_link_local_point_frame

from ..core.models.dynamics.base import spatial_inertia
from ..core.models.dynamics.dynamics import (
    joint_dynamics,
    link_dynamics,
    link_force_cmvec,
    link_momentum_cmvec,
)

def get_dof(robot : RobotStruct, state_type : StateType, dim : int = 3) -> int:
    if "torque" in state_type.data_type:
        joint = robot.joint_list([state_type.owner_name])[0]
        return joint.dof
    else:
        return data_type_dof(state_type.data_type, dim = dim)

def get_value(robot : RobotStruct, state_dict : dict, state_type : StateType):
    if state_type.owner_type == "link":
        link_name = state_type.owner_name
    elif state_type.owner_type == "joint":
        joint = robot.joint_list([state_type.owner_name])[0]
        link_name = robot.links[joint.child_link_id].name

    if state_type.frame_name == "world":
        if state_type.is_dynamics:
            cmtm_wrench = state_dict_to_cmtm_wrench(state_dict, link_name, "link", state_type.key_order)
        else:
            cmtm = state_dict_to_cmtm(state_dict, link_name, "link", state_type.key_order)

    if state_type.data_type == "frame":
        return state_dict_to_frame(state_dict, state_type.owner_name)
    elif state_type.data_type == "cmtm":
        return state_dict_to_cmtm(state_dict, state_type.owner_name, state_type.owner_type)
    elif "momentum" in state_type.data_type:
        if state_type.frame_name == 'world':
            local_momentum = state_dict_to_cmvec(state_dict, state_type.owner_name, \
                                                 state_type.owner_type,
                                                 "momentum", \
                                                 state_type.key_order).cm_vec()
            world_momentum = CMVector.set_cmvecs((cmtm_wrench.mat_adj() @ local_momentum).reshape(-1,6)).vecs()
            return world_momentum[-1]
        else:
            return np.array(state_dict[state_type.alliance])
    elif "force" in state_type.data_type:
        if state_type.frame_name == 'world':
            local_force = state_dict_to_cmvec(state_dict, state_type.owner_name, \
                                                state_type.owner_type,
                                                "force", \
                                                state_type.key_order).cm_vec()
            world_force = CMVector.set_cmvecs((cmtm_wrench.mat_adj() @ local_force).reshape(-1,6)).vecs()
            return world_force[-1]
        else:
            return np.array(state_dict[state_type.alliance])
    elif "torque" in state_type.data_type:
        return np.array(state_dict[state_type.alliance])
    else:
        return np.array(state_dict[state_type.alliance])

def get_cmvec(robot : RobotStruct, state_dict : dict, state_type : StateType, order : int) -> CMVector:
    vec = state_dict_to_cmvec(state_dict, state_type.owner_name, state_type.owner_type, state_type.data_type, state_type.key_order)
    if state_type.frame_name == "world":
        if state_type.owner_type == "link":
            link_name = state_type.owner_name
        elif state_type.owner_type == "joint":
            joint = robot.joint_list([state_type.owner_name])
            link_name = robot.links[joint[0].child_link_id].name
        cmtm_wrench = state_dict_to_cmtm_wrench(state_dict, link_name, "link", order)
        vec = CMTM.change_elemclass(cmtm_wrench, SE3wrench).mat_adj() @ vec.cm_vec()
    return vec

def get_total_cmvec(robot : RobotStruct, state_dict : dict, owner_type : str, data_type : str, frame_name : None, order : int) -> CMVector:
    if owner_type == "link":
        name_list = robot.link_names
    elif owner_type == "joint":
        name_list = robot.joint_names

    for i, name in enumerate(name_list):
        vec = get_cmvec(robot, state_dict, StateType(owner_type, name, data_type, frame_name), order)
        if i == 0:
            total_vec = np.zeros((len(name_list), vec._len))
        total_vec[i] = vec.cm_vec()
    return total_vec.flatten()


def _truncate_link_cmtm_order(link_cmtm: CMTM, order: int) -> CMTM:
  if order < 1:
    raise ValueError(f"Invalid order: {order}. Must be >= 1.")
  if order > link_cmtm._n:
    raise ValueError(f"Invalid order: {order}. Must be <= source order {link_cmtm._n}.")
  if order == link_cmtm._n:
    return link_cmtm
  return CMTM[SE3](SE3.set_mat(link_cmtm.elem_mat()), link_cmtm.vecs()[: order - 1])


def _should_use_jax_kinematics(robot: RobotStruct, backend = None) -> bool:
  if backend is not None:
    if backend == "jax":
      return True
    if backend == "numpy":
      return False
    raise ValueError(f"Unsupported kinematics backend: {backend}. Use 'numpy' or 'jax'.")

  if len(robot.links) == 0:
    return False

  return getattr(robot.links[0], "lib", "numpy") == "jax" and all(link.dof == 0 for link in robot.links)


def build_kinematics_state(robot : RobotStruct, motions, order = 3, backend = None) -> dict:
  '''
  Forward kinematics computation
  Args:
    robot (RobotStruct): robot model
    motions : robot motion
  Returns:
    dict: state data
  '''
  if _should_use_jax_kinematics(robot, backend):
    from .diff.outward_jax import build_kinematics_state_jax

    return build_kinematics_state_jax(robot, motions, order)

  return build_kinematics_outward_state(robot, motions, order).to_state_dict(robot)


def build_kinematics_outward_state(robot : RobotStruct, motions, order = 3) -> OutwardState:
  '''
  Forward kinematics computation in the internal CMTM-backed state format.
  '''
  return _build_kinematics_state_with_cmtm(robot, motions, order)


def _build_kinematics_state_with_cmtm(robot: RobotStruct, motions, order: int = 3) -> OutwardState:
  motion = np.asarray(motions, dtype=float).reshape(-1)
  if robot.dof * order > motion.size:
    raise ValueError(f"Invalid motion length: {motion.size}. Must be {robot.dof * order}.")

  link_cmtm_dict = {}
  joint_cmtm_dict = {}

  # The world link is the parent link of the first joint.
  world_name = robot.links[robot.joints[0].parent_link_id].name
  world_cmtm = CMTM.eye(SE3, order)
  link_cmtm_dict[world_name] = world_cmtm

  for joint in robot.joints:
    parent = robot.links[joint.parent_link_id]
    child = robot.links[joint.child_link_id]

    joint_data = convert_joint_to_data(joint)
    link_data = convert_link_to_data(child)

    joint_motions = motion[RobotMotions.owner_vec_index(joint.dof, joint.dof_index, order)]
    link_motions = motion[RobotMotions.owner_vec_index(child.dof, child.dof_index, order)]

    parent_cmtm = link_cmtm_dict[parent.name]
    joint_rel = joint_rel_cmtm(joint_data, joint_motions, order)
    link_local = soft_link_local_cmtm(link_data, link_motions, order)

    child_cmtm = parent_cmtm @ joint_rel @ link_local

    link_cmtm_dict[child.name] = child_cmtm

    # Keep joint local CMTM in state for Jacobian and derivative routines.
    joint_local = joint_local_cmtm(joint_data, joint_motions, order)
    joint_cmtm_dict[joint.name] = joint_local

  return OutwardState(order=order, link_cmtm=link_cmtm_dict, joint_cmtm=joint_cmtm_dict)


def calc_link_total_point_frame(robot : RobotStruct, motions : RobotMotions, state : dict, point : float) -> SE3:
  base = 0.0
  p_link = robot.links[0]
  for l in robot.links:
      if point > base + l.length:
          base += l.length
          p_link = l
          continue
      p_link_frame = state_dict_to_frame(state, p_link.name)
      coord = motions.link_motions(l.dof, l.dof_index, 1)[0]
      return calc_link_local_point_frame(l, coord, p_link_frame, point - base)

# specific 3d space (magic number 6)
def build_dynamics_state(robot : RobotStruct, joint_motions) -> dict:  
  state_dict = build_kinematics_state(robot, joint_motions, 3)

  world_name = robot.links[robot.joints[0].parent_link_id].name
  state_dict.update([(state_dict_key("link", world_name, "force") , [0.,0.,0.,0.,0.,0.])])

  for joint in reversed(robot.joints):
    child = robot.links[joint.child_link_id]
    joint_data = convert_joint_to_data(joint)
    
    joint_coord = joint_motions[joint.dof_index:joint.dof_index+joint.dof]

    inertia = spatial_inertia(child.mass, child.inertia, child.cog)

    link_veloc = np.array(state_dict[state_dict_key("link", child.name, "vel")])
    link_accel = np.array(state_dict[state_dict_key("link", child.name, "acc")])
    
    link_force = link_dynamics(inertia, link_veloc, link_accel)  
    state_dict.update([(state_dict_key("link", child.name, "force") , link_force.tolist())])
    
    joint_frame = joint_rel_frame(joint_data, joint_coord)

    p_joint_force = np.zeros(6)
    for id in child.child_joint_ids:
      p_joint_force += state_dict[state_dict_key("joint", robot.joints[id].name, "force")]

    joint_torque, joint_force = joint_dynamics(joint.select_mat, joint_frame, p_joint_force, link_force)
    
    state_dict.update([(state_dict_key("joint", joint.name, "force") , joint_force.tolist())])
    state_dict.update([(state_dict_key("joint", joint.name, "torque") , joint_torque.tolist())])
    
  return state_dict

def build_dynamics_outward_state(robot : RobotStruct, motions, dynamics_order = 1) -> OutwardState:
  outward_state = build_kinematics_outward_state(robot, motions, dynamics_order + 2)
  link_cmtm_dict = outward_state.link_cmtm
  joint_momentum_cmvec = {}
  momentum_order = dynamics_order + 1
  factor_mat = Factorial.mat(momentum_order, 6)
  momentum_link_cmtm_dict = {
    name: _truncate_link_cmtm_order(link_cmtm, momentum_order)
    for name, link_cmtm in link_cmtm_dict.items()
  }

  for joint in reversed(robot.joints):
    child = robot.links[joint.child_link_id]
    child_joint_ids = child.child_joint_ids

    inertia = spatial_inertia(child.mass, child.inertia, child.cog)
    link_cmtm = link_cmtm_dict[child.name]

    # calculate link momentum
    link_momentum = link_momentum_cmvec(inertia, link_cmtm.cmvecs())
    outward_state.link_momentum[child.name] = link_momentum

    # calculate link force
    if dynamics_order > 0:
      link_force = link_force_cmvec(link_cmtm.cmvecs(), link_momentum)
      outward_state.link_force[child.name] = link_force

    # calculate joint momentum
    joint_momentums = np.asarray(link_momentum.vec(), dtype=float).copy()
    for c_id in child_joint_ids:
      c_joint = robot.joints[c_id]
      c_joint_link = robot.links[c_joint.child_link_id]

      c_joint_momentum = joint_momentum_cmvec.get(c_joint.name)
      if c_joint_momentum is None:
        raise ValueError(f"Missing child joint momentum for '{c_joint.name}'.")

      c_joint_rel_cmtm = momentum_link_cmtm_dict[child.name].inv() @ momentum_link_cmtm_dict[c_joint_link.name]
      c_joint_cmtm_wrench = CMTM.change_elemclass(c_joint_rel_cmtm, SE3wrench)

      joint_momentums += factor_mat @ c_joint_cmtm_wrench.mat_adj() @ c_joint_momentum.cm_vec()

    # calculate joint force and torque
    joint_momentum = CMVector(joint_momentums.reshape(-1,6))
    joint_momentum_cmvec[joint.name] = joint_momentum
    outward_state.joint_momentum[joint.name] = joint_momentum
    if dynamics_order > 0:
      joint_force = link_force_cmvec(link_cmtm.cmvecs(), joint_momentum)
      outward_state.joint_force[joint.name] = joint_force

      if joint.dof == 0:
        continue
      joint_torque = joint_select_diag_mat(joint.select_mat, dynamics_order).T @ joint_force.vec()
      outward_state.joint_torque[joint.name] = joint_torque.reshape(-1, joint.dof)

  # Compute for the world link
  world_link = robot.links[0]
  inertia = spatial_inertia(world_link.mass, world_link.inertia, world_link.cog)
  link_cmtm = link_cmtm_dict[world_link.name]

  link_vel = CMVector(link_cmtm.vecs())
  link_momentum = link_momentum_cmvec(inertia, link_vel)
  outward_state.link_momentum[world_link.name] = link_momentum

  if dynamics_order > 0:
    link_force = link_force_cmvec(link_vel, link_momentum)
    outward_state.link_force[world_link.name] = link_force
    
  return outward_state


def build_dynamics_cmtm_state(robot : RobotStruct, motions, dynamics_order = 1) -> dict:
  return build_dynamics_outward_state(robot, motions, dynamics_order).to_state_dict(robot)
