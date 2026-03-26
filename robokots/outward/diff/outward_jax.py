#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki
# outward computation module from motion and robot_model to state

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from mathrobo import CMTM, SE3

from ...core.motion import RobotMotions
from ...core.robot import RobotStruct
from ...core.state_dict import cmtm_to_state_list
from ...core.models.kinematics.base import convert_joint_to_data
from ...core.models.kinematics.kinematics_jax import joint_local_cmtm, joint_rel_cmtm


class KState(NamedTuple):
    names: tuple[str, ...]
    state: jnp.ndarray


def _motion_vector(robot: RobotStruct, motions, order: int) -> np.ndarray:
    if isinstance(motions, RobotMotions):
        motion = np.zeros(robot.dof * order)
        for joint in robot.joints:
            idx = RobotMotions.owner_vec_index(joint.dof, joint.dof_index, order)
            motion[idx] = np.asarray(
                motions.joint_motions(joint.dof, joint.dof_index, order),
                dtype=float,
            ).reshape(-1)
        for link in robot.links:
            idx = RobotMotions.owner_vec_index(link.dof, link.dof_index, order)
            motion[idx] = np.asarray(
                motions.link_motions(link.dof, link.dof_index, order),
                dtype=float,
            ).reshape(-1)
        return motion

    motion = np.asarray(motions, dtype=float).reshape(-1)
    if robot.dof * order > motion.size:
        raise ValueError(f"Invalid motion length: {motion.size}. Must be {robot.dof * order}.")
    return motion


def _validate_rigid_links(robot: RobotStruct) -> None:
    if any(link.dof > 0 for link in robot.links):
        raise NotImplementedError(
            "build_kinematics_state_jax currently supports rigid-link kinematics only."
        )


def _build_kinematics_state_with_cmtm_jax(robot: RobotStruct, motions, order: int = 3):
    if order < 1:
        raise ValueError(f"Invalid order: {order}. Must be >= 1.")

    _validate_rigid_links(robot)
    motion = _motion_vector(robot, motions, order)

    state_dict = {}
    link_cmtm_dict = {}
    joint_cmtm_dict = {}

    world_name = robot.links[robot.joints[0].parent_link_id].name
    world_cmtm = CMTM.eye(SE3, order)
    link_cmtm_dict[world_name] = world_cmtm
    state_dict.update(cmtm_to_state_list(world_cmtm, "link", world_name))

    for joint in robot.joints:
        parent = robot.links[joint.parent_link_id]
        child = robot.links[joint.child_link_id]

        joint_data = convert_joint_to_data(joint)
        joint_motions = motion[RobotMotions.owner_vec_index(joint.dof, joint.dof_index, order)]

        parent_cmtm = link_cmtm_dict[parent.name]
        joint_rel = joint_rel_cmtm(joint_data, joint_motions, order)
        joint_local = joint_local_cmtm(joint_data, joint_motions, order)
        child_cmtm = parent_cmtm @ joint_rel

        link_cmtm_dict[child.name] = child_cmtm
        joint_cmtm_dict[joint.name] = joint_local

        state_dict.update(cmtm_to_state_list(child_cmtm, "link", child.name))
        state_dict.update(cmtm_to_state_list(joint_local, "joint", joint.name))

    return state_dict, link_cmtm_dict, joint_cmtm_dict


def build_kinematics_state_jax(robot: RobotStruct, motions, order: int = 3) -> dict:
    state_dict, _, _ = _build_kinematics_state_with_cmtm_jax(robot, motions, order)
    return state_dict


def kinematics_jax(robot: RobotStruct, motions, order: int = 1) -> KState:
    _, link_cmtm_dict, _ = _build_kinematics_state_with_cmtm_jax(robot, motions, order)
    world_name = robot.links[robot.joints[0].parent_link_id].name
    names = (world_name,) + tuple(robot.links[joint.child_link_id].name for joint in robot.joints)
    state = jnp.stack([jnp.asarray(link_cmtm_dict[name].elem_mat()) for name in names], axis=0)
    return KState(names=names, state=state)
