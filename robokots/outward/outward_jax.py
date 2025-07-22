#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki
# outward computation module from motion and robot_model to state

import jax.numpy as jnp

from mathrobo import SE3

from ..basic.robot import RobotStruct
from ..basic.motion import RobotMotions
from ..kinematics.base import convert_joint_to_data
from ..kinematics.kinematics_jax import joint_rel_frame

from typing import NamedTuple

class KState(NamedTuple):
    names: tuple[str, ...]
    state: jnp.ndarray

def kinematics_jax(robot : RobotStruct, motions : RobotMotions):
    world_name = robot.links[robot.joints[0].parent_link_id].name
    state = KState(
        names=(world_name,),
        state = jnp.zeros((1, 4, 4)).at[0].set(SE3.eye().mat())
    )
    
    for joint in robot.joints:
        parent = robot.links[joint.parent_link_id]
        child = robot.links[joint.child_link_id]
        
        joint_data = convert_joint_to_data(joint)

        joint_motions = motions.joint_motions(joint.dof, joint.dof_index, order=1)
        print(f"joint dof: {joint.dof}, joint dof_index: {joint.dof_index}")
        print(f"joint_motions shape: {joint_motions.shape}")
        pidx = state.names.index(parent.name)

        p_link_frame = state.state[pidx]
        joint_frame = joint_rel_frame(joint_data, joint_motions)

        link_frame = p_link_frame @ joint_frame
        n_ = state.names + (child.name,)
        s_  = jnp.concatenate([state.state, link_frame.mat()], axis=0)

        state = KState(names=n_, state=s_)

    return state