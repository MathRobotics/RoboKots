#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025.09.22 Created by T.Ishigaki
# kinematics module (JAX version)

from typing import List

import jax
import jax.numpy as jnp
from jax import lax
from mathrobo import SE3, CMTM
from .base import JointData

# Joint-space twist calculation with proper JAX conditional
def local_tan_vec(select_mat: jnp.ndarray, joint_vec: jnp.ndarray) -> jnp.ndarray:
  if len(joint_vec) == 0:
    return jnp.zeros(6)
  else:
    return select_mat @ joint_vec

# Frame generation from twist
def local_frame(select_mat: jnp.ndarray, joint_coord: jnp.ndarray) -> SE3:
    twist = local_tan_vec(select_mat, joint_coord)
    return SE3.set_mat(SE3.exp(twist, LIB='jax'), LIB='jax')

# Higher-order CMTM construction
def local_cmtm(select_mat: jnp.ndarray, joint_motions: jnp.ndarray,
               dof: int = 1, order: int = 3) -> CMTM:
    if order < 1:
        raise ValueError(f"Invalid order: {order}. Must be >= 1.")

    frame = local_frame(select_mat, joint_motions[:dof].reshape((dof,)))
    vecs = jnp.zeros((order - 1, 6))
    def body_fun(i, arr):
        vec = local_tan_vec(
            select_mat,
            joint_motions[(i+1)*dof:(i+2)*dof].reshape((dof,))
        )
        return arr.at[i].set(vec)
    vecs = jax.lax.fori_loop(0, order-1, body_fun, vecs)

    return CMTM[SE3](frame, vecs)

# Joint-specific wrappers

def joint_local_frame(joint: JointData, joint_coord: jnp.ndarray) -> SE3:
    return local_frame(joint.select_mat, joint_coord)

joint_local_vel = local_tan_vec
joint_local_acc = local_tan_vec
joint_local_jerk = local_tan_vec

def joint_local_cmtm(joint: JointData, joint_motions: jnp.ndarray, order: int = 3) -> CMTM:
    return local_cmtm(joint.select_mat, joint_motions, joint.dof, order)

# Relative transformations

def joint_rel_frame(joint: JointData, joint_coord: jnp.ndarray) -> SE3:
    return joint.origin @ local_frame(joint.select_mat, joint_coord)

def joint_rel_cmtm(joint: JointData, joint_motions: jnp.ndarray, order: int = 3) -> CMTM:
    if order < 1:
        raise ValueError(f"Invalid order: {order}. Must be >= 1.")
    dof = joint.dof
    frame = joint_rel_frame(joint, joint_motions[:dof].reshape((dof,)))
    vecs = jnp.zeros((order - 1, 6))
    def rel_body(i, arr):
        v = local_tan_vec(
            joint.select_mat,
            joint_motions[(i+1)*dof:(i+2)*dof].reshape((dof,))
        )
        return arr.at[i].set(v)
    vecs = jax.lax.fori_loop(0, order-1, rel_body, vecs)
    return CMTM[SE3](frame, vecs)

# Kinematic transforms

def kinematics(joint: JointData, p_link_frame: SE3, joint_coord: jnp.ndarray) -> SE3:
    return p_link_frame @ joint_rel_frame(joint, joint_coord)

def forward_kinematics(joints: List[JointData], joint_coords: jnp.ndarray):
    frame_list = [SE3.eye(LIB='jax')]
    for joint in joints:
        joint_coord = joint_coords[joint.dof_index : joint.dof_index + joint.dof]
        frame = kinematics(joint, frame_list[-1], joint_coord)
        frame_list.append(frame)
    return frame_list

def kinematics_vel(joint: JointData, p_link_vel: jnp.ndarray,
                   joint_coord: jnp.ndarray, joint_veloc: jnp.ndarray) -> jnp.ndarray:
    rel = joint_rel_frame(joint, joint_coord)
    rel_vel = joint_local_vel(joint.select_mat, joint_veloc)
    return rel.mat_inv_adj() @ p_link_vel + rel_vel

def forward_kinematics_vel(joints: List[JointData], joint_motions: jnp.ndarray):
    joint_motions = joint_motions.flatten()
    vel_list = [jnp.zeros(6)]
    for joint in joints:
        offset = joint.dof_index * 2
        joint_coord = joint_motions[offset : offset + joint.dof]
        joint_veloc = joint_motions[offset + joint.dof : offset + 2*joint.dof]
        vel = kinematics_vel(joint, vel_list[-1], joint_coord, joint_veloc)
        vel_list.append(vel)
    return vel_list

def kinematics_acc(joint: JointData, p_link_vel: jnp.ndarray, p_link_acc: jnp.ndarray,
                   joint_coord: jnp.ndarray, joint_veloc: jnp.ndarray,
                   joint_accel: jnp.ndarray) -> jnp.ndarray:
    rel = joint_rel_frame(joint, joint_coord)
    rv = joint_local_vel(joint.select_mat, joint_veloc)
    ra = joint_local_acc(joint.select_mat, joint_accel)
    return (rel.mat_inv_adj() @ p_link_acc
            + SE3.hat_adj(rel.mat_inv_adj() @ p_link_vel, LIB = "jax") @ rv
            + ra)

def forward_kinematics_acc(joints: List[JointData], joint_motions: jnp.ndarray):
    joint_motions = joint_motions.flatten()
    vel_list = [jnp.zeros(6)]
    acc_list = [jnp.zeros(6)]
    for joint in joints:
        offset = joint.dof_index * 3
        joint_coord = joint_motions[offset : offset + joint.dof]
        joint_veloc = joint_motions[offset + joint.dof : offset + 2*joint.dof]
        joint_accel = joint_motions[offset + 2*joint.dof : offset + 3*joint.dof]
        vel = kinematics_vel(joint, vel_list[-1], joint_coord, joint_veloc)
        acc = kinematics_acc(joint, vel_list[-1], acc_list[-1], joint_coord, joint_veloc, joint_accel)
        vel_list.append(vel)
        acc_list.append(acc)
    return acc_list

def kinematics_jerk(joint: JointData, p_link_vel: jnp.ndarray, p_link_acc: jnp.ndarray,
                   p_link_jerk: jnp.ndarray, joint_coord: jnp.ndarray,
                   joint_veloc: jnp.ndarray, joint_accel: jnp.ndarray,
                   joint_jerk: jnp.ndarray) -> jnp.ndarray:
    rel = joint_rel_frame(joint, joint_coord)
    rv = joint_local_vel(joint.select_mat, joint_veloc)
    ra = joint_local_acc(joint.select_mat, joint_accel)
    rj = joint_local_jerk(joint.select_mat, joint_jerk)
    return (rel.mat_inv_adj() @ p_link_jerk
            + 2 * SE3.hat_adj(rel.mat_inv_adj() @ p_link_acc, LIB = "jax") @ rv
            + SE3.hat_adj( SE3.hat_adj(rel.mat_inv_adj() @ p_link_vel, LIB = "jax") @ rv, LIB="jax") @ rv
            + SE3.hat_adj(rel.mat_inv_adj() @ p_link_vel, LIB = "jax") @ ra
            + rj)

def forward_kinematics_jerk(joints: List[JointData], joint_motions: jnp.ndarray):
    joint_motions = joint_motions.flatten()
    vel_list = [jnp.zeros(6)]
    acc_list = [jnp.zeros(6)]
    jerk_list = [jnp.zeros(6)]
    for joint in joints:
        offset = joint.dof_index * 4
        joint_coord = joint_motions[offset : offset + joint.dof]
        joint_veloc = joint_motions[offset + joint.dof : offset + 2*joint.dof]
        joint_accel = joint_motions[offset + 2*joint.dof : offset + 3*joint.dof]
        joint_jerk = joint_motions[offset + 3*joint.dof : offset + 4*joint.dof]
        vel = kinematics_vel(joint, vel_list[-1], joint_coord, joint_veloc)
        acc = kinematics_acc(joint, vel_list[-1], acc_list[-1], joint_coord, joint_veloc, joint_accel)
        jerk = kinematics_jerk(joint, vel_list[-1], acc_list[-1], jerk_list[-1], 
                              joint_coord, joint_veloc, joint_accel, joint_jerk)
        vel_list.append(vel)
        acc_list.append(acc)
        jerk_list.append(jerk)
    return jerk_list

def kinematics_cmtm(joint: JointData, p_link_cmtm: CMTM,
                    joint_motions: jnp.ndarray, order: int = 3) -> CMTM:
    if joint.dof * order != joint_motions.size:
        raise ValueError(f"Invalid motions length: {joint_motions.size}. "
                         f"Expected {joint.dof * order}.")
    return p_link_cmtm @ joint_rel_cmtm(joint, joint_motions, order)

def part_link_jacob(joint: JointData, rel_frame: SE3) -> jnp.ndarray:
    return rel_frame.mat_inv_adj()[:, joint.select_indeces]

def part_link_cmtm_tan_jacob(joint: JointData, rel_cmtm: CMTM,
                              joint_cmtm: CMTM) -> jnp.ndarray:
    n = rel_cmtm._n
    mat = jnp.zeros((n*6, n*joint.dof))
    tmp = rel_cmtm.mat_inv_adj() @ joint_cmtm.tan_map()
    def ij_body(idx, m):
        i, j = idx
        block = tmp[i*6:(i+1)*6, j*6:(j+1)*6]
        return m.at[i*6:(i+1)*6, j*joint.dof:(j+1)*joint.dof].set(
            block[:, joint.select_indeces])
    idxs = [(i, j) for i in range(n) for j in range(i+1)]
    return jax.lax.fori_loop(0, len(idxs), lambda k, m: ij_body(idxs[k], m), mat)

def part_link_cmtm_jacob(joint: JointData, rel_cmtm: CMTM,
                          joint_cmtm: CMTM, link_cmtm: CMTM) -> jnp.ndarray:
    return link_cmtm.tan_map_inv() @ part_link_cmtm_tan_jacob(joint, rel_cmtm, joint_cmtm)
