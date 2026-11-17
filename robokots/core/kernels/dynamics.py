#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025.04.07 Created by T.Ishigaki
# dynamics module

import numpy as np

from mathrobo import CMVector
from mathrobo import SE3, SE3wrench, CMTM

from robokots.core import JointStruct
from robokots.core.kernels.cmtm_apply import apply_mat_inv_adj
from robokots.core.kernels.joint import local_tangent_mat


def joint_project_wrench(joint: JointStruct, wrench: np.ndarray, joint_coord: np.ndarray = None) -> np.ndarray:
    """Project spatial wrench values to generalized joint forces.

    The common 1-DoF joint types avoid dense ``select_mat`` multiplication.
    Multi-DoF and future custom joints can still fall back to the motion
    subspace matrix.
    """
    wrench = np.asarray(wrench)
    if joint.dof == 0:
        return np.zeros(wrench.shape[:-1] + (0,), dtype=wrench.dtype)
    if joint_coord is not None:
        select_mat = np.asarray(joint.select_mat, dtype=wrench.dtype)
        joint_coord = np.asarray(joint_coord, dtype=wrench.dtype)
        if joint_coord.ndim == 1:
            tangent_mat = local_tangent_mat(select_mat, joint_coord)
            return wrench @ tangent_mat
        tan = joint_coord @ select_mat.T
        tangent_mat = SE3.exp_integ_adj(-tan, 1.0) @ select_mat
        # Wrench series have derivative axes after the configuration's batch
        # axes. Broadcast the same joint tangent over those axes, not samples.
        while tangent_mat.ndim < wrench.ndim + 1:
            tangent_mat = np.expand_dims(tangent_mat, axis=-3)
        return np.einsum("...i,...ij->...j", wrench, tangent_mat)
    axis = np.asarray(joint.axis, dtype=wrench.dtype)
    if joint.type == "revolute" and joint.dof == 1:
        return np.sum(wrench[..., :3] * axis, axis=-1)[..., None]
    if joint.type == "prismatic" and joint.dof == 1:
        return np.sum(wrench[..., 3:6] * axis, axis=-1)[..., None]
    return wrench @ np.asarray(joint.select_mat, dtype=wrench.dtype)

def link_momentum(inertia : np.ndarray, veloc : np.ndarray) -> np.ndarray:
    """
    Calculate the momentum of a link.
    Args:
        inertia (numpy.ndarray): 6x6 spatial inertia matrix of the link.
        veloc (numpy.ndarray): 6x1 spatial velocity vector of the link.
    Returns:
        numpy.ndarray: 6x1 spatial momentum vector of the link.
    """
    return inertia @ veloc

def link_dynamics(inertia : np.ndarray, veloc : np.ndarray, accel : np.ndarray) -> np.ndarray:
    """
    Calculate the inverse dynamics of a link.
    Args:
        inertia (numpy.ndarray): 6x6 spatial inertia matrix of the link.
        veloc (numpy.ndarray): 6x1 spatial velocity vector of the link.
        accel (numpy.ndarray): 6x1 spatial acceleration vector of the link.
    Returns:
        numpy.ndarray: 6x1 spatial force vector acting on the link.
    """
    force = link_momentum(inertia, accel) + SE3wrench.hat_adj(veloc) @ link_momentum(inertia, veloc)
    return force

def joint_dynamics(joint_select : np.ndarray, rel_frame : SE3, p_joint_force : np.ndarray, link_force : np.ndarray) -> tuple:
    """
    Calculate the joint dynamics.
    Args:
        joint (Joint): joint object.
        rel_frame (SE3): relative frame of the joint.
        p_joint_force (numpy.ndarray): spatial force vector acting on the joint.
        link_force (numpy.ndarray): spatial force vector acting on the link.
    Returns:
        numpy.ndarray: joint force vector.
        numpy.ndarray: joint torque vector.
    """
    joint_force = rel_frame.mat_inv_adj() @ p_joint_force - link_force
    joint_torque = joint_select.T @ joint_force
    return joint_torque, joint_torque

def link_momentum_cmvec(inertia : np.ndarray, vel : CMVector) -> CMVector:
    """
    Calculate the link momentum and centripetal momentum.
    Args:
        inertia (numpy.ndarray): 6x6 spatial inertia matrix of the link.
        vel (CMVector): nx6 spatial vectors of the link.
    Returns:
        CMVector: momentum coefficients with the same row count as vel.
    """
    return CMVector.set_cmvecs(vel.cm_vecs() @ inertia.T)

def link_force_cmvec(vel: CMVector, momentum: CMVector, dim: int = 6) -> CMVector:
    """Compute force coefficients directly from velocity and momentum series.

    For normalized coefficients p_bar[k] = p^(k) / k!, the derivative
    contributes (k + 1) * p_bar[k + 1]. The cross term is a convolution
    of the velocity and momentum coefficients. The result has one fewer
    row than momentum, and preserves broadcast batch axes.
    """
    momentum_cm = momentum.cm_vecs()
    force_order = momentum._n - 1
    if force_order < 0:
        raise ValueError("Momentum must contain at least one row")
    if vel._n < force_order:
        raise ValueError("Velocity must contain at least as many rows as the force")
    if momentum_cm.shape[-1] != dim:
        raise ValueError("Momentum dimension must match dim")

    derivative_cm = momentum_cm[..., 1:, :] * np.arange(1, force_order + 1)[:, None]
    vel_hat = CMTM.hat_adj(SE3wrench, vel.cm_vecs()[..., :force_order, :])
    momentum_flat = momentum_cm[..., :force_order, :].reshape(
        momentum_cm.shape[:-2] + (force_order * dim,)
    )
    cross_cm = (vel_hat @ momentum_flat[..., None])[..., 0]
    force_cm = derivative_cm + cross_cm.reshape(cross_cm.shape[:-1] + (force_order, dim))
    return CMVector.set_cmvecs(force_cm)


def link_dynamics_cmvec(inertia : np.ndarray, vecs : np.ndarray) -> np.ndarray:
    """
    Calculate the link force and centripetal momentum.
    Args:
        momentum (numpy.ndarray): 6xn spatial momentum vectors of the link.
        vecs (numpy.ndarray): nx6 spatial vectors of the link.
    Returns:
        numpy.ndarray: 6xn spatial force vectors of the link.
    """
    n = vecs.shape[0]
    frac = np.ones((n,1))
    for i in range(1,n):
        frac[i] = frac[i-1] * i

    ## remain : implement frac
    return link_momentum_cmvec(inertia, vecs[1:]) + CMTM.hat_adj(SE3, vecs[:-1]) @ link_momentum_cmvec(inertia, vecs[:-1])

def joint_dynamics_cmvec(joint : JointStruct, rel_cmtm : CMTM, p_joint_force : np.ndarray, link_force : np.ndarray) -> tuple:
    joint_force = apply_mat_inv_adj(rel_cmtm, p_joint_force) - link_force
    joint_torque = np.zeros(joint.dof*rel_cmtm._n)
    for i in range(rel_cmtm._n):
        joint_torque[i*joint.dof:(i+1)*joint.dof] = joint_project_wrench(joint, joint_force[i*6:(i+1)*6])
    return joint_torque, joint_force
