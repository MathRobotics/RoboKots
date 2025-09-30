#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025.04.07 Created by T.Ishigaki
# dynamics module

import math
import numpy as np

from mathrobo import SE3, SE3wrench, CMTM

from ..basic.robot import JointStruct


def diag_factorials(order : int, dim: int):
    v = np.repeat([math.factorial(i) for i in range(order)], dim)
    return np.diag(v)

def diag_inv_factorials(order: int, dim: int):
    v = np.repeat([1.0 / math.factorial(i) for i in range(order)], dim)
    return np.diag(v)

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

def link_momentum_cmtm(inertia : np.ndarray, vecs : np.ndarray) -> np.ndarray:
    """
    Calculate the link momentum and centripetal momentum.
    Args:
        inertia (numpy.ndarray): 6x6 spatial inertia matrix of the link.
        vecs (numpy.ndarray): nx6 spatial vectors of the link.
    Returns:
        numpy.ndarray: 6n spatial momentum vectors of the link.
    """
    return (vecs @ inertia.T).reshape(-1)

def link_force_cmtm(vels : np.ndarray, momentums : np.ndarray, dim : int = 6) -> np.ndarray:
    """
    Calculate the link force and centripetal momentum.
    Args:
        vecs (numpy.ndarray): dim x n spatial vectors of the link.
        momentums (numpy.ndarray): dim x n+1 spatial momentum vectors of the link.
    Returns:
        numpy.ndarray: dim x n spatial force vectors of the link.
    Note:    
        o @ f = d/dt(o @ m) - hat_adj(o @ vel).T @ (o @ m)
        f = o_inv @ d/dt(o @ m) - o_inv @ hat_adj(o @ vel).T @ (o @ m)
          = d/dt(m) - o_inv @ hat_adj(o @ vel).T @ (o @ m)
    """
    o = diag_inv_factorials(momentums.shape[0]//dim -1, dim=dim)
    o_inv = diag_factorials(momentums.shape[0]//dim -1, dim=dim)
    v = np.zeros_like(vels)
    
    for i in range(momentums.shape[0]//dim-1):
        v[i] = vels[i] / math.factorial(i)

    return momentums[dim:] + o_inv @ CMTM.hat_adj(SE3wrench, v) @ o @ momentums[:-dim]

def link_dynamics_cmtm(inertia : np.ndarray, vecs : np.ndarray) -> np.ndarray:
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
    return link_momentum_cmtm(inertia, vecs[1:]) + CMTM.hat_adj(SE3, vecs[:-1]) @ link_momentum_cmtm(inertia, vecs[:-1])

def joint_dynamics_cmtm(joint : JointStruct, rel_cmtm : CMTM, p_joint_force : np.ndarray, link_force : np.ndarray) -> tuple:
    joint_force = rel_cmtm.mat_inv_adj() @ p_joint_force - link_force
    joint_torque = np.zeros(joint.dof*rel_cmtm._n)
    for i in range(rel_cmtm._n):
        joint_torque[i*joint.dof:(i+1)*joint.dof] = (joint_force[i*6:(i+1)*6])[joint.select_indeces]
    return joint_torque, joint_force