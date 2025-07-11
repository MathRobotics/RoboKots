#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025.04.07 Created by T.Ishigaki
# dynamics module

import numpy as np

from mathrobo import SO3, SE3, CMTM

from .robot import JointStruct

def inertia(i_vec : np.ndarray) -> np.ndarray:
    """
    Calculate the inertia tensor of a rigid body.
    Args:
        i_vec (numpy.ndarray): 6x1 vector representing the inertia tensor in the local frame.
    Returns:
        numpy.ndarray: 3x3 inertia tensor matrix.
    """
    if i_vec.shape != (6,):
        raise ValueError("i_vec must be a 6x1 vector")  
      
    i = np.eye(3)
    i[0,0] = i_vec[0]
    i[1,1] = i_vec[1]
    i[2,2] = i_vec[2]
    i[0,1] = i[1,0] = i_vec[3]
    i[0,2] = i[2,0] = i_vec[4]
    i[1,2] = i[2,1] = i_vec[5]

    return i


def spatial_inertia(m : float, i : np.ndarray, c : np.ndarray) -> np.ndarray:
    """
    Calculate the spatial inertia matrix of a rigid body.
    Args:
        m (float): mass of the body.
        i (numpy.ndarray): 3x3 inertia tensor of the body in its local frame.
        c (numpy.ndarray): 3x1 center of mass position vector in the body frame.
    Returns:
        numpy.ndarray: 6x6 spatial inertia matrix.
    """
    i_mat = np.zeros((6,6))

    c_hat = SO3.hat(c)
    i_mat[0:3,0:3] = inertia(i) - c_hat@c_hat
    i_mat[3:6,3:6] = m * np.eye(3)
    i_mat[3:6,0:3] = m * c_hat
    i_mat[0:3,3:6] = -m * c_hat
    return i_mat

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
    force = link_momentum(inertia, accel) - SE3.hat_adj(veloc).T @ link_momentum(inertia, veloc)
    return force

def joint_dynamics(joint : JointStruct, rel_frame : SE3, p_joint_force : np.ndarray, link_force : np.ndarray) -> tuple:
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
    joint_torque = joint.select_mat.T @ joint_force
    return joint_torque, joint_torque

def link_mometum_cmtm(inertia : np.ndarray, vecs : np.ndarray) -> np.ndarray:
    """
    Calculate the link momentum and centripetal momentum.
    Args:
        inertia (numpy.ndarray): 6x6 spatial inertia matrix of the link.
        vecs (numpy.ndarray): 6xn spatial vectors of the link.
    Returns:
        numpy.ndarray: 6xn spatial momentum vectors of the link.
    """
    return inertia @ vecs.T

def link_dynamics_cmtm(inertia : np.ndarray, vecs : np.ndarray) -> np.ndarray:
    """
    Calculate the link force and centripetal momentum.
    Args:
        momentum (numpy.ndarray): 6xn spatial momentum vectors of the link.
        vecs (numpy.ndarray): 6xn spatial vectors of the link.
    Returns:
        numpy.ndarray: 6xn spatial force vectors of the link.
    """
    n = vecs.shape[0]
    frac = np.ones((n,1))
    for i in range(1,n):
        frac[i] = frac[i-1] * i

    ## remain : implement frac
    return link_momentum(inertia, vecs[1:]) - CMTM.hat_adj(SE3, vecs[:-1]).T @ link_momentum(inertia, vecs[:-1])

def joint_dynamics_cmtm(joint : JointStruct, rel_cmtm : CMTM, p_joint_force : np.ndarray, link_force : np.ndarray) -> tuple:
    joint_force = rel_cmtm.mat_inv_adj() @ p_joint_force - link_force
    joint_torque = np.zeros(joint.dof*rel_cmtm._n)
    for i in range(rel_cmtm._n):
        joint_torque[i*6:(i+1)*6] = joint.selector(joint_force[i*6:(i+1)*6])
    return joint_torque, joint_torque