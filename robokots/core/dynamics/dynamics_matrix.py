#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025. 6.20 Created by T.Ishigaki
# dynamics computation module by matrix formulation

import numpy as np
from mathrobo import Factorial, CMVector
from mathrobo import SO3wrench, SE3wrench, CMTM

from ..state import dim_to_dof
from .dynamics import link_momentum_cmvec

def inertia_diag_mat(inertia : np.ndarray, order : int = 1) -> np.ndarray:
    if inertia.shape != (6, 6):
        raise ValueError("Inertia matrix must be 6x6")
    
    return np.kron(np.eye(order), inertia)

def momentum_to_force_mat(link_cmtm : CMTM, force_order : int = 1, dim : int = 3) -> np.ndarray:
    dof = dim_to_dof(dim)
    momentum_dof = dof * (force_order+1)
    force_dof = dof * force_order
    mat = np.zeros((force_dof, momentum_dof))

    mat[:, dof:] = np.diag(np.repeat(np.arange(1, force_order+1), dof))
    mat[:, :-dof] += CMTM.hat_adj(SE3wrench, link_cmtm.cmvecs().cm_vecs()[:force_order+1])
    return Factorial.mat(force_order, dof) @ mat @ Factorial.mat_inv(force_order+1, dof)

def link_sp_vel_to_link_force_grad_mat(link_cmtm : CMTM, inertia : np.ndarray, force_order : int = 1, dim : int = 3) -> np.ndarray:
    dof = dim_to_dof(dim) 
    momentum_dof = dof * (force_order+2)
    force_dof = dof * force_order
    mat = np.zeros((force_dof, momentum_dof))
    m = np.zeros((force_dof, momentum_dof - dof))

    momentum = link_momentum_cmvec(inertia, link_cmtm.cmvecs())

    m[:, dof:] = np.diag(np.repeat(np.arange(1, force_order+1), dof)) @ inertia_diag_mat(inertia, force_order)
    m[:, :-dof] += (CMTM.hat_adj(SE3wrench, link_cmtm.cmvecs().cm_vecs()[:force_order+1]) @ inertia_diag_mat(inertia, force_order) 
                  + CMTM.hat_commute_adj(SE3wrench, momentum.cm_vecs()[:force_order+1])) 
    mat[:, dof:] = Factorial.mat(force_order, dof) @ m @ Factorial.mat_inv(force_order+1, dof)
    return mat

def partial_momentum_to_force_grad_mat(link_cmtm : CMTM, force_order : int = 1, dim : int = 3) -> np.ndarray:
    return momentum_to_force_mat(link_cmtm, force_order=force_order, dim=dim)

def partial_link_sp_vel_to_force_grad_mat(momentum : CMVector, force_order : int = 1, dim : int = 3) -> np.ndarray:
    dof = dim_to_dof(dim)
    momentum_dof = dof * (force_order+2)
    force_dof = dof * force_order
    mat = np.zeros((force_dof, momentum_dof))
    m = np.zeros((force_dof, momentum_dof - dof))

    m[:, :-dof] = CMTM.hat_commute_adj(SE3wrench, momentum.cm_vecs()[:force_order+1])
    mat[:, dof:] = Factorial.mat(force_order, dof) @ m @ Factorial.mat_inv(force_order+1, dof)
    return mat