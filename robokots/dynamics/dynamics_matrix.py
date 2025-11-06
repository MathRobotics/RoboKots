#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025. 6.20 Created by T.Ishigaki
# dynamics computation module by matrix formulation

import numpy as np
from mathrobo import Factorial
from mathrobo import SO3wrench, SE3wrench, CMTM

from .dynamics import link_momentum_cmtm

def inertia_diag_mat(inertia : np.ndarray, order : int = 1) -> np.ndarray:
    if inertia.shape != (6, 6):
        raise ValueError("Inertia matrix must be 6x6")
    
    return np.kron(np.eye(order), inertia)

def momentum_to_force_mat(link_cmtm : CMTM, force_order : int = 1, dim : int = 6) -> np.ndarray:
    momentum_dof = dim * (force_order+1)
    force_dof = dim * force_order
    mat = np.zeros((force_dof, momentum_dof))

    mat[:, dim:] = np.diag(np.repeat(np.arange(1, force_order+1), dim))
    if dim == 6:
      mat[:, :-dim] += CMTM.hat_adj(SE3wrench, link_cmtm.cmvecs().vecs()[:force_order+1])
    elif dim == 3:
      mat[:, :-dim] += CMTM.hat_adj(SO3wrench, link_cmtm.cmvecs().vecs()[:force_order+1])
    return Factorial.mat(force_order, dim) @ mat @ Factorial.mat_inv(force_order+1, dim)

def link_to_force_tan_map_mat(link_cmtm : CMTM, inertia : np.ndarray, force_order : int = 1, dim : int = 6) -> np.ndarray:
    momentum_dof = dim * (force_order+2)
    force_dof = dim * force_order
    mat = np.zeros((force_dof, momentum_dof))
    m = np.zeros((force_dof, momentum_dof - dim))

    momentum = link_momentum_cmtm(inertia, link_cmtm.cmvecs())

    m[:, dim:] = np.diag(np.repeat(np.arange(1, force_order+1), dim)) @ inertia_diag_mat(inertia, force_order)
    if dim == 6:
      m[:, :-dim] += (CMTM.hat_adj(SE3wrench, link_cmtm.cmvecs().cm_vecs()[:force_order+1]) @ inertia_diag_mat(inertia, force_order) 
                    + CMTM.hat_commute_adj(SE3wrench, momentum.cm_vecs()[:force_order+1])) 
    elif dim == 3:
      m[:, :-dim] += (CMTM.hat_adj(SO3wrench, link_cmtm.cmvecs().cm_vecs()[:force_order+1]) @ inertia_diag_mat(inertia, force_order) 
                      + CMTM.hat_commute_adj(SO3wrench, momentum.cm_vecs()[:force_order+1]))
    mat[:, dim:] = Factorial.mat(force_order, dim) @ m @ Factorial.mat_inv(force_order+1, dim)
    return mat