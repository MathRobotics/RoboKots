#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025. 6.20 Created by T.Ishigaki
# dynamics computation module by matrix formulation

import math
import numpy as np
from mathrobo import Factorial
from mathrobo import SO3, SE3wrench, CMTM

def inertia_diag_mat(inertia : np.ndarray, order : int = 1) -> np.ndarray:
    if inertia.shape != (6, 6):
        raise ValueError("Inertia matrix must be 6x6")
    
    return np.kron(np.eye(order), inertia)

def  natural_num_diag_mat(order : int = 1, dim : int = 6) -> np.ndarray:
    '''
    return diagonal matrix with natural numbers from 1 to order repeated dim times
    size is (dim * order, dim * order)
    '''
    v = np.repeat(np.arange(1, order+1), dim)
    return np.diag(v)

def  natural_num_inv_diag_mat(order : int = 1, dim : int = 6) -> np.ndarray:
    '''
    return diagonal matrix with natural numbers from 1 to order repeated dim times
    size is (dim * order, dim * order)
    '''
    v = np.repeat(np.arange(1, order + 1, dtype=float), dim)
    return np.diag(1.0 / v)

def momentum_to_force_mat(link_cmtm : CMTM, force_order : int = 1, dim : int = 6) -> np.ndarray:
    momentum_dof = dim * (force_order+1)
    force_dof = dim * force_order
    mat = np.zeros((force_dof, momentum_dof))

    v = np.zeros_like(link_cmtm.vecs(force_order+1))
    vecs = link_cmtm.vecs(force_order+1)
    for i in range(momentum_dof//dim-1):
        v[i] = vecs[i] / math.factorial(i)

    mat[:, dim:] = np.diag(np.repeat(np.arange(1, force_order+1), dim))
    if dim == 6:
      mat[:, :-dim] += CMTM.hat_adj(SE3wrench, v)
    elif dim == 3:
      mat[:, :-dim] += CMTM.hat_adj(SO3, v)
    return Factorial.mat(force_order, dim) @ mat @ Factorial.inverse_mat(force_order+1, dim)

def link_to_force_tan_map_mat(link_cmtm : CMTM, inertia : np.ndarray, force_order : int = 1, dim : int = 6) -> np.ndarray:
    momentum_dof = dim * (force_order+2)
    force_dof = dim * force_order
    mat = np.zeros((force_dof, momentum_dof))
    m = np.zeros((force_dof, momentum_dof - dim))

    v = np.zeros_like(link_cmtm.vecs(force_order+1))
    p = np.zeros_like(link_cmtm.vecs(force_order+1))
    vecs = link_cmtm.vecs(force_order+1)

    for i in range((momentum_dof-dim)//dim-1):
        v[i] = vecs[i] / math.factorial(i)
        p[i] = inertia @ vecs[i] / math.factorial(i)

    m[:, dim:] = np.diag(np.repeat(np.arange(1, force_order+1), dim)) @ inertia_diag_mat(inertia, force_order)
    if dim == 6:
      m[:, :-dim] += (CMTM.hat_adj(SE3wrench, v) @ inertia_diag_mat(inertia, force_order) + CMTM.hat_commute_adj(SE3wrench, p)) 
    elif dim == 3:
      m[:, :-dim] += (CMTM.hat_adj(SO3, v) @ inertia_diag_mat(inertia, force_order) + CMTM.hat_commute_adj(SO3, p))
    mat[:, dim:] = Factorial.mat(force_order, dim) @ m @ Factorial.inverse_mat(force_order+1, dim)
    return mat