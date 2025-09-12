#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025. 6.20 Created by T.Ishigaki
# dynamics computation module by matrix formulation

import math
import numpy as np
from mathrobo import SO3, SE3, SE3wrench, CMTM

from .dynamics import diag_factorials, diag_inv_factorials

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

def momentum_to_force_mat(m : CMTM, force_order : int = 1, dim : int = 6) -> np.ndarray:
    momentum_dof = dim * (force_order+1)
    force_dof = dim * force_order
    mat = np.zeros((force_dof, momentum_dof))

    v = np.zeros_like(m.vecs(force_order+1))
    vecs = m.vecs(force_order+1)
    for i in range(momentum_dof//dim-1):
        v[i] = vecs[i] / math.factorial(i)

    mat[:, dim:] = np.diag(np.repeat(np.arange(1, force_order+1), dim))
    if dim == 6:
      mat[:, :-dim] += CMTM.hat_adj(SE3wrench, v)
    elif dim == 3:
      mat[:, :-dim] += CMTM.hat_adj(SO3, v)
    return diag_factorials(force_order, dim) @ mat @ diag_inv_factorials(force_order+1, dim)
    # mat[:, dim:] = np.eye(force_dof)
    # if dim == 6:
    #   mat[:, :-dim] += diag_factorials(force_order, dim) @ CMTM.hat_adj(SE3wrench, v) @ diag_inv_factorials(force_order, dim)
    # elif dim == 3:
    #   mat[:, :-dim] += diag_factorials(force_order, dim) @ CMTM.hat_adj(SO3, v) @ diag_inv_factorials(force_order, dim)
    # return mat