#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025. 6.20 Created by T.Ishigaki
# dynamics computation module by matrix formulation

import numpy as np
from mathrobo import SO3, SE3, CMTM

def inertia_diag_mat(inertia : np.ndarray, order : int = 1) -> np.ndarray:
    if inertia.shape != (6, 6):
        raise ValueError("Inertia matrix must be 6x6")
    
    return np.kron(np.eye(order), inertia)

def  natural_num_diag_mat(order : int = 1, dim : int = 6) -> np.ndarray:
    '''
    return diagonal matrix with natural numbers from 1 to order repeated dim times
    size is (dim * order, dim * order)
    '''
    v = np.repeat(np.arange(1, order), dim)
    return np.diag(v)

def momentum_to_force_mat(m : CMTM, order : int = 1, dim : int = 6) -> np.ndarray:
    momentum_dof = dim * (order+1)
    force_dof = dim * order
    mat = np.zeros((force_dof, momentum_dof))
    
    mat[:, dim:] = natural_num_diag_mat(force_dof, dim)
    if dim == 6:
      mat[:, :force_dof] += -CMTM.hat(SE3, m.tan_vecs(order+1)).T
    elif dim == 3:
      mat[:, :force_dof] += -CMTM.hat(SO3, m.tan_vecs(order+1)).T
      
    return mat