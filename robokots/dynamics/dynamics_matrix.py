#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025. 6.20 Created by T.Ishigaki
# dynamics computation module by matrix formulation

import numpy as np

def inertia_diag_mat(inertia : np.ndarray, order : int = 1) -> np.ndarray:
    if inertia.shape != (6, 6):
        raise ValueError("Inertia matrix must be 6x6")
    
    return np.kron(np.eye(order), inertia)

def  natural_num_diag_mat(order : int = 1) -> np.ndarray:
    v = np.repeat(np.arange(1, order), 6)
    return np.diag(v)

