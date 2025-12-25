#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025. 6.20 Created by T.Ishigaki
# kinematics computation module by matrix formulation

import numpy as np

def joint_select_diag_mat(select_mat : np.ndarray, order : int = 1) -> np.ndarray:
    return np.kron(np.eye(order), select_mat)