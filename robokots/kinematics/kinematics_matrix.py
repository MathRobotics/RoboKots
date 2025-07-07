#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025. 6.20 Created by T.Ishigaki
# kinematics computation module by matrix formulation

import numpy as np

import mathrobo as mr

from .base import JointData

def joint_select_diag_mat(joint : JointData, order : int = 1) -> np.ndarray:
    return np.kron(np.eye(order), joint.select_mat)