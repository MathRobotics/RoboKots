#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025.04.05 Created by T.Ishigaki
# inward computation module from state to variables

import numpy as np
from typing import List
from robokots.core.robot import RobotStruct
from robokots.core.state import StateType
from robokots.outward import term
from robokots.outward.values import update_outward_state
from robokots.outward.state import get_value, build_kinematics_state
from robokots.outward.diff.outward_total_gradient import outward_kinematics_jacobian

def inverse_kinematics(robot : RobotStruct, target_type : List[StateType], target_value : List[np.ndarray],
                    q_init : np.ndarray, opt_func : None) -> np.ndarray:
    class TargetQuantity(term.Quantity):
        """Quantity that returns target error and block Jacobian."""
        out_dim = 3
        state_dict_ = {}

        def __init__(self, motion_var: term.Variable) -> None:
            self._motion_var = motion_var
            self.vars = [self._motion_var]

        def _update_state(self, q: term.Variable) -> None:
            """Update the RoboKots model with a new joint configuration."""
            self.state_dict_ = build_kinematics_state(robot, q, order=1)

            # self.state_dict_ = update_outward_state(
            #     robot,
            #     motions=self._motion_var,
            #     state_cache=None,
            #     is_dynamics=False,
            #     order=max(t.time_order for t in target_type),
            # )

        def value(self) -> np.ndarray:
            # self._update_state(self._motion_var)
            q = np.asarray(self._motion_var.x, dtype=float).reshape(-1)
            self._update_state(q)
            value = [get_value(robot, self.state_dict_, t) - v for t, v in zip(target_type, target_value)]
            return np.concatenate(value)

        def jacobian_blocks(self) -> list[np.ndarray]:
            q = np.asarray(self._motion_var.x, dtype=float).reshape(-1)
            self._update_state(q)
            # self._update_state(self._motion_var)
            return outward_kinematics_jacobian(robot, self.state_dict_, target_type, list_output=True)

    joint_var = term.Variable(name="q", x=np.zeros(robot.dof, dtype=float))
    variables = term.VariablePack([joint_var])

    target_quantity_raw = TargetQuantity(joint_var)
    target_quantity = term.CachedQuantity(target_quantity_raw, variables)
    # target_quantity = target_quantity_raw
    target_residual = term.VectorSquaredSumResidual("target_error", target_quantity)
    target_cost = term.L2Cost()

    problem = term.Problem(variables=variables, terms=[(target_residual, target_cost)])

    def _set_state_from_vector(x: np.ndarray) -> None:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != variables.n_total:
            raise ValueError(f"Expected {variables.n_total} decision variables, got {x.size}")
        variables.vars[0].x = x

    def residual(x: np.ndarray) -> np.ndarray:
        """Residual callable"""

        problem.set_from_vector(x)
        r_all, _ = problem.linearize()
        return r_all


    def jacobian(x: np.ndarray) -> np.ndarray:
        """Jacobian callable"""

        problem.set_from_vector(x)
        _, J_all = problem.linearize()
        return J_all

    if opt_func is None:
        import liteopt
        opt_func = liteopt.nls

    x_star, cost, iters, rnorm, dxnorm, converged = opt_func(
        residual,
        jacobian,
        x0=q_init,
        max_iters=200,
        tol_r=1e-10,
        tol_dx=1e-10,
    )

    return x_star
