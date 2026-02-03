#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025.04.05 Created by T.Ishigaki
# inward computation module from state to variables

import numpy as np
from typing import List, Sequence, Tuple
from robokots.core.robot import RobotStruct
from robokots.core.state import StateType
from robokots.inward import term
from robokots.outward.state import get_value, build_kinematics_state
from robokots.outward.diff.outward_total_gradient import outward_kinematics_jacobian

def inverse_kinematics(robot : RobotStruct, target_type : List[StateType], target_value : List[np.ndarray],
                    q_init : np.ndarray, opt_func : None) -> np.ndarray:
    class TargetExpr:
        """Expr that returns target error and block Jacobian."""
        name: str
        vars: Sequence[term.Variable]

        def __init__(self, motion_var: term.Variable) -> None:
            self.name = "target_error"
            self.vars = [motion_var]

        def deps(self):
            return []

        def eval(self, ctx: term.EvalContext) -> Tuple[np.ndarray, Sequence[np.ndarray]]:
            q = np.asarray(self.vars[0].x, dtype=float).reshape(-1)
            state_dict = build_kinematics_state(robot, q, order=1)
            value = [get_value(robot, state_dict, t) - v for t, v in zip(target_type, target_value)]
            r = np.concatenate(value)
            J = outward_kinematics_jacobian(robot, state_dict, target_type, list_output=False)
            return r, [J]

    joint_var = term.Variable(name="q", x=np.zeros(robot.dof, dtype=float))
    variables = term.VariablePack([joint_var])

    target_expr = TargetExpr(joint_var)
    target_cost = term.L2Cost()

    problem = term.Problem(variables=variables, terms=[(target_expr, target_cost)])
    ctx = term.EvalContext(pack=variables)

    def _set_state_from_vector(x: np.ndarray) -> None:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != variables.n_total:
            raise ValueError(f"Expected {variables.n_total} decision variables, got {x.size}")
        variables.vars[0].x = x

    def residual(x: np.ndarray) -> np.ndarray:
        """Residual callable"""

        problem.set_from_vector(x)
        r_all, _ = problem.linearize(ctx=ctx)
        return r_all


    def jacobian(x: np.ndarray) -> np.ndarray:
        """Jacobian callable"""

        problem.set_from_vector(x)
        _, J_all = problem.linearize(ctx=ctx)
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
