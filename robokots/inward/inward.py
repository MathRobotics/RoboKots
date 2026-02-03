#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025.04.05 Created by T.Ishigaki
# inward computation module from state to variables

import numpy as np
from typing import List, Sequence, Tuple

from robokots.core.robot import RobotStruct
from robokots.core.state import StateType
from robokots.inward import term
from robokots.inward.opt import solve_gauss_newton
from robokots.core.state_cache import StateCache, OwnerKey, StateKey
from robokots.outward.state import get_value, build_kinematics_state
from robokots.outward.diff.outward_total_gradient import outward_kinematics_jacobian

def inverse_kinematics(
    robot: RobotStruct,
    target_type: List[StateType],
    target_value: List[np.ndarray],
    q_init: np.ndarray,
    opt_func: None = None,
    *,
    max_iters: int = 200,
    tol_r: float = 1e-10,
    tol_dx: float = 1e-10,
) -> np.ndarray:
    if len(target_type) != len(target_value):
        raise ValueError("target_type and target_value must have the same length.")
    if not StateType.is_list_all_in_kinematics(target_type):
        raise ValueError("inverse_kinematics expects only kinematics StateType entries.")

    order = StateType.max_time_order(target_type)
    owner = OwnerKey(owner_type="ik", owner_name="target")
    key_value = StateKey(k=0, owner=owner, dtype="residual", field="value")
    key_jac = StateKey(k=0, owner=owner, dtype="residual", field="jac_q")

    target_value = [np.asarray(v, dtype=float).reshape(-1) for v in target_value]

    def build_state(x_all: np.ndarray, *, time=None, required=None) -> dict:
        q = np.asarray(x_all, dtype=float).reshape(-1)
        state_dict = build_kinematics_state(robot, q, order=order)

        r_parts = []
        for st, v in zip(target_type, target_value):
            y = np.asarray(get_value(robot, state_dict, st), dtype=float).reshape(-1)
            if y.size != v.size:
                raise ValueError(
                    f"target value size mismatch for '{st.owner_name}'. "
                    f"value has {y.size}, target has {v.size}"
                )
            r_parts.append(y - v)
        r = np.concatenate(r_parts, axis=0) if r_parts else np.zeros((0,), dtype=float)

        J = outward_kinematics_jacobian(robot, state_dict, target_type, list_output=False)

        return {
            key_value: r,
            key_jac: np.asarray(J, dtype=float),
        }

    state_cache = StateCache(build_state=build_state)

    class TargetExpr:
        """Expr that reads residual and Jacobian from StateCache."""
        name: str
        vars: Sequence[term.Variable]

        def __init__(self, motion_var: term.Variable) -> None:
            self.name = "target_error"
            self.vars = [motion_var]
            self._deps = [key_value, key_jac]

        def deps(self):
            return self._deps

        def eval(self, ctx: term.EvalContext) -> Tuple[np.ndarray, Sequence[np.ndarray]]:
            if ctx.state is None:
                raise ValueError("EvalContext.state (StateCache) is required.")
            r = np.asarray(ctx.state.get(key_value), dtype=float).reshape(-1)
            J = np.asarray(ctx.state.get(key_jac), dtype=float)
            return r, [J]

    q_init = np.asarray(q_init, dtype=float).reshape(-1)
    if q_init.size != robot.dof:
        raise ValueError(f"q_init must have size {robot.dof}, got {q_init.size}")

    joint_var = term.Variable(name="q", x=q_init.copy())
    variables = term.VariablePack([joint_var])

    target_expr = TargetExpr(joint_var)
    target_cost = term.L2Cost()

    problem = term.Problem(variables=variables, terms=[(target_expr, target_cost)])
    ctx = term.EvalContext(pack=variables, state=state_cache)
    required = target_expr.deps()

    if opt_func is None:
        solve_gauss_newton(
            problem,
            variables,
            max_iters=max_iters,
            tol_r=tol_r,
            tol_dx=tol_dx,
            ctx=ctx,
            required=required,
        )
        return variables.get()

    def residual(x: np.ndarray) -> np.ndarray:
        problem.set_from_vector(x)
        state_cache.update_if_needed(variables, required=required)
        r_all, _ = problem.linearize(ctx=ctx, required=required)
        return r_all

    def jacobian(x: np.ndarray) -> np.ndarray:
        problem.set_from_vector(x)
        state_cache.update_if_needed(variables, required=required)
        _, J_all = problem.linearize(ctx=ctx, required=required)
        return J_all

    x_star, _cost, _iters, _rnorm, _dxnorm, _converged = opt_func(
        residual,
        jacobian,
        x0=q_init,
        max_iters=max_iters,
        tol_r=tol_r,
        tol_dx=tol_dx,
    )

    return x_star
