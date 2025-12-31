"""
Simple example demonstrating how to build and solve an inward optimization problem.

The script optimizes a single scalar variable so that it reaches a desired target
while also staying close to a soft prior.  It uses the inward ``Problem`` and
``VariablePack`` helpers together with the residual and cost utilities from the
``outward`` module to form and solve a small least-squares problem.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from robokots.inward.problem import Problem
from robokots.outward.term import (
    VariablePack,
    L2Cost,
    ScalarWeightCost,
    Variable,
    VectorSquaredSumResidual,
)
class TwoVarLinearQuantity:
    """
    m=2 の Quantity 例:
        y1 = x + 2y - 1
        y2 = x -  y - 0.2
    """

    def __init__(self, x: Variable, y: Variable, name: str = "two_var_linear"):
        self.name = name
        self.out_dim = 2
        self.vars = [x, y]
        self._x = x
        self._y = y

    def value(self) -> np.ndarray:
        x = float(self._x.x[0])
        y = float(self._y.x[0])
        return np.array(
            [
                x + 2.0 * y - 1.0,
                x - 1.0 * y - 0.2,
            ],
            dtype=float,
        )

    def jacobian_blocks(self):
        # dy/dx (2x1), dy/dy (2x1)
        Jx = np.array([[1.0], [1.0]], dtype=float)
        Jy = np.array([[2.0], [-1.0]], dtype=float)
        return [Jx, Jy]


def solve_gauss_newton(problem: Problem, variables: VariablePack, max_iters: int = 10) -> None:
    """最小の GN ループ（line searchなし）"""
    for k in range(max_iters):
        r_all, J_all = problem.linearize()
        cost = float(r_all @ r_all)

        print(f"[iter {k}] x_all={variables.get()}  cost={cost:.6g}")
        # 収束判定（残差が十分小さい）
        if np.linalg.norm(r_all) < 1e-10:
            break

        lhs = J_all.T @ J_all
        rhs = -J_all.T @ r_all
        dx, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)

        if np.linalg.norm(dx) < 1e-12:
            break

        variables.apply_dx(dx)


def main() -> None:
    # -----------------------
    # Variables (x, y)
    # -----------------------
    x = Variable(name="x", x=np.array([0.0], dtype=float))
    y = Variable(name="y", x=np.array([0.0], dtype=float))
    variables = VariablePack([x, y])

    # -----------------------
    # Quantity -> Residual
    # -----------------------
    q = TwoVarLinearQuantity(x, y, name="two_var_linear")
    residual = VectorSquaredSumResidual("fit_two_equations", q)

    # ついでに "弱いprior" も足してみる（任意）
    # x と y を 0 に寄せる（ただし弱く）
    class PriorQuantity:
        def __init__(self, v: Variable, name: str):
            self.name = name
            self.out_dim = 1
            self.vars = [v]
            self._v = v

        def value(self):
            return np.array([float(self._v.x[0]) - 0.0], dtype=float)

        def jacobian_blocks(self):
            return [np.array([[1.0]], dtype=float)]

    prior_x = VectorSquaredSumResidual("prior_x", PriorQuantity(x, "prior_x"))
    prior_y = VectorSquaredSumResidual("prior_y", PriorQuantity(y, "prior_y"))

    # -----------------------
    # Problem
    # -----------------------
    problem = Problem(
        variables=variables,
        terms=[
            (residual, L2Cost()),
            (prior_x, ScalarWeightCost(w=1e-3)),
            (prior_y, ScalarWeightCost(w=1e-3)),
        ],
    )

    print("Initial:", variables.get())
    print("Initial cost:", problem.cost_value())

    solve_gauss_newton(problem, variables, max_iters=10)

    print("Final:", variables.get())
    print("Final cost:", problem.cost_value())

    # 解の確認（この線形系は一意解になる）
    # y2 = x - y - 0.2 = 0 -> x = y + 0.2
    # y1 = x + 2y - 1 = 0 -> (y+0.2) + 2y = 1 -> 3y = 0.8 -> y = 0.266666...
    # x = 0.466666...
    print("Expected approx: x≈0.4666667, y≈0.2666667")


if __name__ == "__main__":
    main()
