"""
Simple example demonstrating how to build and solve an inward optimization problem.

The script optimizes two scalar variables so that they satisfy a small linear
system while also staying close to a soft prior. It uses the inward ``Problem``
and ``VariablePack`` helpers together with a custom ``Expr``.
"""
import numpy as np

from robokots.inward.opt import solve_gauss_newton
from robokots.inward import term


class TwoVarLinearExpr:
    """
    Example residual with m=2:
        r1 = x + 2y - 1
        r2 = x -  y - 0.2
    """

    name = "two_var_linear"

    def __init__(self, x: term.Variable, y: term.Variable):
        self.vars = [x, y]
        self._x = x
        self._y = y

    def deps(self):
        return []

    def eval(self, ctx: term.EvalContext):
        x = float(self._x.x[0])
        y = float(self._y.x[0])
        r = np.array([x + 2.0 * y - 1.0, x - 1.0 * y - 0.2], dtype=float)

        Jx = np.array([[1.0], [1.0]], dtype=float)
        Jy = np.array([[2.0], [-1.0]], dtype=float)
        return r, [Jx, Jy]


def main() -> None:
    # -----------------------
    # Variables (x, y)
    # -----------------------
    x = term.Variable(name="x", x=np.array([0.0], dtype=float))
    y = term.Variable(name="y", x=np.array([0.0], dtype=float))
    variables = term.VariablePack([x, y])

    # -----------------------
    # Expr -> Residual
    # -----------------------
    residual = TwoVarLinearExpr(x, y)

    class PriorExpr:
        def __init__(self, v: term.Variable, name: str):
            self.name = name
            self.vars = [v]
            self._v = v

        def deps(self):
            return []

        def eval(self, ctx: term.EvalContext):
            r = np.array([float(self._v.x[0])], dtype=float)
            J = np.array([[1.0]], dtype=float)
            return r, [J]

    prior_x = PriorExpr(x, "prior_x")
    prior_y = PriorExpr(y, "prior_y")

    # -----------------------
    # Problem
    # -----------------------
    problem = term.Problem(
        variables=variables,
        terms=[
            (residual, term.L2Cost()),
            (prior_x, term.ScalarWeightCost(w=1e-3)),
            (prior_y, term.ScalarWeightCost(w=1e-3)),
        ],
    )

    print("Initial:", variables.get())
    print("Initial cost:", problem.cost_value())

    solve_gauss_newton(problem, variables, max_iters=10)

    print("Final:", variables.get())
    print("Final cost:", problem.cost_value())

    # check solution (this linear system has a unique solution)
    # y2 = x - y - 0.2 = 0 -> x = y + 0.2
    # y1 = x + 2y - 1 = 0 -> (y+0.2) + 2y = 1 -> 3y = 0.8 -> y = 0.266666...
    # x = 0.466666...
    print("Expected approx: x≈0.4666667, y≈0.2666667")


if __name__ == "__main__":
    main()
