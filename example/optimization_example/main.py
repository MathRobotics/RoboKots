"""
Simple example demonstrating how to build and solve an inward optimization problem.

The script optimizes a single scalar variable so that it reaches a desired target
while also staying close to a soft prior.  It uses the inward ``Problem`` and
``VariablePack`` helpers together with the residual and cost utilities from the
``outward`` module to form and solve a small least-squares problem.
"""
import numpy as np

from robokots.inward.problem import Problem
from robokots.inward.opt import solve_gauss_newton
from robokots.outward.term import (
    VariablePack,
    L2Cost,
    ScalarWeightCost,
    Variable,
    VectorSquaredSumResidual,
)
class TwoVarLinearQuantity:
    """
    Example of a Quantity with m=2:
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

    # also add a "weak prior" (optional)
    # pull x and y toward 0 (weakly)
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

    # check solution (this linear system has a unique solution)
    # y2 = x - y - 0.2 = 0 -> x = y + 0.2
    # y1 = x + 2y - 1 = 0 -> (y+0.2) + 2y = 1 -> 3y = 0.8 -> y = 0.266666...
    # x = 0.466666...
    print("Expected approx: x≈0.4666667, y≈0.2666667")


if __name__ == "__main__":
    main()
