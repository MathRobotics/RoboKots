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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the inward optimization primitives and outward residual/cost helpers.
# These imports are intentionally explicit to keep the example clear and to
# avoid pulling in optional dependencies that a heavy wildcard import might
# trigger.
from robokots.inward.problem import Problem
from robokots.inward.variables import VariablePack
from robokots.outward.term import (
    L2Cost,
    ScalarWeightCost,
    Variable,
    VectorSquaredSumResidual,
)


class ScalarTargetQuantity:
    """Quantity that measures the difference between a variable and a target."""

    def __init__(self, variable: Variable, target: float, name: str):
        self.name = name
        self.out_dim = 1
        self.vars = [variable]
        self._variable = variable
        self._target = float(target)

    def value(self) -> np.ndarray:
        return np.array([self._variable.x[0] - self._target], dtype=float)

    def jacobian(self) -> np.ndarray:
        return np.array([[1.0]], dtype=float)


def solve_linearized_step(problem: Problem, variables: VariablePack) -> np.ndarray:
    """
    Solve a single Gauss-Newton step for the given problem.

    Returns the delta vector that was applied to the variables.
    """

    # Linearize all residuals into a stacked residual vector r and Jacobian J.
    # For a Gauss-Newton update, we solve the normal equations J^T J dx = -J^T r.

    r_all, J_all = problem.linearize()
    lhs = J_all.T @ J_all
    rhs = -J_all.T @ r_all
    dx, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    variables.apply_dx(dx)
    return dx


def main() -> None:
    # Define the decision variable and initial guess.
    offset = Variable(name="joint_offset", x=np.array([0.5], dtype=float))
    variables = VariablePack([offset])

    # Create two quantities: one pulling the variable toward a target value and
    # another that softly keeps it close to zero.
    target_quantity = ScalarTargetQuantity(offset, target=2.0, name="hit_target")
    prior_quantity = ScalarTargetQuantity(offset, target=0.0, name="stay_near_zero")

    # Wrap the quantities as residuals with different weights.
    hit_target = VectorSquaredSumResidual("hit_target", target_quantity)
    stay_small = VectorSquaredSumResidual("stay_near_zero", prior_quantity)

    # Build the least-squares problem using two terms: the main target cost
    # and a softer regularization cost to discourage large deviations.
    problem = Problem(
        terms=[
            (hit_target, L2Cost()),
            (stay_small, ScalarWeightCost(w=0.1)),
        ]
    )

    print("Initial variable:", variables.get())
    print("Initial cost:", problem.cost_value())

    # Apply a single Gauss-Newton step to improve the variable estimate.
    dx = solve_linearized_step(problem, variables)

    print("Applied delta:", dx)
    print("Updated variable:", variables.get())
    print("Final cost:", problem.cost_value())


if __name__ == "__main__":
    main()