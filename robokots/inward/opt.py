import numpy as np

from robokots.inward.problem import Problem
from robokots.inward.term import VariablePack

def solve_gauss_newton(problem: Problem, variables: VariablePack, max_iters: int = 20) -> None:
    """Minimal Gauss-Newton loop that operates on an inward Problem."""

    for k in range(max_iters):
        r_all, J_all = problem.linearize()

        if np.linalg.norm(r_all) < 1e-10:
            break

        lhs = J_all.T @ J_all
        rhs = -J_all.T @ r_all
        dx, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)

        if np.linalg.norm(dx) < 1e-12:
            break

        variables.apply_dx(dx)