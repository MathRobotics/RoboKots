"""LiteOpt inverse-kinematics example that uses RoboKots utilities.

This script solves for joint angles of the provided 2-DOF arm model using
``liteopt.nls``. Residuals and Jacobians come directly from RoboKots'
``Quantity`` and ``Residual`` plumbing so we can keep least-squares assembly
in one place while still passing plain callables to LiteOpt.

Run from the repository root so imports resolve correctly:

```
python -m example.optimization_example.liteopt_two_link
```
"""
from __future__ import annotations

from pathlib import Path

import liteopt
import numpy as np

from robokots.core.state import StateType
from robokots.kots import Kots
from robokots.inward import term


MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "2dof_arm.json"
STATE_TYPE = StateType("link", "arm2", "pos")
TARGET_POSITION = np.array([2.2, 0.3, 0.0], dtype=float)
KOTS = Kots.from_json_file(str(MODEL_PATH), order=1)


class EndEffectorError(term.Quantity):
    """Quantity that returns end-effector position error and block Jacobian."""

    name = "end_effector_error"
    out_dim = 3

    def __init__(self, joint_var: term.Variable) -> None:
        self._joint_var = joint_var
        self.vars = [joint_var]

    def value(self) -> np.ndarray:
        q = np.asarray(self._joint_var.x, dtype=float).reshape(-1)
        _update_motion(q)
        return KOTS.state_info(STATE_TYPE) - TARGET_POSITION

    def jacobian_blocks(self) -> list[np.ndarray]:
        q = np.asarray(self._joint_var.x, dtype=float).reshape(-1)
        _update_motion(q)
        # One block per variable; here we only have the joint variable.
        return [KOTS.jacobian(STATE_TYPE)]


def _update_motion(q: np.ndarray) -> None:
    """Update the RoboKots model with a new joint configuration."""

    KOTS.import_motions(q)
    KOTS.kinematics()


def _build_problem() -> tuple[term.VariablePack, term.Problem]:
    """Create a Residual + Cost problem tree for least squares."""

    joint_var = term.Variable(name="q", x=np.zeros(KOTS.dof(), dtype=float))
    variables = term.VariablePack([joint_var])

    ee_quantity = term.CachedQuantity(EndEffectorError(joint_var), variables)
    ee_residual = term.VectorSquaredSumResidual("ee_error", ee_quantity)
    ee_cost = term.L2Cost()

    problem = term.Problem(variables=variables, terms=[(ee_residual, ee_cost)])
    return variables, problem


VARIABLES, PROBLEM = _build_problem()


def residual(x: np.ndarray) -> np.ndarray:
    """Residual callable compatible with ``liteopt.nls``."""

    PROBLEM.set_from_vector(x)
    r_all, _ = PROBLEM.linearize()
    return r_all


def jacobian(x: np.ndarray) -> np.ndarray:
    """Jacobian callable compatible with ``liteopt.nls``."""

    PROBLEM.set_from_vector(x)
    _, J_all = PROBLEM.linearize()
    return J_all


def main() -> None:
    # Start from a neutral joint guess of the correct dimension
    x0 = np.zeros(KOTS.dof(), dtype=float)

    import timeit
    start_time = timeit.default_timer()
    x_star, cost, iters, rnorm, dxnorm, converged = liteopt.nls(
        residual,
        jacobian,
        x0=x0,
        max_iters=200,
        tol_r=1e-10,
        tol_dx=1e-10,
    )
    end_time = timeit.default_timer()
    print(f"Optimization took {end_time - start_time:.4f} seconds.")

    _update_motion(np.asarray(x_star))
    end_effector = KOTS.state_info(STATE_TYPE)

    print("Converged:", converged)
    print("Iterations:", iters)
    print("Final cost:", cost)
    print("Residual norm:", rnorm)
    print("Step norm:", dxnorm)
    print("Solution (rad):", x_star)
    print("End-effector:", end_effector)
    print("Target:", TARGET_POSITION)

    start_time = timeit.default_timer()
    x_star = KOTS.inverse_kinematics([STATE_TYPE], [TARGET_POSITION], x0)
    end_time = timeit.default_timer()
    print(f"IK Optimization took {end_time - start_time:.4f} seconds.")

    print("IK Solution (rad):", x_star)


if __name__ == "__main__":
    main()
