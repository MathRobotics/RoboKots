"""Inverse kinematics least-squares optimization example using Quantities.

This demo mirrors the small inward/outward example in ``main.py`` but replaces
the synthetic linear system with a 2-DOF arm reaching task powered by
``robokots.kots.Kots``. The end-effector position is exposed as a
``Quantity`` so it can be paired with ``VectorSquaredSumResidual`` and solved
through a Gauss-Newton loop with the inward ``Problem`` utilities.
"""
from __future__ import annotations

import numpy as np

from robokots.core.state import StateType
from robokots.inward.problem import Problem
from robokots.kots import Kots
from robokots.outward.term import (
    L2Cost,
    Variable,
    VariablePack,
    VectorSquaredSumResidual,
)


class EndEffectorPositionQuantity:
    """Quantity that returns the translational error to a target position."""

    def __init__(
        self,
        kots: Kots,
        state_type: StateType,
        q_var: Variable,
        target: np.ndarray,
        name: str = "ee_position",
    ) -> None:
        self.name = name
        self.out_dim = int(target.size)
        self.vars = [q_var]
        self._kots = kots
        self._state_type = state_type
        self._q_var = q_var
        self._target = np.asarray(target, dtype=float).reshape(-1)

    def _update_motion(self) -> None:
        q = np.asarray(self._q_var.x, dtype=float).reshape(-1)
        self._kots.import_motions(q)

    def value(self) -> np.ndarray:
        self._update_motion()
        self._kots.kinematics()
        current_pos = self._kots.state_info(self._state_type)
        return current_pos - self._target

    def jacobian_blocks(self):
        # Keep the translational rows of the analytic Jacobian to match the
        # position-only residual. The Quantity interface requires blocks aligned
        # with ``vars`` ordering, so return a single block for ``q_var``.
        J_full = self._kots.jacobian(self._state_type)
        desired_rows = self.out_dim
        if J_full.shape[0] >= desired_rows:
            J_full = J_full[:desired_rows, :]
        if J_full.shape[0] != desired_rows and J_full.shape[1] == desired_rows:
            J_full = J_full.T
        if J_full.shape[0] != desired_rows:
            raise ValueError(
                "Jacobian shape is incompatible with residual: "
                f"J={J_full.shape}, residual=({desired_rows},)"
            )
        return [J_full]


def solve_gauss_newton(problem: Problem, variables: VariablePack, max_iters: int = 20) -> None:
    """Minimal Gauss-Newton loop that operates on an inward Problem."""

    for k in range(max_iters):
        r_all, J_all = problem.linearize()
        cost = float(r_all @ r_all)

        print(f"[iter {k}] q={variables.get()}  cost={cost:.6g}")

        if np.linalg.norm(r_all) < 1e-10:
            break

        lhs = J_all.T @ J_all
        rhs = -J_all.T @ r_all
        dx, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)

        if np.linalg.norm(dx) < 1e-12:
            break

        variables.apply_dx(dx)


def main() -> None:
    # Target position at x=2.2[m], y=0.3[m]
    target_pos = np.array([2.2, 0.3, 0.0], dtype=float)

    # Load the 2-DOF arm model and create a joint variable pack
    kots = Kots.from_json_file("../model/2dof_arm.json", order=1)
    q = Variable(name="q", x=np.zeros(kots.dof(), dtype=float))
    variables = VariablePack([q])

    state_type = StateType("link", "arm2", "pos")
    residual = VectorSquaredSumResidual(
        "end_effector_position",
        EndEffectorPositionQuantity(kots, state_type, q, target_pos),
    )

    problem = Problem(variables=variables, terms=[(residual, L2Cost())])

    print("Initial q:", variables.get())
    print("Initial cost:", problem.cost_value())

    solve_gauss_newton(problem, variables, max_iters=20)

    kots.kinematics()
    final_pos = kots.state_info(state_type)
    print("Final joint angles:", variables.get())
    print("Final position:", final_pos)
    print("Target position:", target_pos)


if __name__ == "__main__":
    main()
