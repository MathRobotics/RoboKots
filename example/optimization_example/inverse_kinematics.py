"""Inverse kinematics least-squares optimization example.

Implements a simple Gauss-Newton method that uses `robokots.kots.Kots`
forward kinematics and analytic Jacobians to align the end-effector
(`arm2` link) position with a target.
"""
import numpy as np

from robokots.core.state import StateType
from robokots.kots import Kots


def solve_inverse_kinematics(target_pos: np.ndarray, max_iters: int = 20) -> None:
    # Load the 2-DOF arm model and work with first-order (position-only) states
    kots = Kots.from_json_file("../model/2dof_arm.json", order=1)
    kots.import_motions(np.zeros(kots.dof(), dtype=float))

    state_type = StateType("link", "arm2", "pos")

    for i in range(max_iters):
        # Compute forward kinematics to obtain the current end-effector position
        kots.kinematics()
        current_pos = kots.state_info(state_type)
        residual = current_pos - target_pos

        cost = 0.5 * float(residual @ residual)
        print(f"[iter {i}] q={kots.motion()}  pos={current_pos}  cost={cost:.6f}")

        if np.linalg.norm(residual) < 1e-8:
            break

        # Retrieve the analytic Jacobian (derivative of link position w.r.t. joint angles)
        jacobian = kots.jacobian(state_type)

        # The kinematics API returns both translational and rotational terms stacked
        # vertically. Only the translational (x, y, z) rows are relevant for a
        # position-only residual, so keep the top rows that match the residual
        # dimension. This covers the common 6xN Jacobian output (position + rotation)
        # as well as 3xN outputs.
        desired_rows = residual.shape[0]
        if jacobian.shape[0] >= desired_rows:
            jacobian = jacobian[:desired_rows, :]

        # If the library returns derivatives stacked column-wise instead, transpose
        # to align rows with the residual dimension.
        if jacobian.shape[0] != residual.shape[0] and jacobian.shape[1] == residual.shape[0]:
            jacobian = jacobian.T

        if jacobian.shape[0] != residual.shape[0]:
            raise ValueError(
                "Jacobian shape is incompatible with residual: "
                f"J={jacobian.shape}, residual={residual.shape}"
            )

        # Compute the Gauss-Newton update
        dq, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
        if np.linalg.norm(dq) < 1e-10:
            break

        new_motion = kots.motion().copy()
        new_motion[: dq.shape[0]] += dq
        kots.import_motions(new_motion)

    print("Final joint angles:", kots.motion())
    print("Final position:", kots.state_info(state_type))
    print("Target position:", target_pos)


def main() -> None:
    # Target position at x=2.2[m], y=0.3[m]
    target_pos = np.array([2.2, 0.3, 0.0], dtype=float)
    solve_inverse_kinematics(target_pos)


if __name__ == "__main__":
    main()
