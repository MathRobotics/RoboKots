"""Reference fixed-base forward dynamics.

This deliberately small implementation constructs the joint-space mass
matrix through inverse dynamics and solves it with NumPy.  It is the oracle
for the later O(n) articulated-body (Rust ABA) backend, not its replacement.
"""
from __future__ import annotations

from collections.abc import Callable
import numpy as np


InverseDynamics = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def _single_forward_dynamics_reference(
    q: np.ndarray,
    v: np.ndarray,
    tau: np.ndarray,
    gravity: np.ndarray,
    inverse_dynamics: InverseDynamics,
) -> np.ndarray:
    mass, bias = mass_and_bias_reference(q, v, gravity, inverse_dynamics)
    try:
        return np.linalg.solve(mass, tau - bias)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError("forward dynamics mass matrix is singular") from exc


def mass_and_bias_reference(q, v, gravity, inverse_dynamics):
    dof = q.shape[0]
    zero_acceleration = np.zeros(dof, dtype=q.dtype)
    bias = np.asarray(inverse_dynamics(q, v, zero_acceleration, gravity), dtype=float)

    # RNEA(q, 0, e_i, 0) gives column i of M(q).  Repeating q/v allows an
    # existing batched inverse-dynamics provider to produce all columns in
    # one call without materialising a Jacobian in Python.
    q_basis = np.broadcast_to(q, (dof, dof)).copy()
    v_zero = np.zeros((dof, dof), dtype=q.dtype)
    mass = np.asarray(
        inverse_dynamics(q_basis, v_zero, np.eye(dof, dtype=q.dtype), np.zeros(3)),
        dtype=float,
    ).T
    return mass, bias


def forward_dynamics_reference(
    q: np.ndarray,
    v: np.ndarray,
    tau: np.ndarray,
    gravity: np.ndarray,
    inverse_dynamics: InverseDynamics,
) -> np.ndarray:
    """Solve ``M(q) a + bias(q, v, gravity) = tau``.

    Inputs are validated by the public Kots API.  This function supports a
    single sample ``(dof,)`` or a flat batch ``(batch, dof)``.
    """
    if q.ndim == 1:
        return _single_forward_dynamics_reference(q, v, tau, gravity, inverse_dynamics)
    if q.shape[0] == 0:
        return np.empty_like(q)
    return np.stack([
        _single_forward_dynamics_reference(qi, vi, taui, gravity, inverse_dynamics)
        for qi, vi, taui in zip(q, v, tau)
    ])


def inverse_dynamics_numpy(robot, q, v, a, gravity):
    """Compute generalized efforts through the NumPy outward recurrence."""
    from ...core.motion import MotionTensor
    from ...outward.state import build_dynamics_outward_state

    if any(link.dof for link in robot.links) or any(
        joint.type not in ("fixed", "revolute", "prismatic") for joint in robot.joints
    ):
        raise NotImplementedError("NumPy inverse dynamics supports rigid links and fixed/revolute/prismatic joints")
    if q.size == 0:
        return np.zeros_like(q)
    motion = MotionTensor.from_dof_order(np.stack((q, v, a), axis=-1), robot.motion_owners()).to_flat_owner_major().data
    state = build_dynamics_outward_state(robot, motion, dynamics_order=1, gravity=gravity)
    efforts = np.zeros_like(q)
    for joint in robot.joints:
        if joint.dof:
            efforts[..., joint.dof_index:joint.dof_index + joint.dof] = np.asarray(
                state.joint_torque[joint.name]).reshape(q.shape[:-1] + (joint.dof,))
    return efforts
