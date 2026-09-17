"""Evaluation-only time AD starting from ordinary ID(q, qdot, qddot).

The base ID always uses motion order 3 and undifferentiated torque. Higher
torque derivatives are nested total-time JVPs, not dynamics time recurrences.
The FK-based experiment in time_autodiff.py is intentionally left unchanged.
"""
import jax.numpy as jnp

from robokots.core.state_spec import StateType, keys_torque
from robokots.outward.diff.dynamics_jax import dynamics_state_vector_jax
from .time_autodiff import _flat_function, _validate_robot, total_time_derivative


def make_id_time_ad_value(robot, state, order, gravity=(0., 0., 0.)):
    """Return one joint's torque/time derivative as a function of motion.

    Inner time AD is forward JVP. Apply jacfwd/jacrev externally for a dense
    input Jacobian, optionally with jit. No joint trajectory is constructed.
    """
    _validate_robot(robot)
    if state.owner_type != "joint" or state.data_type not in keys_torque:
        raise ValueError("ID time AD requires one joint torque/torque_diffN state")
    if robot.joint(state.owner_name) is None:
        raise ValueError("Unknown joint")
    if state.frame_name not in (None, "local"):
        raise ValueError("ID time AD torque uses the local joint axis")
    if order < state.time_order:
        raise ValueError("Insufficient motion order")
    gravity = jnp.asarray(gravity)
    if gravity.shape != (3,):
        raise ValueError("gravity must have shape (3,)")
    base_state = StateType("joint", state.owner_name, "torque")

    def ordinary_id(motion):
        qva = motion[:, :3].reshape(-1)
        return dynamics_state_vector_jax(robot, qva, [base_state], 3, gravity)

    derivative = ordinary_id
    for _ in range(state.key_order - 1):
        derivative = total_time_derivative(derivative)
    return _flat_function(derivative, robot, order)
