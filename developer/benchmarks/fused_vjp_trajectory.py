"""Trajectory-scale fused VJP benchmark for direct-collocation objectives.

The timed region intentionally excludes state construction.  It measures the
derivative work after a 509-step CMTM dynamics trajectory has been cached,
which matches a DOC objective evaluation that has already assembled its
primal state.
"""
from __future__ import annotations

import argparse

import numpy as np

from robokots.core.state import StateType
from robokots.kots import Kots

from .common import build_model, format_time, measure, select_unit


def _embed(kots: Kots, state: StateType, value: np.ndarray, order: int) -> np.ndarray:
    return kots._embed_motion_order_rhs(
        np.asarray(value), StateType.max_time_order([state]), order, rhs_is_matrix=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=509)
    parser.add_argument("--dof", type=int, default=69)
    parser.add_argument("--rhs-cols", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1 or args.dof < 1 or args.rhs_cols < 1:
        raise ValueError("steps, dof, and rhs-cols must be positive")

    rng = np.random.default_rng(20260831)
    order = 5
    batch_shape = (args.steps,)
    kots = Kots.from_json_data(build_model(args.dof, "serial"), order=order)
    kots.import_motions(rng.standard_normal(batch_shape + (args.dof * order,)))
    kots.dynamics(order=order, backend="rust", gravity=(0.2, -0.3, -9.81), materialize_dict=False)

    torque = StateType("total_joint", "total_joint", "torque")
    torque_d1 = StateType("total_joint", "total_joint", "torque_diff1")
    torque_d2 = StateType("total_joint", "total_joint", "torque_diff2")
    energy = StateType("total_body", "total_body", "kinetic_energy")
    torque_rhs = rng.standard_normal(batch_shape + (args.dof, args.rhs_cols))
    torque_d1_rhs = rng.standard_normal(batch_shape + (args.dof, args.rhs_cols))
    torque_d2_rhs = rng.standard_normal(batch_shape + (args.dof, args.rhs_cols))
    energy_rhs = rng.standard_normal(batch_shape + (1, args.rhs_cols))
    requests = [
        (torque, torque_rhs),
        (torque_d1, torque_d1_rhs),
        (torque_d2, torque_d2_rhs),
        (energy, energy_rhs),
    ]

    def separate():
        return sum(
            _embed(kots, state, kots.jacobian_transpose_mul(state, rhs), order)
            for state, rhs in requests
        )

    def fused():
        return kots.jacobian_transpose_mul_many(requests)

    expected = separate()
    actual = fused()
    # The two paths have identical derivatives but accumulate the torque and
    # energy seeds in a different floating-point order on long trajectories.
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)
    separate_stats = measure(separate, repeats=args.repeat, warmup=args.warmup)
    fused_stats = measure(fused, repeats=args.repeat, warmup=args.warmup)
    unit = select_unit([separate_stats["mean_ms"], fused_stats["mean_ms"]])
    speedup = separate_stats["mean_ms"] / fused_stats["mean_ms"]
    print("=== fused trajectory VJP ===")
    print(f"steps={args.steps}, dof={args.dof}, order={order}, rhs_cols={args.rhs_cols}")
    print(f"separate  {format_time(separate_stats['mean_ms'], unit)}")
    print(f"fused     {format_time(fused_stats['mean_ms'], unit)}")
    print(f"speedup   {speedup:.2f}x")
    print(f"max_error {np.max(np.abs(actual - expected)):.3e}")


if __name__ == "__main__":
    main()
