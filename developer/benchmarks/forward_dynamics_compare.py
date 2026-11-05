"""Fair fixed-base ABA timing against Pinocchio.

The scalar benchmark reports both one-shot and cached RoboKots calls.  The
latter matches Pinocchio's normal usage (a persistent ``Data`` object), while
the former includes Python-result and workspace allocation costs.  Pinocchio
does not expose a batched ABA kernel, so its batch number is a Python loop.
"""

from __future__ import annotations

import argparse

import numpy as np

from robokots.kots import Kots

from .common import build_model, format_time, measure, select_unit
from .pinocchio_compare import _optional_pinocchio, build_pinocchio_model


def _report(label: str, stats: dict[str, float], pin_ms: float) -> None:
    unit = select_unit([stats["mean_ms"], pin_ms])
    ratio = stats["mean_ms"] / pin_ms if pin_ms else float("inf")
    print(
        f"  {label:24s} {format_time(stats['mean_ms'], unit):>10s} "
        f"(ratio={ratio:.2f}x)",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dof", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()
    pin = _optional_pinocchio()
    if pin is None:
        print("Pinocchio is not installed; skipping.")
        return

    rng = np.random.default_rng(20260827)
    gravity = np.array([0.2, -0.3, -9.81])
    print("=== fixed-base ABA: RoboKots vs Pinocchio ===")
    for dof in args.dof:
        model_data = build_model(dof, "humanoid")
        kots = Kots.from_json_data(model_data, order=3)
        robot = kots._rust_inverse_dynamics_robot()
        pin_model = build_pinocchio_model(pin, model_data)
        pin_model.gravity.linear = gravity
        pin_data = pin_model.createData()
        aba_data = robot.create_aba_data()
        q = rng.standard_normal(dof)
        v = rng.standard_normal(dof)
        inward_cache = kots.create_inward_cache().prepare(q, v, gravity)
        tau = rng.standard_normal(dof)
        q_batch = rng.standard_normal((args.batch, dof))
        v_batch = rng.standard_normal((args.batch, dof))
        tau_batch = rng.standard_normal((args.batch, dof))

        pin_scalar = measure(
            lambda: pin.aba(pin_model, pin_data, q, v, tau),
            repeats=args.repeat,
            warmup=args.warmup,
        )
        pin_batch = measure(
            lambda: [pin.aba(pin_model, pin_data, q_batch[i], v_batch[i], tau_batch[i]) for i in range(args.batch)],
            repeats=args.repeat,
            warmup=args.warmup,
        )
        print(f"dof={dof}, batch={args.batch}")
        print(f"  {'pinocchio scalar':24s} {format_time(pin_scalar['mean_ms'], select_unit([pin_scalar['mean_ms']]))}")
        _report("robokots one-shot", measure(lambda: robot.aba(q, v, tau, gravity), repeats=args.repeat, warmup=args.warmup), pin_scalar["mean_ms"])
        _report("robokots cached", measure(lambda: aba_data.compute(q, v, tau, gravity), repeats=args.repeat, warmup=args.warmup), pin_scalar["mean_ms"])
        _report("robokots public API", measure(lambda: kots.forward_dynamics(q, v, tau, gravity=gravity, backend="rust"), repeats=args.repeat, warmup=args.warmup), pin_scalar["mean_ms"])
        print(f"  {'pinocchio batch loop':24s} {format_time(pin_batch['mean_ms'], select_unit([pin_batch['mean_ms']]))}")
        _report("robokots aba_batch", measure(lambda: robot.aba_batch(q_batch, v_batch, tau_batch, gravity), repeats=args.repeat, warmup=args.warmup), pin_batch["mean_ms"])
        cached_many = np.broadcast_to(tau, (args.batch, dof)).copy()
        cached_many_stats = measure(lambda: inward_cache.forward_dynamics_many(cached_many), repeats=args.repeat, warmup=args.warmup)
        print(f"  {'inward cache same-state':24s} {format_time(cached_many_stats['mean_ms'], select_unit([cached_many_stats['mean_ms']]))}")


if __name__ == "__main__":
    main()
