"""
Inverse kinematics least-squares optimization demo (JSON task spec).

This example is intentionally thin:
- task.json is parsed
- library builder creates (problem, ctx)
- solver runs GN

All heavy logic lives in the library:
- StateCache.build_state (Kots/Pinocchio backend) in outward/
- Expr nodes in outward/ (cache readers)
- JSON -> Problem builder in inward/
"""
from __future__ import annotations

import json

from robokots.kots import Kots
from robokots.outward.backends import KotsBackend
from robokots.inward.opt import solve_gauss_newton
from robokots.inward.expr.json_builder import build_problem_from_spec

def main() -> None:

    kots = Kots.from_json_file("../model/2dof_arm.json", order=1)
    backend = KotsBackend(kots)

    with open("task.json", "r", encoding="utf-8") as f:
        spec = json.load(f)

    # ctx should contain at least: pack, cache, time
    problem, ctx = build_problem_from_spec(spec, backend=backend)

    print("Initial q:", ctx.pack.get())

    # If your solver already updates cache internally, you can remove these two lines
    ctx.cache.update_if_needed(ctx.pack, time=getattr(ctx, "time", None), required=None)
    print("Initial cost:", problem.cost_value(ctx=ctx, time=getattr(ctx, "time", None), required=None))

    solve_gauss_newton(problem, ctx.pack, max_iters=20, ctx=ctx)

    ctx.cache.update_if_needed(ctx.pack, time=getattr(ctx, "time", None), required=None)
    print("Final q:", ctx.pack.get())
    print("Final cost:", problem.cost_value(ctx=ctx, time=getattr(ctx, "time", None), required=None))


if __name__ == "__main__":
    main()
