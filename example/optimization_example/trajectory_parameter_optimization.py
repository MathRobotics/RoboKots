"""Trajectory optimization example driven by project.json.

This example is intentionally thin:
- project.json is parsed
- library builder creates (problem, ctx)
- solver runs Gauss-Newton

All heavy logic lives in the library:
- StateCache.build_state (Kots backend)
- Expr nodes (cache readers)
- JSON -> Problem builder (inward/)
"""
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from robokots.inward.builder import build_problem_from_project_file, prepare_problem_for_solve
from robokots.inward.opt import solve_gauss_newton


def main() -> None:
    project_path = Path(__file__).with_name("project_trajectory.json")
    problem, ctx, solver = build_problem_from_project_file(project_path)

    print("Initial traj:", ctx.pack.get())

    required = prepare_problem_for_solve(problem, ctx)
    print("Initial cost:", problem.cost_value(ctx=ctx, time=getattr(ctx, "time", None), required=required))

    solve_gauss_newton(
        problem,
        ctx.pack,
        max_iters=solver.max_iters,
        tol_r=solver.tol_r,
        tol_dx=solver.tol_dx,
        ctx=ctx,
        required=required,
    )

    prepare_problem_for_solve(problem, ctx, required=required)
    print("Final traj:", ctx.pack.get())
    print("Final cost:", problem.cost_value(ctx=ctx, time=getattr(ctx, "time", None), required=required))


if __name__ == "__main__":
    main()
