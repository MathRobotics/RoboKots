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

from pathlib import Path

from robokots.inward.opt import solve_gauss_newton
from robokots.inward.builder import build_problem_from_project_file
from robokots.inward.builder import prepare_problem_for_solve

def main() -> None:

    project_path = Path(__file__).with_name("project_task_file.json")
    problem, ctx, solver = build_problem_from_project_file(project_path)

    print("Initial q:", ctx.pack.get())

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
    print("Final q:", ctx.pack.get())
    print("Final cost:", problem.cost_value(ctx=ctx, time=getattr(ctx, "time", None), required=required))


if __name__ == "__main__":
    main()
