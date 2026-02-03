import numpy as np
from typing import Any, Optional, Callable

from robokots.inward.problem import Problem
from robokots.inward.term import VariablePack


def solve_gauss_newton(
    problem: Problem,
    variables: VariablePack,
    max_iters: int = 20,
    *,
    ctx: Any = None,
    required: Any = None,
    tol_r: float = 1e-10,
    tol_dx: float = 1e-12,
    on_iter: Optional[Callable[[int, float, float], None]] = None,
) -> None:
    """
    Minimal Gauss-Newton loop for Expr-based Problem.

    Expected design:
      - Problem.linearize(ctx=..., time=..., required=...) is valid.
      - StateCache update is done here (once per iteration) if ctx has .cache.
      - VariablePack.revision must change when variables change (cache invalidation key).
        If VariablePack.apply_dx does not bump revision, we bump here.
    """

    for k in range(max_iters):
        # 1) Update state cache once per evaluation point (if available)
        if ctx is not None and hasattr(ctx, "state"):
            time = getattr(ctx, "time", None)
            # ctx.state is StateCache-like; update_if_needed(pack, time, required)
            ctx.state.update_if_needed(variables, time=time, required=required)

        # 2) Linearize
        try:
            time = getattr(ctx, "time", None)
            r_all, J_all = problem.linearize(ctx=ctx, time=time, required=required)
        except TypeError:
            # Backward compat if Problem.linearize() has no kwargs yet
            r_all, J_all = problem.linearize()

        nr = float(np.linalg.norm(r_all))
        if on_iter is not None:
            on_iter(k, nr, 0.0)

        if nr < tol_r:
            break

        # 3) Solve normal equations (basic; you may add damping later)
        lhs = J_all.T @ J_all
        rhs = -J_all.T @ r_all

        dx, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        ndx = float(np.linalg.norm(dx))

        if on_iter is not None:
            on_iter(k, nr, ndx)

        if ndx < tol_dx:
            break

        # 4) Apply update
        variables.apply_dx(dx)

        # IMPORTANT: bump revision so StateCache knows x changed.
        # (If you later move revision bump into apply_dx, remove this.)
        if hasattr(variables, "revision"):
            variables.revision += 1
