from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, Tuple, List, Callable, Any
import numpy as np

from typing import Protocol, Iterable
from robokots.core.state_cache import StateKey


Array = np.ndarray


# ============================================================
# Variable block (decision variables)
# ============================================================
@dataclass
class Variable:
    name: str
    x: Array  # shape (n,)

    def dim(self) -> int:
        return int(np.asarray(self.x).size)


def pack(vars: Sequence[Variable]) -> Array:
    if len(vars) == 0:
        return np.zeros((0,), dtype=float)
    return np.concatenate([np.asarray(v.x).reshape(-1) for v in vars], axis=0)


def total_dim(vars: Sequence[Variable]) -> int:
    return int(sum(v.dim() for v in vars))


# ============================================================
# VariablePack: global ordering (the ONLY source of column order)
# ============================================================
@dataclass
class VariablePack:
    vars: Sequence[Variable]
    revision: int = 0  # optional revision number for caching

    def __post_init__(self) -> None:
        names = [v.name for v in self.vars]
        if len(names) != len(set(names)):
            raise ValueError(f"VariablePack: duplicate variable names: {names}")

        self.slices: dict[str, Tuple[int, int]] = {}
        col = 0
        for v in self.vars:
            n = v.dim()
            self.slices[v.name] = (col, col + n)
            col += n
        self.n_total = int(col)

    def get(self) -> Array:
        return pack(self.vars)

    def apply_dx(self, dx: Array) -> None:
        dx = np.asarray(dx, dtype=float).reshape(-1)
        if dx.size != self.n_total:
            raise ValueError(f"apply_dx: expected {self.n_total}, got {dx.size}")

        for v in self.vars:
            s, e = self.slices[v.name]
            v.x = np.asarray(v.x, dtype=float).reshape(-1) + dx[s:e]


# ============================================================
# EvalContext: what Expr is allowed to see
#   - read-only handle to caches / model state.
#   - keep it as Any here to avoid inward/core circular deps in term.py.
# ============================================================
@dataclass(frozen=True)
class EvalContext:
    """
    Minimal, dependency-light evaluation context.

    Recommended contents (you can expand later):
      - pack: VariablePack
      - state_cache: StateCache (or similar)
      - time_grid: TimeGrid (optional)
      - revision: int (optional global revision for caching)
    """
    pack: VariablePack
    state: Any = None      # e.g., StateCache or a computed state_dict
    time: Any = None       # e.g., TimeGrid
    revision: int = 0


# ============================================================
# Expr: returns residual vector r(x) and block Jacobians
# ============================================================
class Expr(Protocol):
    name: str
    vars: Sequence[Variable]

    def eval(self, ctx: EvalContext) -> Tuple[Array, Sequence[Array]]:
        """
        Returns:
          r: (m,)
          blocks[i]: (m, vars[i].dim()) aligned with self.vars
        """
        ...

    def deps(self) -> Iterable[StateKey]:
        """
        Returns:
          Iterable of StateKey that this Expr depends on.
        """
        ...


# ============================================================
# Example helper Exprs (optional convenience)
# ============================================================
@dataclass
class DirectVectorExpr:
    """
    A minimal adapter for legacy callables.

    fn_value(ctx) -> (m,)
    fn_blocks(ctx) -> list of blocks aligned to vars
    """
    name: str
    vars: Sequence[Variable]
    fn_value: Callable[[EvalContext], Array]
    fn_blocks: Callable[[EvalContext], Sequence[Array]]

    def eval(self, ctx: EvalContext) -> Tuple[Array, Sequence[Array]]:
        r = np.asarray(self.fn_value(ctx), dtype=float).reshape(-1)

        blocks = [np.asarray(B, dtype=float) for B in self.fn_blocks(ctx)]
        if len(blocks) != len(self.vars):
            raise ValueError(f"{self.name}: len(blocks) != len(vars): {len(blocks)} vs {len(self.vars)}")

        m = int(r.size)
        checked: List[Array] = []
        for v, B in zip(self.vars, blocks):
            if B.shape != (m, v.dim()):
                raise ValueError(
                    f"{self.name}: block shape mismatch for var '{v.name}'. "
                    f"expected {(m, v.dim())}, got {B.shape}"
                )
            checked.append(B)
        return r, checked


# ============================================================
# Cost functions: operate on (r, blocks)
# ============================================================
class Cost(Protocol):
    name: str

    def apply(self, r: Array, blocks: Sequence[Array]) -> Tuple[Array, Sequence[Array]]:
        ...


@dataclass
class L2Cost:
    name: str = "L2"

    def apply(self, r: Array, blocks: Sequence[Array]) -> Tuple[Array, Sequence[Array]]:
        r = np.asarray(r, dtype=float).reshape(-1)
        blocks2 = [np.asarray(B, dtype=float) for B in blocks]
        return r, blocks2


@dataclass
class DiagonalWeightCost:
    w: Array  # (m,)
    name: str = "DiagWeight"

    def __post_init__(self) -> None:
        self.w = np.asarray(self.w, dtype=float).reshape(-1)
        if np.any(self.w < 0):
            raise ValueError("DiagonalWeightCost: w must be >= 0.")

    def apply(self, r: Array, blocks: Sequence[Array]) -> Tuple[Array, Sequence[Array]]:
        r = np.asarray(r, dtype=float).reshape(-1)
        if r.size != self.w.size:
            raise ValueError(f"DiagonalWeightCost: size mismatch. r={r.size}, w={self.w.size}")

        sw = np.sqrt(self.w)
        r2 = sw * r
        blocks2 = [sw[:, None] * np.asarray(B, dtype=float) for B in blocks]
        return r2, blocks2


@dataclass
class ScalarWeightCost:
    w: float
    name: str = "ScalarWeight"

    def __post_init__(self) -> None:
        if self.w < 0:
            raise ValueError("ScalarWeightCost: w must be >= 0.")

    def apply(self, r: Array, blocks: Sequence[Array]) -> Tuple[Array, Sequence[Array]]:
        r = np.asarray(r, dtype=float).reshape(-1)
        sw = float(np.sqrt(self.w))
        r2 = sw * r
        blocks2 = [sw * np.asarray(B, dtype=float) for B in blocks]
        return r2, blocks2


@dataclass
class HuberCost:
    delta: float
    name: str = "Huber"

    def __post_init__(self) -> None:
        if self.delta <= 0:
            raise ValueError("HuberCost: delta must be > 0.")

    def apply(self, r: Array, blocks: Sequence[Array]) -> Tuple[Array, Sequence[Array]]:
        r = np.asarray(r, dtype=float).reshape(-1)
        nr = float(np.linalg.norm(r)) + 1e-12
        w = 1.0 if nr <= self.delta else (self.delta / nr)
        sw = float(np.sqrt(w))
        r2 = sw * r
        blocks2 = [sw * np.asarray(B, dtype=float) for B in blocks]
        return r2, blocks2


# ============================================================
# Helper: expr + cost composition
# ============================================================
def evaluate_expr_with_cost(expr: Expr, cost: Cost, ctx: EvalContext) -> Tuple[Array, Sequence[Array]]:
    r, blocks = expr.eval(ctx)
    r2, blocks2 = cost.apply(r, blocks)

    r2 = np.asarray(r2, dtype=float).reshape(-1)
    blocks2 = [np.asarray(B, dtype=float) for B in blocks2]

    m = int(r2.size)
    if len(blocks2) != len(expr.vars):
        raise ValueError(
            f"evaluate_expr_with_cost: len(blocks) != len(vars) after cost '{cost.name}'. "
            f"{len(blocks2)} vs {len(expr.vars)}"
        )
    for v, B in zip(expr.vars, blocks2):
        if B.ndim != 2 or B.shape[0] != m or B.shape[1] != v.dim():
            raise ValueError(
                f"evaluate_expr_with_cost: incompatible shapes after cost '{cost.name}'. "
                f"r={r2.shape}, block={B.shape}, var='{v.name}', var_dim={v.dim()}"
            )
    return r2, blocks2


# ============================================================
# Problem: the ONLY place that assembles global J and stacks
# ============================================================
@dataclass
class Problem:
    variables: VariablePack
    terms: Sequence[Tuple[Expr, Cost]]  # [(expr, cost), ...]

    _last_revision: int | None = None
    _last_ctx_revision: int | None = None
    _last_r: Array | None = None
    _last_J: Array | None = None

    _last_set_x: Array | None = None

    def set_from_vector(self, x: Array) -> None:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.variables.n_total:
            raise ValueError(f"set_from_vector: expected {self.variables.n_total}, got {x.size}")

        if self._last_set_x is not None and np.array_equal(x, self._last_set_x):
            return

        for v in self.variables.vars:
            s, e = self.variables.slices[v.name]
            expected = e - s
            if v.dim() != expected:
                raise ValueError(
                    f"set_from_vector: variable '{v.name}' dim changed. "
                    f"slices expects {expected}, actual {v.dim()}"
                )
            v.x = x[s:e].copy()

        self.variables.revision += 1
        self._last_set_x = x.copy()

    def linearize(
        self,
        ctx: EvalContext | None = None,
        *,
        time: Any = None,
        required: Any = None,
    ) -> Tuple[Array, Array]:
        """
        Build stacked residual and Jacobian.

        ctx: EvalContext for Expr evaluation (optional).
        time/required: accepted for API compatibility; currently unused.
        """
        if ctx is None:
            ctx = EvalContext(pack=self.variables)

        rev = int(self.variables.revision)
        ctx_rev = int(getattr(ctx, "revision", 0))

        if (
            self._last_revision == rev
            and self._last_ctx_revision == ctx_rev
            and self._last_r is not None
            and self._last_J is not None
        ):
            return self._last_r, self._last_J

        r_list: List[Array] = []
        J_list: List[Array] = []

        n_total = int(self.variables.n_total)
        slices = self.variables.slices

        for expr, cost in self.terms:
            r, blocks = evaluate_expr_with_cost(expr, cost, ctx)
            m = int(r.size)

            Jg = np.zeros((m, n_total), dtype=float)
            for v, B in zip(expr.vars, blocks):
                if v.name not in slices:
                    raise ValueError(f"Problem: var '{v.name}' not found in VariablePack.")
                s, e = slices[v.name]
                if B.shape != (m, v.dim()):
                    raise ValueError(
                        f"Problem: block shape mismatch for var '{v.name}'. "
                        f"expected {(m, v.dim())}, got {B.shape}"
                    )
                Jg[:, s:e] = B

            r_list.append(r)
            J_list.append(Jg)

        r_all = np.concatenate(r_list, axis=0) if r_list else np.zeros((0,), dtype=float)
        J_all = np.vstack(J_list) if J_list else np.zeros((0, n_total), dtype=float)

        self._last_revision = rev
        self._last_ctx_revision = ctx_rev
        self._last_r = r_all
        self._last_J = J_all
        return r_all, J_all

    def cost_value(
        self,
        ctx: EvalContext | None = None,
        *,
        time: Any = None,
        required: Any = None,
    ) -> float:
        r_all, _ = self.linearize(ctx=ctx, time=time, required=required)
        return float(r_all @ r_all)
