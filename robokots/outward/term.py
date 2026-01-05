from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, Tuple, Optional, List
import numpy as np

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
        # name collision is a common source of silent bugs
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
# Quantity: y(x) with block Jacobians
# ============================================================
class Quantity(Protocol):
    name: str
    out_dim: int
    vars: Sequence[Variable]

    def value(self) -> Array:
        """y: (m,)"""
        ...

    def jacobian_blocks(self) -> Sequence[Array]:
        """
        blocks[i]: (m, vars[i].dim())
        i corresponds to vars[i]
        """
        ...

@dataclass
class CachedQuantity:
    """
    Wrap any Quantity and cache both:
      - y = value()
      - blocks = jacobian_blocks()
    per (variables.revision, variables.get()).

    Key behavior:
      - If solver calls value() then jacobian_blocks() for the same x,
        underlying forward is executed only once.
      - If multiple residuals share the SAME CachedQuantity instance,
        the cache is shared too.
    """
    base: "Quantity"
    variables: "VariablePack"

    # passthrough fields
    name: str = ""
    out_dim: int = 0
    vars: Sequence["Variable"] = ()

    # cache
    _last_revision: Optional[int] = None
    _last_x: Optional[Array] = None
    _cached_value: Optional[Array] = None
    _cached_blocks: Optional[List[Array]] = None

    def __post_init__(self) -> None:
        self.name = getattr(self.base, "name", "CachedQuantity")
        self.out_dim = int(getattr(self.base, "out_dim"))
        self.vars = getattr(self.base, "vars")

    def _cache_key_matches(self) -> bool:
        if self._last_revision is None or self._last_x is None:
            return False
        x = self.variables.get()
        return (
            int(self.variables.revision) == int(self._last_revision)
            and x.shape == self._last_x.shape
            and np.array_equal(x, self._last_x)
        )

    def _evaluate_both(self) -> None:
        """
        Evaluate both value and jacobian_blocks exactly once.
        Important: call base.value() and base.jacobian_blocks() back-to-back
        after state update is implicitly done inside base (or via its own caching).
        """
        v = np.asarray(self.base.value(), dtype=float).reshape(-1)
        blocks = [np.asarray(B, dtype=float) for B in self.base.jacobian_blocks()]

        self._cached_value = v
        self._cached_blocks = blocks
        self._last_revision = int(self.variables.revision)
        self._last_x = self.variables.get().copy()

    def ensure(self) -> None:
        if self._cache_key_matches():
            return
        self._evaluate_both()

    def value(self) -> Array:
        self.ensure()
        return self._cached_value  # type: ignore[return-value]

    def jacobian_blocks(self) -> Sequence[Array]:
        self.ensure()
        return self._cached_blocks  # type: ignore[return-value]

# ============================================================
# Residual: r(x) and its *block* Jacobians
# ============================================================
class Residual(Protocol):
    name: str
    vars: Sequence[Variable]
    m: int

    def evaluate(self) -> Tuple[Array, Sequence[Array]]:
        """
        Returns:
            r: (m,)
            blocks[i]: (m, vars[i].dim())
        """
        ...


# ============================================================
# Residual implementation
# ============================================================
@dataclass
class VectorSquaredSumResidual:
    """
    Minimizing ||v(x)||^2 in least-squares form:
        r(x) = v(x)
        blocks = dv/d(vars[i])
    """
    name: str
    quantity: Quantity

    @property
    def vars(self) -> Sequence[Variable]:
        return self.quantity.vars

    @property
    def m(self) -> int:
        return int(self.quantity.out_dim)

    def evaluate(self) -> Tuple[Array, Sequence[Array]]:
        r = np.asarray(self.quantity.value(), dtype=float).reshape(-1)
        if r.size != self.m:
            raise ValueError(f"{self.name}: residual size mismatch. expected {self.m}, got {r.size}")

        blocks = self.quantity.jacobian_blocks()
        if len(blocks) != len(self.vars):
            raise ValueError(f"{self.name}: len(blocks) != len(vars): {len(blocks)} vs {len(self.vars)}")

        # block shape check (local)
        checked: List[Array] = []
        for v, B in zip(self.vars, blocks):
            B = np.asarray(B, dtype=float)
            if B.shape != (self.m, v.dim()):
                raise ValueError(
                    f"{self.name}: block shape mismatch for var '{v.name}'. "
                    f"expected {(self.m, v.dim())}, got {B.shape}"
                )
            checked.append(B)

        return r, checked
    
@dataclass
class EqualityConstraintResidual:
    """Treat g(x)=0 as a residual: r = g(x)."""
    name: str
    quantity: Quantity  # g(x)

    @property
    def vars(self) -> Sequence[Variable]:
        return self.quantity.vars

    @property
    def m(self) -> int:
        return int(self.quantity.out_dim)

    def evaluate(self) -> Tuple[Array, Sequence[Array]]:
        r = np.asarray(self.quantity.value(), dtype=float).reshape(-1)
        if r.size != self.m:
            raise ValueError(f"{self.name}: size mismatch.")
        blocks = self.quantity.jacobian_blocks()
        return r, blocks

@dataclass
class InequalityConstraintResidual:
    """
    Turn h(x) <= 0 into a penalty residual:
      r_i = max(0, h_i(x))
    Jacobian uses dh/dx only on active rows (h_i>0); inactive rows are zero.
    """
    name: str
    quantity: Quantity  # h(x)

    @property
    def vars(self) -> Sequence[Variable]:
        return self.quantity.vars

    @property
    def m(self) -> int:
        return int(self.quantity.out_dim)

    def evaluate(self) -> Tuple[Array, Sequence[Array]]:
        h = np.asarray(self.quantity.value(), dtype=float).reshape(-1)
        if h.size != self.m:
            raise ValueError(f"{self.name}: size mismatch.")

        blocks = [np.asarray(B, dtype=float) for B in self.quantity.jacobian_blocks()]
        # active mask
        active = (h > 0.0).astype(float)  # (m,) 0 or 1

        r = np.maximum(0.0, h)  # (m,)
        # row scaling per block (inactive rows become 0)
        blocks2 = [active[:, None] * B for B in blocks]
        return r, blocks2

# ============================================================
# Cost functions: now operate on (r, blocks)
# ============================================================
class Cost(Protocol):
    name: str

    def apply(self, r: Array, blocks: Sequence[Array]) -> Tuple[Array, Sequence[Array]]:
        """Return transformed (r, blocks)."""
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
    """
    minimize sum_i w_i * r_i^2 == || sqrt(W) r ||^2
    """
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

        sw = np.sqrt(self.w)  # (m,)
        r2 = sw * r

        blocks2: List[Array] = []
        for B in blocks:
            B = np.asarray(B, dtype=float)
            if B.shape[0] != r.size:
                raise ValueError("DiagonalWeightCost: block row size must match residual size.")
            blocks2.append(sw[:, None] * B)

        return r2, blocks2


@dataclass
class ScalarWeightCost:
    """
    minimize w * ||r||^2 == || sqrt(w) r ||^2
    """
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
    """
    Vector Huber via a single scalar weight on the whole residual vector:
        w = 1                  if ||r|| <= delta
            delta / ||r||      otherwise
    """
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
# Helper: residual + cost composition (still useful)
# ============================================================
def evaluate_residual_with_cost(residual: Residual, cost: Cost) -> Tuple[Array, Sequence[Array]]:
    r, blocks = residual.evaluate()
    r2, blocks2 = cost.apply(r, blocks)

    r2 = np.asarray(r2, dtype=float).reshape(-1)
    blocks2 = [np.asarray(B, dtype=float) for B in blocks2]

    # sanity: all blocks must have row size == len(r2)
    m = r2.size
    for B in blocks2:
        if B.ndim != 2 or B.shape[0] != m:
            raise ValueError(
                f"evaluate_residual_with_cost: incompatible shapes after cost '{cost.name}'. "
                f"r={r2.shape}, block={B.shape}"
            )
    return r2, blocks2


# ============================================================
# Problem: the ONLY place that assembles global J and stacks
# ============================================================
@dataclass
class Problem:
    variables: VariablePack
    terms: Sequence[Tuple[Residual, Cost]]  # [(residual, cost), ...]

    _last_x: Array | None = None
    _last_revision: int | None = None
    _last_r: Array | None = None
    _last_J: Array | None = None

    _last_set_x: Array | None = None

    def set_from_vector(self, x: Array) -> None:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.variables.n_total:
            raise ValueError(f"set_from_vector: expected {self.variables.n_total}, got {x.size}")

        # Same x => do nothing (NO assignment, NO revision bump)
        if self._last_set_x is not None and np.array_equal(x, self._last_set_x):
            return

        # Scatter x into each Variable by slices
        for v in self.variables.vars:
            s, e = self.variables.slices[v.name]
            expected = e - s
            if v.dim() != expected:
                raise ValueError(
                    f"set_from_vector: variable '{v.name}' dim changed. "
                    f"slices expects {expected}, actual {v.dim()}"
                )
            v.x = x[s:e].copy()   # copy: avoid aliasing x buffer

        self.variables.revision += 1
        self._last_set_x = x.copy()

    def linearize(self) -> Tuple[Array, Array]:
        x = self.variables.get()
        rev = int(self.variables.revision)
        if (
            self._last_x is not None
            and self._last_revision == rev
            and x.shape == self._last_x.shape
            and np.array_equal(x, self._last_x)
        ):
            return self._last_r, self._last_J

        r_list: List[Array] = []
        J_list: List[Array] = []

        n_total = self.variables.n_total
        slices = self.variables.slices

        for residual, cost in self.terms:
            r, blocks = evaluate_residual_with_cost(residual, cost)
            m = int(r.size)

            Jg = np.zeros((m, n_total), dtype=float)
            for v, B in zip(residual.vars, blocks):
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

        self._last_x = x.copy()
        self._last_revision = rev
        self._last_r = r_all
        self._last_J = J_all

        return r_all, J_all

    def cost_value(self) -> float:
        r_all, _ = self.linearize()
        return float(r_all @ r_all)
