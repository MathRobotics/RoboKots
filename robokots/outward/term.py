from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, Tuple
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
        """Return the dimension of this variable."""
        return int(np.asarray(self.x).size)


def pack(vars: Sequence[Variable]) -> Array:
    """
    Concatenate all variable vectors into a single vector.

    Returns:
        x_all: shape (n_total,)
    """
    if len(vars) == 0:
        return np.zeros((0,), dtype=float)
    return np.concatenate([np.asarray(v.x).reshape(-1) for v in vars], axis=0)


def total_dim(vars: Sequence[Variable]) -> int:
    """Return the total dimension of all variables."""
    return int(sum(v.dim() for v in vars))


# ============================================================
# Quantity: y(x) with Jacobian dy/dx
# ============================================================
class Quantity(Protocol):
    """
    Physical or mathematical quantity y(x) with its Jacobian dy/dx.
    """
    name: str
    out_dim: int                 # output dimension m
    vars: Sequence[Variable]     # variables involved in this quantity

    def value(self) -> Array:
        """
        Compute the value y(x).

        Returns:
            y: shape (m,) or (m, 1)
        """
        ...

    def jacobian(self) -> Array:
        """
        Compute the Jacobian dy/dx.

        Returns:
            J: shape (m, n_total)
        """
        ...


# ============================================================
# Residual: r(x) and its Jacobian
# ============================================================
class Residual(Protocol):
    """
    Residual function r(x) used in least-squares optimization.
    """
    name: str
    vars: Sequence[Variable]
    m: int  # residual dimension

    def evaluate(self) -> Tuple[Array, Array]:
        """
        Evaluate the residual and its Jacobian.

        Returns:
            r: shape (m,)
            J: shape (m, n_total)
        """
        ...


def _ensure_rJ_shapes(
    name: str,
    r: Array,
    J: Array,
    m: int,
    n_total: int,
) -> Tuple[Array, Array]:
    """
    Validate and normalize shapes of residual and Jacobian.
    """
    r = np.asarray(r).reshape(-1)
    J = np.asarray(J)

    if r.size != m:
        raise ValueError(f"{name}: residual size mismatch. expected m={m}, got {r.size}.")
    if J.ndim != 2:
        raise ValueError(f"{name}: Jacobian must be 2D. got shape {J.shape}.")
    if J.shape != (m, n_total):
        raise ValueError(
            f"{name}: Jacobian shape mismatch. expected {(m, n_total)}, got {J.shape}."
        )
    return r, J


# ============================================================
# Residual implementation
# ============================================================
@dataclass
class VectorSquaredSumResidual:
    """
    Residual corresponding to minimizing ||v(x)||^2.

    This is represented in least-squares form as:
        r(x) = v(x)
        J(x) = dv/dx
    """
    name: str
    quantity: Quantity

    @property
    def vars(self) -> Sequence[Variable]:
        """Variables involved in this residual."""
        return self.quantity.vars

    @property
    def m(self) -> int:
        """Residual dimension."""
        return int(self.quantity.out_dim)

    def evaluate(self) -> Tuple[Array, Array]:
        r = np.asarray(self.quantity.value()).reshape(-1)
        J = np.asarray(self.quantity.jacobian())

        n_total = total_dim(self.vars)
        return _ensure_rJ_shapes(self.name, r, J, self.m, n_total)


# ============================================================
# Cost functions (residual transformation)
# ============================================================
class Cost(Protocol):
    """
    Transform a residual (r, J) into a weighted or robustified form.
    """
    name: str

    def apply(self, r: Array, J: Array) -> Tuple[Array, Array]:
        """
        Apply the cost to residual and Jacobian.

        Returns:
            r_tilde, J_tilde
        """
        ...


@dataclass
class L2Cost:
    """
    Standard least-squares cost: no modification.
    """
    name: str = "L2"

    def apply(self, r: Array, J: Array) -> Tuple[Array, Array]:
        return np.asarray(r).reshape(-1), np.asarray(J)


@dataclass
class DiagonalWeightCost:
    """
    Weighted least squares with diagonal weights w >= 0.

    Objective:
        minimize sum_i w_i * r_i^2
        == || sqrt(W) r ||^2
    """
    w: Array  # shape (m,)
    name: str = "DiagWeight"

    def __post_init__(self):
        self.w = np.asarray(self.w).reshape(-1)
        if np.any(self.w < 0):
            raise ValueError("DiagonalWeightCost: w must be >= 0.")

    def apply(self, r: Array, J: Array) -> Tuple[Array, Array]:
        r = np.asarray(r).reshape(-1)
        J = np.asarray(J)

        if r.size != self.w.size:
            raise ValueError(
                f"DiagonalWeightCost: size mismatch. r has {r.size}, w has {self.w.size}."
            )
        if J.ndim != 2 or J.shape[0] != r.size:
            raise ValueError(
                "DiagonalWeightCost: Jacobian first dimension must match residual size."
            )

        sw = np.sqrt(self.w)  # shape (m,)
        return sw * r, sw[:, None] * J


@dataclass
class ScalarWeightCost:
    """
    Scalar-weighted least squares.

    Objective:
        minimize w * ||r||^2
        == || sqrt(w) r ||^2
    """
    w: float
    name: str = "ScalarWeight"

    def __post_init__(self):
        if self.w < 0:
            raise ValueError("ScalarWeightCost: w must be >= 0.")

    def apply(self, r: Array, J: Array) -> Tuple[Array, Array]:
        r = np.asarray(r).reshape(-1)
        J = np.asarray(J)

        sw = float(np.sqrt(self.w))
        return sw * r, sw * J


@dataclass
class HuberCost:
    """
    Huber robust cost applied to a vector residual.

    Weight definition:
        w = 1                     if ||r|| <= delta
            delta / ||r||         otherwise

    The transformed residual is:
        r' = sqrt(w) r
        J' = sqrt(w) J
    """
    delta: float
    name: str = "Huber"

    def __post_init__(self):
        if self.delta <= 0:
            raise ValueError("HuberCost: delta must be > 0.")

    def apply(self, r: Array, J: Array) -> Tuple[Array, Array]:
        r = np.asarray(r).reshape(-1)
        J = np.asarray(J)

        nr = float(np.linalg.norm(r)) + 1e-12
        w = 1.0 if nr <= self.delta else (self.delta / nr)
        sw = float(np.sqrt(w))

        return sw * r, sw * J


# ============================================================
# Helper: residual + cost composition
# ============================================================
def evaluate_residual_with_cost(
    residual: Residual,
    cost: Cost,
) -> Tuple[Array, Array]:
    """
    Evaluate a residual and apply a cost function.

    Returns:
        r_tilde: weighted / robustified residual
        J_tilde: corresponding Jacobian
    """
    r, J = residual.evaluate()
    r2, J2 = cost.apply(r, J)

    r2 = np.asarray(r2).reshape(-1)
    J2 = np.asarray(J2)

    if J2.ndim != 2 or J2.shape[0] != r2.size:
        raise ValueError(
            f"evaluate_residual_with_cost: incompatible shapes "
            f"r={r2.shape}, J={J2.shape} after applying cost '{cost.name}'."
        )

    return r2, J2
