from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, List, Tuple
import numpy as np

Array = np.ndarray

# ---------------------------
# Variable block (x)
# ---------------------------
@dataclass
class Variable:
    name: str
    x: Array  # shape (n,)

    def dim(self) -> int:
        return int(self.x.size)


def pack(vars: List[Variable]) -> Array:
    return np.concatenate([v.x.reshape(-1) for v in vars], axis=0)


class Quantity(Protocol):
    """Any physical quantity y(x) with Jacobian dy/dx."""
    name: str
    out_dim: int
    vars: List[Variable]

    def value(self) -> Array:
        """y: (m,)"""
        ...

    def jacobian(self) -> Array:
        """J: (m, n_total)"""
        ...

# ---------------------------
# Residual interface
# r(x) and J(x)
# ---------------------------
class Residual(Protocol):
    name: str
    vars: List[Variable]
    m: int  # residual dimension

    def evaluate(self) -> Tuple[Array, Array]:
        """
        Returns:
            r: (m,)
            J: (m, n_total)  where n_total = sum(var.dim())
        """
        ...


# ---------------------------
# Cost (loss) interface
# Apply weights / robustification to r, J
# ---------------------------
class Cost(Protocol):
    name: str

    def weight(self, r: Array) -> float:
        """Return scalar weight w >= 0 applied to r and J (sqrt form)."""
        ...

    def apply(self, r: Array, J: Array) -> Tuple[Array, Array]:
        w = self.weight(r)
        sw = np.sqrt(w)
        return sw * r, sw * J

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .interface import Quantity

Array = np.ndarray

@dataclass
class VectorSquaredSumResidual:
    """
    Minimize ||v(x)||^2 by returning residual r=v and Jacobian J=dv/dx.
    """
    name: str
    quantity: "Quantity"   # value()->(m,), jacobian()->(m,N)

    def evaluate(self) -> Tuple[Array, Array]:
        r = self.quantity.value().reshape(-1)      # (m,)
        J = self.quantity.jacobian()               # (m,N)
        return r, J
    
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Tuple, Optional
import numpy as np

Array = np.ndarray


class Cost(Protocol):
    """Transforms (r, J) -> (r_tilde, J_tilde)."""
    name: str

    def apply(self, r: Array, J: Array) -> Tuple[Array, Array]:
        ...


@dataclass
class L2Cost:
    """No change: minimize ||r||^2."""
    name: str = "L2"

    def apply(self, r: Array, J: Array) -> Tuple[Array, Array]:
        return r, J


@dataclass
class DiagonalWeightCost:
    """
    Weighted least squares with diagonal weights w (>=0):
        minimize sum_i w_i * r_i^2  == || sqrt(W) r ||^2
    """
    w: Array  # shape (m,)
    name: str = "DiagWeight"

    def __post_init__(self):
        self.w = np.asarray(self.w).reshape(-1)
        if np.any(self.w < 0):
            raise ValueError("DiagonalWeightCost: w must be >= 0.")

    def apply(self, r: Array, J: Array) -> Tuple[Array, Array]:
        r = np.asarray(r).reshape(-1)
        if r.size != self.w.size:
            raise ValueError(
                f"DiagonalWeightCost: size mismatch. r has {r.size}, w has {self.w.size}."
            )
        sw = np.sqrt(self.w).reshape(-1, 1)  # (m,1)
        return (sw[:, 0] * r), (sw * J)


@dataclass
class ScalarWeightCost:
    """minimize w * ||r||^2  == || sqrt(w) r ||^2"""
    w: float
    name: str = "ScalarWeight"

    def __post_init__(self):
        if self.w < 0:
            raise ValueError("ScalarWeightCost: w must be >= 0.")

    def apply(self, r: Array, J: Array) -> Tuple[Array, Array]:
        sw = float(np.sqrt(self.w))
        return sw * r, sw * J


@dataclass
class HuberCost:
    """
    Robustify a vector residual using Huber weight based on its norm.
    This is the common simple robustification:
        r' = sqrt(w) r,  J' = sqrt(w) J
    with
        w = 1                     if ||r|| <= delta
            delta / ||r||         otherwise
    """
    delta: float
    name: str = "Huber"

    def __post_init__(self):
        if self.delta <= 0:
            raise ValueError("HuberCost: delta must be > 0.")

    def apply(self, r: Array, J: Array) -> Tuple[Array, Array]:
        r = np.asarray(r).reshape(-1)
        s = float(r @ r)
        nr = float(np.sqrt(s)) + 1e-12
        w = 1.0 if nr <= self.delta else (self.delta / nr)
        sw = float(np.sqrt(w))
        return sw * r, sw * J

