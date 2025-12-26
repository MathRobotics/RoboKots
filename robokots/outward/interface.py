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
