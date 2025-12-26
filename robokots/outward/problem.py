from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Protocol
import numpy as np

from .interface import Residual, Cost

Array = np.ndarray


@dataclass
class Problem:
    """
    A collection of (residual, cost) terms that can be stacked into one big LS problem.
    """
    terms: List[Tuple[Residual, Cost]]

    def linearize(self) -> Tuple[Array, Array]:
        rs: List[Array] = []
        Js: List[Array] = []
        for res, cost in self.terms:
            r, J = res.evaluate()
            r = np.asarray(r).reshape(-1)
            J = np.asarray(J)
            if J.shape[0] != r.size:
                raise ValueError(
                    f"Problem.linearize: shape mismatch in term '{res.name}'. "
                    f"r has {r.size}, J has {J.shape}."
                )
            r2, J2 = cost.apply(r, J)
            rs.append(np.asarray(r2).reshape(-1))
            Js.append(np.asarray(J2))

        if len(rs) == 0:
            raise ValueError("Problem.linearize: no terms.")

        r_all = np.concatenate(rs, axis=0)
        J_all = np.vstack(Js)
        return r_all, J_all

    def cost_value(self) -> float:
        """
        Returns sum of squared residuals after cost.apply (i.e., ||r'||^2).
        Useful for debugging / line search.
        """
        r_all, _ = self.linearize()
        return float(r_all @ r_all)
