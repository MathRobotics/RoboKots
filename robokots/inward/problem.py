from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Protocol
import numpy as np

from robokots.outward.term import VariablePack, Residual, Cost

Array = np.ndarray
# - VariablePack: has .n_total, .slices
# - Residual: .name, .vars, .evaluate() -> (r, blocks)
# - Cost: .apply(r, blocks) -> (r2, blocks2)

@dataclass
class Problem:
    """
    A collection of (residual, cost) terms assembled into one stacked LS problem.
    The ONLY place that defines the global Jacobian column order is VariablePack.
    """
    variables: "VariablePack"
    terms: List[Tuple["Residual", "Cost"]]

    def linearize(self) -> Tuple[Array, Array]:
        rs: List[Array] = []
        Js: List[Array] = []

        n_total = self.variables.n_total
        slices = self.variables.slices

        if len(self.terms) == 0:
            raise ValueError("Problem.linearize: no terms.")

        for res, cost in self.terms:
            r, blocks = res.evaluate()          # (m,), [ (m, dim(var_i)), ... ]
            r2, blocks2 = cost.apply(r, blocks) # same structure after weighting/robust

            r2 = np.asarray(r2, dtype=float).reshape(-1)
            m = int(r2.size)

            # Assemble global Jacobian Jg (m, n_total)
            Jg = np.zeros((m, n_total), dtype=float)

            if len(blocks2) != len(res.vars):
                raise ValueError(
                    f"Problem.linearize: len(blocks) mismatch in term '{res.name}'. "
                    f"blocks={len(blocks2)}, vars={len(res.vars)}"
                )

            for v, B in zip(res.vars, blocks2):
                B = np.asarray(B, dtype=float)
                if B.shape[0] != m:
                    raise ValueError(
                        f"Problem.linearize: row mismatch in term '{res.name}', var '{v.name}'. "
                        f"r has m={m}, block has {B.shape}."
                    )

                if v.name not in slices:
                    raise ValueError(
                        f"Problem.linearize: var '{v.name}' not found in VariablePack (term '{res.name}')."
                    )

                s, e = slices[v.name]
                nv = e - s
                if B.shape[1] != nv:
                    raise ValueError(
                        f"Problem.linearize: col mismatch in term '{res.name}', var '{v.name}'. "
                        f"expected block (m,{nv}), got {B.shape}."
                    )

                Jg[:, s:e] = B

            rs.append(r2)
            Js.append(Jg)

        r_all = np.concatenate(rs, axis=0)
        J_all = np.vstack(Js)
        return r_all, J_all

    def cost_value(self) -> float:
        r_all, _ = self.linearize()
        return float(r_all @ r_all)
