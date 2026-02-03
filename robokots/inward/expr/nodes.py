from dataclasses import dataclass
from typing import Sequence
import numpy as np

from robokots.core.state_cache import StateKey
from robokots.inward.term import Variable
from robokots.inward.term import EvalContext
from robokots.inward.term import Expr

@dataclass
class GetStateExpr:
    name: str
    vars: Sequence[Variable]          # 例: [q_var]
    key_value: StateKey               # 値のキー
    key_jac_q: StateKey               # ヤコビアンのキー（qに対する）
    # 将来 vars が増えるなら key_jac_map を持てばよい

    def deps(self):
        return [self.key_value, self.key_jac_q]

    def eval(self, ctx: EvalContext):
        sc = ctx.state  # StateCache想定
        y = np.asarray(sc.get(self.key_value), dtype=float).reshape(-1)
        J = np.asarray(sc.get(self.key_jac_q), dtype=float)
        # ブロック整合（vars が 1 個の最小ケース）
        if J.shape != (y.size, self.vars[0].dim()):
            raise ValueError(f"{self.name}: J shape mismatch: {J.shape} vs {(y.size, self.vars[0].dim())}")
        return y, [J]

@dataclass
class ConstantExpr:
    """
    Constant residual vector.

    - value: constant vector (m,)
    - vars: usually [] (no decision variables)
      But you can also pass through a vars list if your pipeline expects
      all Expr nodes to share the same vars list (less recommended).

    Jacobian blocks are zeros for each variable in `vars`.
    """
    name: str
    value: np.ndarray
    vars: Sequence[Variable] = ()

    @property
    def m(self) -> int:
        return int(np.asarray(self.value).size)

    def deps(self):
        return []

    def eval(self, ctx: EvalContext):
        y = np.asarray(self.value, dtype=float).reshape(-1)

        # Zero blocks aligned with `vars`
        blocks = []
        for v in self.vars:
            blocks.append(np.zeros((y.size, v.dim()), dtype=float))
        return y, blocks

@dataclass
class SubExpr:
    name: str
    a: Expr
    b: Expr

    @property
    def vars(self):
        return self.a.vars

    def deps(self):
        return list(self.a.deps()) + list(self.b.deps())

    def eval(self, ctx: EvalContext):
        ra, Ja = self.a.eval(ctx)
        rb, Jb = self.b.eval(ctx)
        if ra.shape != rb.shape:
            raise ValueError(f"{self.name}: shape mismatch {ra.shape} vs {rb.shape}")
        if len(Ja) != len(Jb):
            raise ValueError(f"{self.name}: block len mismatch {len(Ja)} vs {len(Jb)}")
        r = ra - rb
        blocks = [A - B for A, B in zip(Ja, Jb)]
        return r, blocks
    
@dataclass
class StackExpr:
    name: str
    parts: Sequence[Expr]

    @property
    def vars(self):
        return self.parts[0].vars if self.parts else []

    def deps(self):
        out = []
        for p in self.parts:
            out.extend(list(p.deps()))
        return out

    def eval(self, ctx: EvalContext):
        r_list = []
        J_list = None
        for p in self.parts:
            r, blocks = p.eval(ctx)
            r_list.append(np.asarray(r, float).reshape(-1))
            if J_list is None:
                J_list = [[] for _ in blocks]
            for i, B in enumerate(blocks):
                J_list[i].append(np.asarray(B, float))
        r_all = np.concatenate(r_list, axis=0) if r_list else np.zeros((0,), float)
        blocks_all = [np.vstack(chunks) for chunks in (J_list or [])]
        return r_all, blocks_all
    
@dataclass
class HingeExpr:
    """
    Hinge (ReLU) expression.

    Given a base expression h(x), produces:
        r_i = max(0, h_i)

    Jacobian:
        dr_i/dx = dh_i/dx   if h_i > 0
                  0         otherwise

    Typical use:
      - inequality constraint: h(x) <= 0
      - soft barrier / penalty
    """
    name: str
    base: Expr

    @property
    def vars(self):
        return self.base.vars

    @property
    def m(self) -> int:
        return self.base.m

    def deps(self):
        return self.base.deps()

    def eval(self, ctx: EvalContext):
        h, blocks = self.base.eval(ctx)

        h = np.asarray(h, dtype=float).reshape(-1)
        m = h.size

        # Active set mask
        active = (h > 0.0).astype(float)  # (m,)

        # Residual
        r = np.maximum(0.0, h)

        # Jacobian blocks: inactive rows are zeroed
        blocks2 = []
        for B in blocks:
            B = np.asarray(B, dtype=float)
            if B.shape[0] != m:
                raise ValueError(
                    f"{self.name}: block row mismatch: "
                    f"h has {m}, block has {B.shape}"
                )
            blocks2.append(active[:, None] * B)

        return r, blocks2