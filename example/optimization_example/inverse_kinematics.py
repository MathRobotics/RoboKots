"""
Inverse kinematics least-squares optimization demo (JSON task spec) -- Expr版

- User edits task.json (targets, link names, costs).
- Program auto-builds Expr/Costs/Problem and solves via Gauss-Newton.

Design:
- Heavy computations are centralized in StateCache.build_state(x_all, time, required).
- Expr nodes are thin readers of StateCache.
- StateCache.update_if_needed(...) is called ONLY in solver (once per evaluation point).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Iterable, Sequence, Tuple, Protocol

import numpy as np

from robokots.core.state import StateType
from robokots.core.state_cache import StateCache, StateKey, OwnerKey  # あなたの最新版に合わせる
from robokots.inward.opt import solve_gauss_newton
from robokots.inward.term import (
    Variable,
    VariablePack,
    L2Cost,
    Cost,
)
from robokots.inward.problem import Problem  # Expr版Problemを想定
from robokots.kots import Kots

Array = np.ndarray


# ============================================================
# EvalContext: Exprが必要なものだけを見る窓
# ============================================================
@dataclass
class EvalContext:
    pack: VariablePack
    cache: StateCache
    time: Any = None


# ============================================================
# Expr protocol
# ============================================================
class Expr(Protocol):
    name: str
    vars: Sequence[Variable]
    m: int

    def eval(self, ctx: EvalContext) -> Tuple[Array, Sequence[Array]]:
        ...


# ============================================================
# Registry (Expr / Cost)
# ============================================================
class Registry:
    def __init__(self) -> None:
        self.expr_builders: Dict[str, Callable[[EvalContext, dict], Expr]] = {}
        self.cost_builders: Dict[str, Callable[[dict], Cost]] = {}

    def register_expr(self, typ: str, fn: Callable[[EvalContext, dict], Expr]) -> None:
        self.expr_builders[typ] = fn

    def register_cost(self, typ: str, fn: Callable[[dict], Cost]) -> None:
        self.cost_builders[typ] = fn


# ============================================================
# Expr implementations (cache readers)
# ============================================================
@dataclass
class LinkPositionErrorExpr:
    """
    r = p_link(x) - target
    blocks = dp/dq  (single var 'q' assumed in this demo)

    StateCache must provide:
      key_pos: StateKey(... dtype="frame", field="pos")
      key_J:   StateKey(... dtype="frame", field="pos_J_q")  # 例
    """
    name: str
    link: str
    target: Array
    vars: Sequence[Variable]
    _key_pos: StateKey
    _key_J: StateKey

    @property
    def m(self) -> int:
        return 3

    def eval(self, ctx: EvalContext) -> Tuple[Array, Sequence[Array]]:
        # NOTE: cache.update_if_needed はここでは呼ばない（solver側で一括更新）
        pos = np.asarray(ctx.cache.get(self._key_pos), dtype=float).reshape(3)
        J = np.asarray(ctx.cache.get(self._key_J), dtype=float)

        if len(self.vars) != 1:
            raise ValueError(f"{self.name}: expected one variable (q).")
        qdim = self.vars[0].dim()
        if J.shape != (3, qdim):
            raise ValueError(f"{self.name}: Jacobian shape mismatch: got {J.shape}, expected {(3, qdim)}")

        r = pos - np.asarray(self.target, dtype=float).reshape(3)
        return r, [J]


# ============================================================
# build_state: Kots -> dict[StateKey, Any]
# ============================================================
def make_build_state(kots: Kots, *, link_names: List[str]) -> Callable[..., dict]:
    """
    build_state(x_all, time=None, required=None) -> dict[StateKey, Any]

    まずは required を無視して、必要リンクの pos と J を全部作る簡易版。
    （後で required 対応に最適化できる）
    """
    def build_state(
        x_all: Array,
        *,
        time: Any = None,
        required: Optional[Iterable[StateKey]] = None,
    ) -> dict:
        q = np.asarray(x_all, dtype=float).reshape(-1)

        kots.import_motions(q)
        kots.kinematics()

        st_out: dict[StateKey, Any] = {}

        for link in link_names:
            st = StateType("link", link, "pos")
            pos = np.asarray(kots.state_info(st), dtype=float).reshape(3)
            J = np.asarray(kots.jacobian(st), dtype=float)

            if J.shape[0] != 3:
                raise ValueError(f"Kots jacobian rows unexpected for link={link}: {J.shape}")

            owner = OwnerKey(owner_type="link", owner_name=link)
            key_pos = StateKey(k=0, owner=owner, dtype="frame", field="pos")
            key_J   = StateKey(k=0, owner=owner, dtype="frame", field="pos_J_q")

            st_out[key_pos] = pos
            st_out[key_J] = J

        return st_out

    return build_state


# ============================================================
# Spec -> Problem builder
# ============================================================
def build_problem_from_spec(spec: dict) -> Tuple[Problem, EvalContext]:
    # 1) model
    model = spec["model"]
    if model.get("type") != "kots_json":
        raise ValueError("Only model.type='kots_json' is supported in this demo.")
    path = model["path"]
    order = int(model.get("order", 1))
    kots = Kots.from_json_file(path, order=order)

    # 2) variables
    v0 = spec["variables"]
    if len(v0) != 1 or v0[0]["name"] != "q":
        raise ValueError("This demo expects exactly one variable: name='q'.")
    q_init = np.asarray(v0[0]["init"], dtype=float).reshape(-1)
    if q_init.size != kots.dof():
        raise ValueError(f"q init dim mismatch: got {q_init.size}, expected {kots.dof()}")

    q_var = Variable(name="q", x=q_init.copy())
    pack = VariablePack([q_var])

    # 3) registry
    reg = Registry()
    reg.register_cost("l2", lambda p: L2Cost())

    # 必要なら outward.term から import して登録（あなたの配置に合わせて）
    from robokots.inward.term import DiagonalWeightCost, HuberCost  # もし inward にあるなら
    reg.register_cost("diag_weight", lambda p: DiagonalWeightCost(w=np.asarray(p["w"], float)))
    reg.register_cost("huber", lambda p: HuberCost(delta=float(p["delta"])))

    # 4) Expr builder を登録
    def build_link_pos_error(ctx: EvalContext, p: dict) -> Expr:
        link = p["link"]
        target = np.asarray(p["target"], dtype=float).reshape(3)

        owner = OwnerKey(owner_type="link", owner_name=link)
        key_pos = StateKey(k=0, owner=owner, dtype="frame", field="pos")
        key_J   = StateKey(k=0, owner=owner, dtype="frame", field="pos_J_q")

        return LinkPositionErrorExpr(
            name=p.get("name", f"{link}_pos_error"),
            link=link,
            target=target,
            vars=ctx.pack.vars,
            _key_pos=key_pos,
            _key_J=key_J,
        )

    reg.register_expr("link_position_error", build_link_pos_error)

    # 5) collect links used in spec (for build_state)
    link_names: List[str] = []
    for t in spec.get("terms", []):
        es = t["expr"]
        if es["type"] == "link_position_error":
            link_names.append(es["params"]["link"])
    link_names = sorted(set(link_names))

    # 6) cache
    build_state = make_build_state(kots, link_names=link_names)
    cache = StateCache(build_state=build_state)

    ctx = EvalContext(pack=pack, cache=cache, time=None)

    # 7) terms (Expr, Cost)
    terms: List[Tuple[Expr, Cost]] = []
    for t in spec.get("terms", []):
        es = t["expr"]
        cs = t["cost"]

        etype = es["type"]
        eparams = es.get("params", {})
        if etype not in reg.expr_builders:
            raise ValueError(f"Unknown expr type: {etype}")
        expr = reg.expr_builders[etype](ctx, eparams)

        ctype = cs["type"]
        cparams = cs.get("params", {})
        if ctype not in reg.cost_builders:
            raise ValueError(f"Unknown cost type: {ctype}")
        cost = reg.cost_builders[ctype](cparams)

        terms.append((expr, cost))

    problem = Problem(variables=pack, terms=terms)
    return problem, ctx


# ============================================================
# main
# ============================================================
def main() -> None:
    with open("task.json", "r", encoding="utf-8") as f:
        spec = json.load(f)

    problem, ctx = build_problem_from_spec(spec)

    print("Initial q:", ctx.pack.get())

    ctx.cache.update_if_needed(ctx.pack, time=ctx.time, required=None)
    print("Initial cost:", problem.cost_value(ctx=ctx, time=ctx.time, required=None))

    solve_gauss_newton(problem, ctx.pack, max_iters=20, ctx=ctx)

    ctx.cache.update_if_needed(ctx.pack, time=ctx.time, required=None)
    print("Final q:", ctx.pack.get())
    print("Final cost:", problem.cost_value(ctx=ctx, time=ctx.time, required=None))


if __name__ == "__main__":
    main()
