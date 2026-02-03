from __future__ import annotations
from typing import Any, Iterable, Optional, List, Callable
import numpy as np

from robokots.inward.term import (
    Variable,
    VariablePack,
    Problem,
    L2Cost,
    DiagonalWeightCost,
    ScalarWeightCost,
    HuberCost,
    EvalContext,
)
from robokots.inward.context import BuilderContext
from robokots.inward.expr.registry import Registry
from robokots.inward.expr.nodes import (
    ConstantExpr,
    GetStateExpr,
    SubExpr,
    StackExpr,
    HingeExpr,
)
from robokots.core.state_cache import OwnerKey, StateKey, StateCache
from robokots.core.time_grid import TimeGrid

Array = np.ndarray


def make_build_state_kots(kots: Kots) -> Callable[..., dict]:
    """
    build_state(x_all, time=None, required=None) -> dict[StateKey, Any]

    required に含まれる StateKey だけ計算する（無ければ空dict/または必要最小を計算）。
    まずは dtype="frame", field={"pos", "pos_J_q"} のみ対応（IKの最小セット）。
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

        out: dict[StateKey, Any] = {}

        if required is None:
            # required が無い場合は「何も返さない」だと GetStateExpr が困るので、
            # 最小限の互換のために、ここは空にせず「呼ばれる側が required を必ず渡す」運用を推奨。
            # ただし安全策として、何もできないので out を返す。
            return out

        # required から link ごとに必要項目を整理
        need_by_link: dict[str, set[str]] = {}
        for key in required:
            if key.owner.owner_type != "link":
                continue
            if key.dtype != "frame":
                continue
            link = key.owner.owner_name
            need_by_link.setdefault(link, set()).add(key.field)

        for link, fields in need_by_link.items():
            # pos が欲しい or pos_J_q が欲しいなら StateType("link", link, "pos") でまとめて計算
            if ("pos" in fields) or ("pos_J_q" in fields):
                st = StateType("link", link, "pos")
                pos = np.asarray(kots.state_info(st), dtype=float).reshape(-1)
                J = np.asarray(kots.jacobian(st), dtype=float)

                owner = OwnerKey(owner_type="link", owner_name=link)

                # k は IK では 0 固定（軌道になったら k を回す）
                key_pos = StateKey(k=0, owner=owner, dtype="frame", field="pos")
                key_J   = StateKey(k=0, owner=owner, dtype="frame", field="pos_J_q")

                if "pos" in fields:
                    out[key_pos] = pos
                if "pos_J_q" in fields:
                    out[key_J] = J

        return out

    return build_state


# ============================================================
# Cost builders
# ============================================================
def default_cost_builder(spec: dict):
    typ = spec.get("type", "l2")
    if typ == "l2":
        return L2Cost()
    if typ == "diag":
        return DiagonalWeightCost(np.asarray(spec["w"], float))
    if typ == "scalar":
        return ScalarWeightCost(float(spec["w"]))
    if typ == "huber":
        return HuberCost(float(spec["delta"]))
    raise ValueError(f"unknown cost type: {typ}")


def build_cost(registry: Registry, spec: dict):
    typ = spec.get("type", "l2")
    fn = registry.cost.get(typ, None)
    if fn is not None:
        return fn(spec)
    return default_cost_builder(spec)


# ============================================================
# Variable builders
# ============================================================
def build_variable(v: dict) -> Variable:
    """
    Supports:
      - {"name":"q","dim":7}                -> zeros
      - {"name":"q","init":[...]}           -> dim inferred
      - {"name":"q","dim":7,"init":[...]}   -> checked
    """
    name = v["name"]
    if "init" in v:
        x = np.asarray(v["init"], dtype=float).reshape(-1)
        dim = int(v.get("dim", x.size))
        if x.size != dim:
            raise ValueError(f"variable '{name}': init size {x.size} != dim {dim}")
        return Variable(name=name, x=x.copy())
    else:
        dim = int(v["dim"])
        return Variable(name=name, x=np.zeros((dim,), dtype=float))


# ============================================================
# StateKey parsing helpers (JSON -> OwnerKey/StateKey)
# ============================================================
def parse_owner_key(spec: dict) -> OwnerKey:
    """
    Accept:
      {"owner_type":"link","owner_name":"end"}
    """
    return OwnerKey(
        owner_type=str(spec["owner_type"]),
        owner_name=str(spec["owner_name"]),
    )


def parse_state_key(spec: dict) -> StateKey:
    """
    Accept either:
      {
        "k": 0,
        "owner": {"owner_type":"link","owner_name":"end"},
        "dtype": "frame",
        "field": "pos",
        "frame": null,
        "rel_frame": null
      }

    or flattened:
      {
        "k":0,
        "owner_type":"link",
        "owner_name":"end",
        "dtype":"frame",
        "field":"pos"
      }
    """
    k = int(spec.get("k", 0))

    if "owner" in spec:
        owner = parse_owner_key(spec["owner"])
    else:
        owner = OwnerKey(
            owner_type=str(spec["owner_type"]),
            owner_name=str(spec["owner_name"]),
        )

    return StateKey(
        k=k,
        owner=owner,
        dtype=str(spec["dtype"]),
        field=str(spec["field"]),
        frame=spec.get("frame", None),
        rel_frame=spec.get("rel_frame", None),
    )


# ============================================================
# Expr builders
# ============================================================
def build_expr(ctx: BuilderContext, spec: dict):
    # Normalize legacy spec keys for compatibility
    if "x" in spec and "base" not in spec and spec.get("type") == "hinge":
        spec = dict(spec)
        spec["base"] = spec.pop("x")
    if "items" in spec and "parts" not in spec and spec.get("type") == "stack":
        spec = dict(spec)
        spec["parts"] = spec.pop("items")

    typ = spec["type"]
    fn = ctx.registry.expr.get(typ, None)
    if fn is None:
        raise ValueError(f"unknown expr type: {typ}")
    return fn(ctx, spec)


def register_default_expr_builders(registry: Registry) -> None:
    """
    Optional: If you haven't registered builders elsewhere,
    call this once to get a minimal working set.

    This registers JSON specs for common nodes:
      - constant
      - get_state
      - sub
      - stack
      - hinge  (max(0, x) row-wise)
    """
    # constant: {"type":"constant","value":[...],"vars":["q"]?}
    def b_constant(ctx: BuilderContext, spec: dict):
        val = np.asarray(spec["value"], dtype=float).reshape(-1)
        # constant doesn't depend on vars, but Expr interface might still want vars list.
        # Here we set vars=[] by default.
        return ConstantExpr(name=spec.get("name", "const"), value=val)

    # get_state: {"type":"get_state","key":{...}, "shape":[m]?}
    def b_get_state(ctx: BuilderContext, spec: dict):
        key_value = parse_state_key(spec["key"])

        jac_spec = spec.get("jac", None)
        if jac_spec is None:
            raise ValueError("get_state requires 'jac' spec with field/var.")

        var_name = jac_spec.get("var", "q")
        q = next(v for v in ctx.pack.vars if v.name == var_name)

        jac_field = jac_spec["field"]
        key_jac_q = StateKey(
            k=key_value.k,
            owner=key_value.owner,
            dtype=key_value.dtype,
            field=jac_field,
            frame=key_value.frame,
            rel_frame=key_value.rel_frame,
        )

        return GetStateExpr(
            name=spec.get("name", f"get_{key_value.field}"),
            vars=[q],
            key_value=key_value,
            key_jac_q=key_jac_q,
        )

    # sub: {"type":"sub","a":{expr}, "b":{expr}}
    def b_sub(ctx: BuilderContext, spec: dict):
        a = build_expr(ctx, spec["a"])
        b = build_expr(ctx, spec["b"])
        return SubExpr(name=spec.get("name", "sub"), a=a, b=b)

    # stack: {"type":"stack","parts":[{expr},{expr},...]}  (legacy: items)
    def b_stack(ctx: BuilderContext, spec: dict):
        parts = [build_expr(ctx, s) for s in spec["parts"]]
        return StackExpr(name=spec.get("name", "stack"), parts=parts)

    # hinge: {"type":"hinge","base":{expr}}  (inequality residual; legacy: x)
    def b_hinge(ctx: BuilderContext, spec: dict):
        base = build_expr(ctx, spec["base"])
        return HingeExpr(name=spec.get("name", "hinge"), base=base)

    registry.expr["constant"] = b_constant
    registry.expr["get_state"] = b_get_state
    registry.expr["sub"] = b_sub
    registry.expr["stack"] = b_stack
    registry.expr["hinge"] = b_hinge


# ============================================================
# Problem builder
# ============================================================
def build_problem(
    spec: dict,
    *,
    state_cache: StateCache,
    time: TimeGrid,
    registry: Registry,
    model: Any = None
) -> Problem:
    # Variables
    vars: List[Variable] = []
    for v in spec.get("variables", []):
        vars.append(build_variable(v))
    pack = VariablePack(vars)

    ctx = BuilderContext(pack=pack, state_cache=state_cache, time=time, registry=registry, model=model)

    terms = []
    for t in spec.get("terms", []):
        expr_spec = t["expr"]
        expr = build_expr(ctx, expr_spec)

        cost_spec = t.get("cost", {"type": "l2"})
        cost = build_cost(registry, cost_spec)

        terms.append((expr, cost))

    return Problem(variables=pack, terms=terms)

def build_problem_from_spec(
    spec: dict,
    *,
    backend,  
    registry: Registry,
    model: Any = None,
) -> tuple[Problem, EvalContext]:
    """
    High-level entry point.

    - Creates Registry
    - Creates TimeGrid
    - Creates StateCache
    - Builds Problem
    - Returns (problem, eval_context)

    This is what examples should call.
    """

    # --- registry ---
    registry = Registry()
    # ここで outward 側の register を呼ぶ想定
    # e.g. register_default_exprs(registry)
    # e.g. register_default_costs(registry)

    # --- time grid ---
    time_spec = spec.get("time", None)
    if time_spec is None:
        time = TimeGrid.single_time()
    else:
        time = TimeGrid.from_spec(time_spec)

    state_cache = StateCache(build_state=backend.build_state)

    # --- problem ---
    problem = build_problem(
        spec,
        state_cache=state_cache,
        time=time,
        registry=registry,
        model=model,
    )

    # --- eval context ---
    ctx = make_eval_context(problem, state_cache, time)

    return problem, ctx


# ============================================================
# Required key collection
# ============================================================
def collect_required(problem: Problem):
    """
    Collect StateKey dependencies from all expr nodes.
    Convention: node.deps() -> Iterable[StateKey]
    """
    req: List[StateKey] = []
    for expr, _cost in problem.terms:
        if hasattr(expr, "deps"):
            req.extend(list(expr.deps()))
    # unique while preserving order
    out = []
    seen = set()
    for k in req:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


# ============================================================
# EvalContext + cache update helpers (solver-facing)
# ============================================================
def make_eval_context(problem: Problem, state_cache: StateCache, time: TimeGrid) -> EvalContext:
    """
    Use a single revision integer so Problem's caching and StateCache can coordinate.
    Here we use time.revision as a base; solver should still bump pack.revision when x changes.
    """
    return EvalContext(pack=problem.variables, state=state_cache, time=time, revision=int(time.revision))


def prepare_problem_for_solve(problem: Problem, ctx: EvalContext, *, required: Optional[Iterable[StateKey]] = None) -> List[StateKey]:
    """
    1) Determine required keys (if not provided)
    2) Update cache once
    Returns the required list used.
    """
    if required is None:
        required_list = collect_required(problem)
    else:
        required_list = list(required)

    # IMPORTANT: cache update should happen outside Expr nodes
    ctx.state.update_if_needed(ctx.pack, time=ctx.time, required=required_list)
    return required_list


def linearize_with_context(problem: Problem, ctx: EvalContext, *, required: Optional[Iterable[StateKey]] = None):
    """
    Safe wrapper used by solvers:
      - update cache
      - call problem.linearize(ctx=..., time=..., required=...)
    """
    req = prepare_problem_for_solve(problem, ctx, required=required)
    # Problem.linearize should accept these kwargs in your Expr版実装
    return problem.linearize(ctx=ctx, time=ctx.time, required=req)
