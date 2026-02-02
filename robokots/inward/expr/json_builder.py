from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable
import numpy as np

from robokots.inward.term import Variable, VariablePack, Problem, L2Cost, DiagonalWeightCost, ScalarWeightCost, HuberCost, EvalContext
from robokots.inward.expr.registry import Registry
from robokots.inward.expr.nodes import ConstantExpr, GetStateExpr, SubExpr, StackExpr, HingeExpr
from robokots.core.state_cache import OwnerKey, StateKey, StateCache
from robokots.core.time_grid import TimeGrid


@dataclass
class BuilderContext:
    pack: VariablePack
    state_cache: StateCache
    time: TimeGrid
    registry: Registry
    model: Any = None  # optional


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


def build_problem(spec: dict, *, state_cache: StateCache, time: TimeGrid, registry: Registry, model: Any = None) -> Problem:
    # Variables
    vars = []
    for v in spec.get("variables", []):
        name = v["name"]
        dim = int(v["dim"])
        vars.append(Variable(name=name, x=np.zeros((dim,), float)))
    pack = VariablePack(vars)

    ctx = BuilderContext(pack=pack, state_cache=state_cache, time=time, registry=registry, model=model)

    terms = []
    for t in spec.get("terms", []):
        expr = build_expr(ctx, t["expr"])
        cost_spec = t.get("cost", {"type": "l2"})
        cost = build_cost(registry, cost_spec)
        terms.append((expr, cost))

    return Problem(variables=pack, terms=terms)


def build_cost(registry: Registry, spec: dict):
    typ = spec.get("type", "l2")
    fn = registry.cost.get(typ, None)
    if fn is not None:
        return fn(spec)
    return default_cost_builder(spec)


def build_expr(ctx: BuilderContext, spec: dict):
    typ = spec["type"]
    fn = ctx.registry.expr.get(typ, None)
    if fn is None:
        raise ValueError(f"unknown expr type: {typ}")
    return fn(ctx, spec)


# ---- required key collection (used by solver wrapper) ----
def collect_required(problem: Problem):
    req = []
    for expr, _cost in problem.terms:
        if hasattr(expr, "deps"):
            req.extend(list(expr.deps()))
    return list(dict.fromkeys(req))


def make_eval_context(problem: Problem, state_cache: StateCache, time: TimeGrid) -> EvalContext:
    # ctx.revision: you can combine pack.revision and time.revision if needed
    return EvalContext(pack=problem.variables, state=state_cache, time=time, revision=time.revision)
