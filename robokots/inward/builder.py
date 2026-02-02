from robokots.inward.context import BuilderContext
from robokots.inward.context import Registry
from robokots.inward.problem import Problem

def build_problem(spec: dict, ctx: BuilderContext, reg: Registry) -> Problem:
    # quantities
    qmap = {}
    for qspec in spec.get("quantities", []):
        qmap[qspec["id"]] = reg.quantity[qspec["type"]](ctx, qspec.get("params", {}))

    # terms
    terms = []
    for t in spec.get("terms", []):
        rspec = t["residual"]
        cspec = t["cost"]
        residual = reg.residual[rspec["type"]](ctx, rspec, qmap)
        cost = reg.cost[cspec["type"]](cspec.get("params", {}))
        terms.append((residual, cost))

    return Problem(variables=ctx.pack, terms=terms)