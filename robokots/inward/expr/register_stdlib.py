# inward/expr/stdlib.py
from __future__ import annotations
import numpy as np
from .registry import Registry
from .nodes import ConstantExpr, GetStateExpr, SubExpr, StackExpr, HingeExpr
from ...core.state_cache import OwnerKey, StateKey

def register_stdlib(reg: Registry) -> None:
    reg.register_expr("const", build_const)
    reg.register_expr("get_state", build_get_state)
    reg.register_expr("sub", build_sub)
    reg.register_expr("stack", build_stack)
    reg.register_expr("hinge", build_hinge)

def build_const(ctx, spec):
    q = next(v for v in ctx.pack.vars if v.name == spec.get("var", "q"))
    return ConstantExpr(name=spec.get("name","const"), vars=[q], value=np.asarray(spec["value"], float))

def build_get_state(ctx, spec):
    q = next(v for v in ctx.pack.vars if v.name == spec.get("jac", {}).get("var", "q"))

    k = int(spec["key"].get("k", 0))
    owner = OwnerKey(spec["key"]["owner_type"], spec["key"]["owner_name"])
    dtype = spec["key"]["dtype"]
    field = spec["key"]["field"]

    key_value = StateKey(k=k, owner=owner, dtype=dtype, field=field)

    # Jacobian field naming: e.g. "pos_J_q"
    jac_field = spec["jac"]["field"]
    key_jac = StateKey(k=k, owner=owner, dtype=dtype, field=jac_field)

    return GetStateExpr(name=spec.get("name","get_state"), vars=[q], key_value=key_value, key_jac_q=key_jac)

def build_sub(ctx, spec):
    a = ctx.registry.expr[spec["a"]["type"]](ctx, spec["a"])
    b = ctx.registry.expr[spec["b"]["type"]](ctx, spec["b"])
    return SubExpr(name=spec.get("name","sub"), a=a, b=b)

def _inject_k(spec, k: int):
    if isinstance(spec, dict):
        out = {}
        for key, val in spec.items():
            if key == "key" and isinstance(val, dict):
                key_dict = dict(val)
                key_dict["k"] = k
                out[key] = key_dict
            else:
                out[key] = _inject_k(val, k)
        return out
    if isinstance(spec, list):
        return [_inject_k(v, k) for v in spec]
    return spec


def _stack_ks(ctx, spec):
    if "range" in spec:
        r = spec["range"]
        k0, k1 = int(r["k0"]), int(r["k1"])
        return list(range(k0, k1 + 1))

    time = getattr(ctx, "time", None)
    if time is None or float(getattr(time, "dt", 0.0)) == 0.0:
        return [0]

    return list(time.ks())


def build_stack(ctx, spec):
    if "parts" in spec:
        part_specs = spec["parts"]
    elif "inner" in spec:
        ks = _stack_ks(ctx, spec)
        part_specs = [_inject_k(spec["inner"], k) for k in ks]
    else:
        raise ValueError("stack spec requires 'parts' or 'inner'.")

    parts = [ctx.registry.expr[p["type"]](ctx, p) for p in part_specs]
    return StackExpr(name=spec.get("name","stack"), parts=parts)

def build_hinge(ctx, spec):
    base_spec = spec.get("base", spec.get("x"))
    if base_spec is None:
        raise ValueError("hinge spec requires 'base' (or legacy 'x').")
    base = ctx.registry.expr[base_spec["type"]](ctx, base_spec)
    return HingeExpr(name=spec.get("name","hinge"), base=base)
