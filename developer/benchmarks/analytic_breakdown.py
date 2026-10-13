"""Break down current scalar NumPy dense torque Jacobians without source edits.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m developer.benchmarks.analytic_breakdown
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter_ns
from types import CodeType, FunctionType
from unittest.mock import patch

import numpy as np
from robokots.kots import Kots, StateType
from .numpy_rust_jacobian_compare import serial_model

ROOT = Path(__file__).resolve().parents[2]
GRAVITY = [.2, -.3, -9.81]
SEED = 20261001
SCOPE_CATEGORIES = {
    "robokots.core.kernels.whole_body.dynamics_derivatives": "assembly",
    "robokots.core.kernels.whole_body.gravity_derivatives": "gravity",
    "robokots.core.kernels.whole_body.kinematics_derivatives": "construction",
    "robokots.core.kernels.whole_body.partial_dynamics": "construction",
    "robokots.core.kernels.whole_body.dynamics": "construction",
    "robokots.core.kernels.whole_body.kinematics": "construction",
    "robokots.outward.diff.outward_total_gradient": "selection_dispatch",
}
CATEGORIES = ("construction", "matrix_products", "gravity", "assembly", "selection_dispatch")
ACTIVE = None


class Recorder:
    def __init__(self):
        self.categories = defaultdict(int)
        self.functions = defaultdict(lambda: dict(calls=0, inclusive_ns=0))
        self.products = defaultdict(lambda: dict(calls=0, ns=0, dense_flops_estimate=0))
        self.stack = ["selection_dispatch"]
        self.start = self.last = perf_counter_ns()

    def tick(self):
        now = perf_counter_ns()
        self.categories[self.stack[-1]] += now - self.last
        self.last = now
        return now

    @contextmanager
    def scope(self, category, label):
        start = self.tick()
        self.stack.append(category)
        try:
            yield
        finally:
            end = self.tick()
            self.stack.pop()
            self.functions[label]["inclusive_ns"] += end - start
            self.functions[label]["calls"] += 1

    def finish(self):
        end = self.tick()
        assert len(self.stack) == 1
        assert sum(self.categories.values()) == end - self.start
        return dict(total_ms=(end-self.start)/1e6,
                    categories_ms={c:self.categories[c]/1e6 for c in CATEGORIES},
                    functions=dict(self.functions), products=dict(self.products))


@contextmanager
def scope(category, label):
    if ACTIVE is None:
        yield
    else:
        with ACTIVE.scope(category, label):
            yield


def matmul(a, b, label):
    # Object operators (e.g. CMTM composition) remain in their enclosing scope.
    if ACTIVE is None or not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.ndim == b.ndim == 2):
        return a @ b
    recorder = ACTIVE
    with recorder.scope("matrix_products", label + " @"):
        start = perf_counter_ns()
        result = a @ b
        elapsed = perf_counter_ns() - start
    key = f"{label}: {a.shape} @ {b.shape}"
    entry = recorder.products[key]
    entry["calls"] += 1
    entry["ns"] += elapsed
    entry["dense_flops_estimate"] += 2*a.shape[0]*a.shape[1]*b.shape[1]
    return result


class Instrument(ast.NodeTransformer):
    def __init__(self, module, filename):
        self.module, self.filename = module, filename

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.MatMult):
            return ast.copy_location(ast.Call(func=ast.Name(id="_benchmark_matmul", ctx=ast.Load()),
                args=[node.left, node.right, ast.Constant(f"{self.filename}:{node.lineno}")], keywords=[]), node)
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if self.module in SCOPE_CATEGORIES:
            # Preserve docstrings and all return/exception semantics.
            prefix = 1 if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str) else 0
            call = ast.Call(func=ast.Name(id="_benchmark_scope", ctx=ast.Load()),
                            args=[ast.Constant(SCOPE_CATEGORIES[self.module]), ast.Constant(f"{self.module}.{node.name}")], keywords=[])
            wrapped = ast.With(items=[ast.withitem(context_expr=call, optional_vars=None)], body=node.body[prefix:])
            node.body = node.body[:prefix] + [ast.copy_location(wrapped, node)]
        return node


@contextmanager
def instrumented():
    """Replace code objects in this process; aliases retain function identity."""
    changed, globals_added, sources = [], [], {}
    try:
        for name, module in sorted(list(sys.modules.items())):
            filename = getattr(module, "__file__", None)
            if not name.startswith("robokots.") or not filename or not filename.endswith(".py"):
                continue
            path = Path(filename)
            source = path.read_text()
            relative = str(path.resolve().relative_to(ROOT))
            tree = Instrument(name, relative).visit(ast.parse(source, filename=filename))
            compiled = compile(ast.fix_missing_locations(tree), filename, "exec")
            codes = {c.co_name:c for c in compiled.co_consts if isinstance(c, CodeType)}
            originals = [(f,codes[f.__name__]) for f in vars(module).values()
                         if isinstance(f, FunctionType) and f.__module__ == name and f.__name__ in codes
                         and f.__code__.co_freevars == codes[f.__name__].co_freevars]
            if not originals:
                continue
            for key, value in (("_benchmark_matmul",matmul),("_benchmark_scope",scope)):
                assert key not in vars(module)
                setattr(module,key,value)
                globals_added.append((module,key))
            sources[relative] = hashlib.sha256(source.encode()).hexdigest()
            seen = set()
            for f, code in originals:
                if id(f) in seen:
                    continue
                seen.add(id(f))
                changed.append((f, f.__code__))
                f.__code__ = code
        yield sources
    finally:
        for f, code in reversed(changed):
            f.__code__ = code
        for module, key in globals_added:
            delattr(module,key)


def forbidden(*args, **kwargs):
    raise AssertionError("NumPy measurement must not invoke Rust")


def baseline(k, state, motions, repeats, warmup):
    def once(index):
        start = perf_counter_ns()
        k.import_motions(motions[index])
        imported = perf_counter_ns()
        k.dynamics(gravity=GRAVITY, materialize_dict=False)
        prepared = perf_counter_ns()
        value = k.jacobian(state)
        end = perf_counter_ns()
        return value, dict(import_ms=(imported-start)/1e6, state_ms=(prepared-imported)/1e6,
                           jacobian_ms=(end-prepared)/1e6, full_ms=(end-start)/1e6)
    for i in range(warmup):
        once(i%2)
    samples = [once((warmup+i)%2)[1] for i in range(repeats)]
    return dict(median_ms={key:float(np.median([s[key] for s in samples])) for key in samples[0]}, samples=samples)


def profile_once(k, state):
    global ACTIVE
    ACTIVE = recorder = Recorder()
    try:
        value = k.jacobian(state)
        result = recorder.finish()
    finally:
        ACTIVE = None
    return value, result


def evaluate(dof, derivative, args):
    order = derivative+3
    k = Kots.from_json_data(serial_model(dof),order=order,backend="numpy")
    state = StateType("total_joint","total_joint","torque" if derivative == 0 else f"torque_diff{derivative}")
    motions = [np.ascontiguousarray(x[:,:order]).reshape(-1) for x in
               np.random.default_rng(SEED+dof).normal(scale=.2,size=(2,dof,5))]
    with patch.object(k,"_rust_compiled_robot",forbidden):
        before = baseline(k,state,motions,args.repeats,args.warmup)
        references = []
        for motion in motions:
            k.import_motions(motion)
            k.dynamics(gravity=GRAVITY,materialize_dict=False)
            references.append(k.jacobian(state))
        profiled, max_abs, rel = [], 0., 0.
        with instrumented() as sources:
            for i in range(args.profile_repeats + 1):
                index = i%2
                k.import_motions(motions[index])
                k.dynamics(gravity=GRAVITY,materialize_dict=False)
                value, sample = profile_once(k,state)
                difference = value-references[index]
                max_abs = max(max_abs,float(np.max(np.abs(difference))))
                rel = max(rel,float(np.linalg.norm(difference)/max(np.linalg.norm(references[index]),1e-30)))
                np.testing.assert_array_equal(value,references[index])
                if i:
                    profiled.append(sample)
        after = baseline(k,state,motions,args.repeats,args.warmup)
        products, functions = {}, {}
        for key in sorted({key for s in profiled for key in s["products"]}):
            entries = [s["products"].get(key,dict(calls=0,ns=0,dense_flops_estimate=0)) for s in profiled]
            products[key] = dict(median_ms=float(np.median([e["ns"]/1e6 for e in entries])),
                                 calls_per_sample=[e["calls"] for e in entries],
                                 dense_flops_per_sample=[e["dense_flops_estimate"] for e in entries])
        for key in sorted({key for s in profiled for key in s["functions"]}):
            entries = [s["functions"].get(key,dict(calls=0,inclusive_ns=0)) for s in profiled]
            functions[key] = dict(inclusive_median_ms=float(np.median([e["inclusive_ns"]/1e6 for e in entries])),
                                  calls_per_sample=[e["calls"] for e in entries])
        summary = dict(total_ms=float(np.median([s["total_ms"] for s in profiled])),
                       categories_ms={c:float(np.median([s["categories_ms"][c] for s in profiled])) for c in CATEGORIES})
        row = dict(dof=dof,derivative=derivative,motion_order=order,shape=list(references[0].shape),
                   baseline_before=before,baseline_after=after,profile=summary,raw_profile=profiled,
                   products=products,functions=functions,max_abs=max_abs,relative_frobenius=rel,
                   instrumented_source_hashes=sources)
        print(f"d={dof} derivative={derivative}: plain J={before['median_ms']['jacobian_ms']:.3f} / {after['median_ms']['jacobian_ms']:.3f} ms; "
              f"profiled J={summary['total_ms']:.3f}, construction={summary['categories_ms']['construction']:.3f}, "
              f"products={summary['categories_ms']['matrix_products']:.3f} ms",flush=True)
        return row


def report(result,path):
    m = result["metadata"]
    lines = ["# NumPy analytical Jacobian breakdown","",f"Measured UTC: {m['utc']}; runtime source commit `{m['commit']}`.","",
             f"{m['platform']}; {m['cpu']}; Python {m['python']}; NumPy {m['numpy']}; float64.","",
             "All active-joint dense torque Jacobians, derivatives 0 and 2, fixed motion orders 3 and 5. "
             "Same serial-model family, gravity [0.2,-0.3,-9.81], seed 20261001 + DOF as dynamics_complexity. "
             "Rust calls are blocked. Two motion inputs alternate; dictionary materialization is disabled. "
             "Model construction/imports are excluded. BLAS/OpenMP thread counts are one.","",
             f"Uninstrumented phases: {m['repeats']} samples after {m['warmup']} warmups, measured before AND after profiling. "
             f"Profile: one separate warmup, then {m['profile_repeats']} samples. No numerical/AD/JIT timings. "
             "Both inputs are verified bit-for-bit against uninstrumented outputs. JSON retains samples, shapes, "
             "operation counts, conventional dense FLOP estimates, function times and source hashes.","",
             "## Findings",""]
    high=sorted((r for r in result["rows"] if r["derivative"]==2),key=lambda r:r["dof"])
    if high:
        r=high[-1]
        pr=r["profile"]
        lines += [f"At {r['dof']} DOF / derivative 2, construction is {100*pr['categories_ms']['construction']/pr['total_ms']:.1f}% "
                  f"and intercepted products are {100*pr['categories_ms']['matrix_products']/pr['total_ms']:.1f}% of instrumented Jacobian time.",""]
        if len(high)>1:
            a,b=high[-2:]
            fa,fb=(sum(p['dense_flops_per_sample'][0] for p in r['products'].values()) for r in (a,b))
            lines += [f"{a['dof']}→{b['dof']} DOF: intercepted-product FLOPs increase {fb/fa:.3f}×, "
                      f"product time {b['profile']['categories_ms']['matrix_products']/a['profile']['categories_ms']['matrix_products']:.3f}×, "
                      f"construction time {b['profile']['categories_ms']['construction']/a['profile']['categories_ms']['construction']:.3f}×. "
                      "The matrix-product work approaches cubic scaling, while construction remains a substantial elapsed-time term.",""]
    lines += ["Potential optimization targets visible in the product/routine tables: apply diagonal factorial "
              "factors as row/column scaling (partial_dynamics.py lines 23/137), apply spatial-inertia "
              "blocks without whole-body dense block-diagonal multiplication, and reuse repeated "
              "coordinate-to-link gradient construction within one Jacobian call. These are findings, "
              "not runtime changes implemented by this benchmark.","",
              "## Uninstrumented phase times (median ms)","",
             "The table pools the before/after phase samples; full is measured directly. Phase medians "
             "need not sum to the full median. State includes outward computation and API cache bookkeeping. "
             "Jacobian is the FIRST call on each freshly computed state, including lazy relative-CMTM/geometry "
             "cache construction. It differs from repeated state-ready calls that reuse those caches.","",
             "| Derivative | DOF | Import | State | Jacobian | Full | J after/before |","|---:|---:|---:|---:|---:|---:|---:|"]
    for r in result["rows"]:
        samples=r["baseline_before"]["samples"]+r["baseline_after"]["samples"]
        times={key:float(np.median([s[key] for s in samples])) for key in samples[0]}
        lines.append(f"| {r['derivative']} | {r['dof']} | " + " | ".join(f"{times[k]:.4f}" for k in ("import_ms","state_ms","jacobian_ms","full_ms")) +
                     f" | {r['baseline_after']['median_ms']['jacobian_ms']/r['baseline_before']['median_ms']['jacobian_ms']:.3f} |")
    lines += ["","## Instrumented Jacobian times (median ms)","",
              "Construction: whole-body kinematics/dynamics/partial-matrix builders and their callees, "
              "including CMTM/block geometry and allocation; excludes intercepted 2D ndarray products. "
              "Products: explicit ndarray @ ndarray in loaded RoboKots module functions. "
              "Gravity: gravity derivative routines excluding nested construction/products. "
              "Assembly: dense derivative composition excluding its nested categories (e.g. additions/slicing). "
              "Other: output selection, dispatch and profiler bookkeeping. Categories are exclusive and "
              "sum to each raw sample's elapsed time. Individual category medians need not sum to total median.","",
              "External MathRobo internals/class methods are not rewritten; their small internal operations "
              "stay in the enclosing category. Thus construction is a measured routine group, not pure "
              "allocation time. Instrumentation overhead remains in these samples; do not use their "
              "total as production timing or rescale them to the uninstrumented total.","",
              "| Derivative | DOF | Total | Construction | Products | Gravity | Assembly | Other | Product share | Profile/plain J |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in result["rows"]:
        p=r["profile"]
        plain=float(np.median([s["jacobian_ms"] for side in ("baseline_before","baseline_after") for s in r[side]["samples"]]))
        lines.append(f"| {r['derivative']} | {r['dof']} | {p['total_ms']:.4f} | " + " | ".join(f"{p['categories_ms'][c]:.4f}" for c in CATEGORIES) +
                     f" | {100*p['categories_ms']['matrix_products']/p['total_ms']:.1f}% | {p['total_ms']/plain:.3f} |")
    lines += ["","## Most expensive intercepted matrix products","",
              "Top five per case; source line refers to the unchanged runtime source. "
              "FLOPs use 2mkn per m×k times k×n product, regardless of zero entries. "
              "They are arithmetic estimates, not hardware counters. Operator-only timing excludes "
              "the surrounding profiling bookkeeping.","",
              "| Derivative | DOF | Operation / dimensions | ms | Calls per sample | Estimated FLOPs |",
              "|---:|---:|---|---:|---:|---:|"]
    for r in result["rows"]:
        for key,p in sorted(r["products"].items(),key=lambda item:item[1]["median_ms"],reverse=True)[:5]:
            lines.append(f"| {r['derivative']} | {r['dof']} | `{key}` | {p['median_ms']:.4f} | {p['calls_per_sample'][0]} | {p['dense_flops_per_sample'][0]} |")
    lines += ["","## Construction/gravity function hotspots (inclusive)","",
              "Top six construction/gravity routine scopes, excluding outer assembly/selection wrappers. "
              "Parent/child times overlap; do not sum them. Inclusive routine times also contain nested products.","",
              "| Derivative | DOF | Function | Inclusive ms | Calls |","|---:|---:|---|---:|---:|"]
    for r in result["rows"]:
        for key,p in sorted(((k,v) for k,v in r["functions"].items() if SCOPE_CATEGORIES.get(k.rsplit('.',1)[0]) in ("construction","gravity")),key=lambda item:item[1]["inclusive_median_ms"],reverse=True)[:6]:
            lines.append(f"| {r['derivative']} | {r['dof']} | `{key}` | {p['inclusive_median_ms']:.4f} | {p['calls_per_sample'][0]} |")
    lines += ["","## Fixed-order scaling","",
              "Adjacent-size ratios use pooled uninstrumented samples for state/Jacobian. "
              "Products/construction use instrumented samples. Effective α=log(T₂/T₁)/log(d₂/d₁). "
              "FLOP scaling verifies only the intercepted product work, not every operation in the algorithm.","",
              "| Derivative | DOFs | Quantity | Ratio | Effective α |","|---:|---|---|---:|---:|"]
    for n in (0,2):
        rows=sorted((r for r in result["rows"] if r["derivative"]==n),key=lambda r:r["dof"])
        def quantities(r):
            return dict(state=float(np.median([s["state_ms"] for side in ("baseline_before","baseline_after") for s in r[side]["samples"]])),
                        jacobian=float(np.median([s["jacobian_ms"] for side in ("baseline_before","baseline_after") for s in r[side]["samples"]])),
                        construction=r["profile"]["categories_ms"]["construction"], products=r["profile"]["categories_ms"]["matrix_products"],
                        product_flops=sum(p["dense_flops_per_sample"][0] for p in r["products"].values()))
        for a,b in zip(rows,rows[1:]):
            qa,qb=quantities(a),quantities(b)
            for key in qa:
                ratio=qb[key]/qa[key]
                lines.append(f"| {n} | {a['dof']}→{b['dof']} | {key} | {ratio:.3f} | {math.log(ratio)/math.log(b['dof']/a['dof']):.3f} |")
    path.write_text("\n".join(lines)+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dofs",nargs="+",type=int,default=[4,8,16,32,64])
    p.add_argument("--repeats",type=int,default=5)
    p.add_argument("--warmup",type=int,default=2)
    p.add_argument("--profile-repeats",type=int,default=3)
    p.add_argument("--output",type=Path,default=ROOT/"developer/benchmarks/results/analytic_breakdown.json")
    args=p.parse_args()
    if min(args.dofs)<=0 or min(args.repeats,args.profile_repeats)<=0 or args.warmup<0:
        p.error("DOFs/repeats must be positive and warmup nonnegative")
    cpu=next((s.split(":",1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines() if s.startswith("model name")),platform.processor())
    m=dict(utc=datetime.now(timezone.utc).isoformat(),commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
           platform=platform.platform(),cpu=cpu,python=platform.python_version(),numpy=np.__version__,
           gravity=GRAVITY,seed=SEED,dtype="float64",script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           threads={key:os.getenv(key) for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")},
           **{key:value for key,value in vars(args).items() if key!="output"})
    result=dict(metadata=m,rows=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for n in (0,2):
        for d in args.dofs:
            result["rows"].append(evaluate(d,n,args))
            args.output.write_text(json.dumps(result,indent=2)+"\n")
            report(result,args.output.with_suffix(".md"))


if __name__ == "__main__":
    main()
