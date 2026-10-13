"""Measure DOF scaling of current dense all-joint dynamics Jacobians.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m developer.benchmarks.dynamics_complexity
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from robokots import outward
from robokots.kots import Kots, StateType
from robokots.outward.diff.dynamics_jax import dynamics_state_vector_jax
from .numpy_rust_jacobian_compare import serial_model

ROOT = Path(__file__).resolve().parents[2]
GRAVITY = [.2, -.3, -9.81]
SEED = 20261001


def measure(fn, repeats, warmup):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = np.asarray(fn())
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return result, dict(median_ms=float(np.median(samples)), samples_ms=samples,
                        repeats=repeats, warmup=warmup)


def forbidden(*args, **kwargs):
    raise AssertionError("Rust analytic benchmark fell back to NumPy derivatives")


def evaluate(dof, derivative, args):
    order = derivative + 3
    state = StateType("total_joint", "total_joint", "torque" if derivative == 0 else f"torque_diff{derivative}")
    motions = [np.ascontiguousarray(x[:, :order]).reshape(-1)
               for x in np.random.default_rng(SEED + dof).normal(scale=.2, size=(2, dof, 5))]
    timings, errors = {}, {}
    reference = None

    def record(name, fn, repeats, warmup):
        nonlocal reference
        value, timings[name] = measure(fn, repeats, warmup)
        if reference is None:
            reference = value
        difference = value - reference
        errors[name] = dict(max_abs=float(np.max(np.abs(difference))),
                           relative_frobenius=float(np.linalg.norm(difference) / max(np.linalg.norm(reference), 1e-30)))
        atol, rtol = (2e-5, 2e-5) if "numerical" in name else (1e-8, 1e-9)
        np.testing.assert_allclose(value, reference, atol=atol, rtol=rtol)
        print(f"d={dof} derivative={derivative} {name}: {timings[name]['median_ms']:.4f} ms", flush=True)

    for backend in ("numpy", "rust"):
        k = Kots.from_json_data(serial_model(dof), order=order, backend=backend)

        def prepare(x):
            k.import_motions(x)
            k.dynamics(gravity=GRAVITY, materialize_dict=False)

        prepare(motions[0])
        # Alternate inputs inside full to invalidate the primal cache.
        index = [0]

        def full():
            index[0] = 1 - index[0]
            prepare(motions[index[0]])
            return k.jacobian(state)

        context = patch.object(outward, "outward_jacobian", forbidden) if backend == "rust" else patch.object(k, "_rust_compiled_robot", forbidden)
        with context:
            record(backend + "_ready", lambda: k.jacobian(state), args.repeats, 2)
            # full ends at motion[0] with an even number of calls.
            _, timings[backend + "_full"] = measure(full, args.repeats + args.repeats % 2, 2)
            print(f"d={dof} derivative={derivative} {backend}_full: {timings[backend + '_full']['median_ms']:.4f} ms", flush=True)
            np.testing.assert_allclose(k.jacobian(state), reference, atol=1e-8, rtol=1e-9)
        if dof <= args.numerical_max_dof:
            prepare(motions[0])
            record(backend + "_numerical", lambda: k.jacobian(state, numerical=True), args.numerical_repeats, 0)

    if dof <= args.ad_max_dof:
        k = Kots.from_json_data(serial_model(dof), order=order)
        states = k._state_type_list(state)

        def value(x):
            return dynamics_state_vector_jax(k.robot_, x, states, order, GRAVITY)

        for mode, transform in (("forward", jax.jacfwd), ("reverse", jax.jacrev)):
            eager = transform(value)
            x = jnp.asarray(motions[0])
            record(mode + "_eager", lambda: np.asarray(jax.block_until_ready(eager(x))), args.eager_repeats, 1)
            compiled = jax.jit(eager)

            def run():
                return np.asarray(jax.block_until_ready(compiled(jnp.asarray(motions[0]))))

            record(mode + "_jit_first", run, 1, 0)
            record(mode + "_jit", run, args.jit_repeats, 2)
            # Confirm motion remains a runtime input, including a second value.
            k.import_motions(motions[1])
            k.dynamics(gravity=GRAVITY, materialize_dict=False)
            np.testing.assert_allclose(np.asarray(compiled(jnp.asarray(motions[1]))), k.jacobian(state), atol=1e-8, rtol=1e-9)
        jax.clear_caches()
    return dict(dof=dof, derivative=derivative, motion_order=order, shape=list(reference.shape), timings=timings, errors=errors)


def report(result, path):
    m = result["metadata"]
    lines = ["# Dense dynamics complexity audit", "", f"Measured UTC: {m['utc']}; source commit `{m['commit']}`.", "",
             f"{m['platform']}; {m['cpu']}; Python {m['python']}; NumPy {m['numpy']}; JAX {m['jax']}; {m['devices']}.", "",
             "Current implementations, float64, serial revolute chains with alternating axes and a fixed tool. "
             "All active-joint torques, dense shape d × (p d). Gravity [0.2, -0.3, -9.81], "
             "seed 20261001 + DOF; p=3 for torque, p=5 for its second time derivative. "
             "Orders are held fixed during each DOF sweep. No batch/JVP/VJP. "
             "Rust analytic fallback is blocked; NumPy cannot invoke the Rust model.", "",
             "Ready excludes state calculation; full includes motion import and state calculation and alternates "
             "two inputs to invalidate caches. Numerical uses central differences eps=1e-8 and includes all perturbed "
             "state evaluations; Rust scales each step by max(1, |x_i|), NumPy uses a fixed step. "
             "NumPy and Rust numerical are explicitly separated. AD eager uses pure jacfwd/jacrev "
             "without jit, independent of the public API's current jit=True default. JIT input is a runtime argument; "
             "warm/first include input conversion, synchronized execution and host output conversion. "
             "Construction/import costs are excluded; JIT first includes tracing and compilation. "
             "Compiled outputs are also checked at a second input.", "",
             f"Analytic: {m['repeats']} samples ready / next even count full, two warmups. "
             f"Numerical: {m['numerical_repeats']} samples, no warmup (single measurements are noisy). "
             f"Eager: {m['eager_repeats']} samples, one warmup. Warm JIT: {m['jit_repeats']} samples, two warmups. "
             "No competing benchmark process. Raw samples, errors, extension hash, environment and limits are in JSON.", "",
             "## Findings", "",
             "Compare each measured ratio with its fixed-order work model: doubling d predicts "
             "8× for O(d³) and 4× for O(d²) arithmetic. Agreement in some methods and differences "
             "in others must be assessed separately; the work model is not an exact elapsed-time law.", ""]
    pair = {r["dof"]:r for r in result["rows"] if r["derivative"] == 2 and r["dof"] in (8,16)}
    if len(pair) == 2:
        lines += ["Second time derivative, 8→16 DOF (same output and order):", "",
                  "| Method | Work model | 8 DOF ms | 16 DOF ms | Observed ratio |", "|---|---|---:|---:|---:|"]
        for name, cost in (("numpy_full","O(d³)"),("rust_full","O(d²)"),("numpy_numerical","O(d³)"),
                           ("rust_numerical","O(d²)"),("forward_jit","O(d²)"),("reverse_jit","O(d²)")):
            if name in pair[8]["timings"] and name in pair[16]["timings"]:
                a, b = (pair[d]["timings"][name]["median_ms"] for d in (8,16))
                lines.append(f"| {name} | {cost} | {a:.5f} | {b:.5f} | {b/a:.3f}× |")
        lines += [""]
    lines += ["## Median milliseconds", "",
             "| Time derivative | DOF | Shape | Method | ms |", "|---:|---:|---|---|---:|"]
    for r in result["rows"]:
        for name, t in r["timings"].items():
            lines.append(f"| {r['derivative']} | {r['dof']} | {'×'.join(map(str,r['shape']))} | {name} | {t['median_ms']:.5f} |")
    lines += ["", "## Adjacent-size scaling", "",
              "Effective exponent α = log(T₂/T₁)/log(d₂/d₁); doubling predicts 4× for quadratic work, "
              "8× for cubic. These finite-size timing exponents do not prove asymptotic complexity. "
              "Compilation time is excluded from this table.", "",
              "| Derivative | DOFs | Method | Time ratio | Effective α |", "|---:|---|---|---:|---:|"]
    for derivative in (0, 2):
        rows = sorted((r for r in result["rows"] if r["derivative"] == derivative), key=lambda r:r["dof"])
        for a, b in zip(rows, rows[1:]):
            for name in sorted(a["timings"].keys() & b["timings"].keys()):
                if name.endswith("first"):
                    continue
                ratio = b["timings"][name]["median_ms"] / a["timings"][name]["median_ms"]
                lines.append(f"| {derivative} | {a['dof']}→{b['dof']} | {name} | {ratio:.3f} | {math.log(ratio)/math.log(b['dof']/a['dof']):.3f} |")
    maxima = {kind:max((e["max_abs"] for r in result["rows"] for name,e in r["errors"].items()
                       if ("numerical" in name) == (kind == "numerical")), default=0.)
              for kind in ("numerical", "analytic/AD")}
    relative = {kind:max((e["relative_frobenius"] for r in result["rows"] for name,e in r["errors"].items()
                         if ("numerical" in name) == (kind == "numerical")), default=0.) for kind in maxima}
    lines += ["", "## Agreement", "",
              "All measured methods are checked against NumPy analytic output, a comparison reference "
              "rather than an exact oracle. First, warm and eager AD outputs are included.", "",
              "| Methods | Maximum absolute difference | Maximum relative Frobenius error |",
              "|---|---:|---:|"]
    for kind in maxima:
        lines.append(f"| {kind} | {maxima[kind]:.6g} | {relative[kind]:.6g} |")
    lines += ["", "## Interpretation and implementation cost", "",
              "For fixed p and L=O(d) links, the scalar NumPy analytic path builds whole-body dense "
              "matrices and multiplies them (conventional dense multiplication O(d³)); Rust analytical "
              "propagates dense derivatives along the tree (O(d²)); both AD directions require O(d²) "
              "arithmetic for this all-joint dense output (forward has pd tangent directions, reverse d "
              "output directions; a primal evaluation is O(d)). These are arithmetic-work models, "
              "not exact wall-clock time laws.", "",
              "NumPy numerical runs a separate finite-difference pass for every selected joint. "
              "The present central-difference helper also evaluates the unperturbed state once: "
              "d(2pd+1) full state evaluations, each O(d), hence O(d³). Rust numerical shares "
              "2pd full state evaluations across all selected outputs, hence O(d²). "
              "Output allocation alone requires Ω(pd²) space/work for a dense d×pd matrix; "
              "this lower bound alone does not classify the implementation's full computation.", "",
              "The observed finite-size exponents should be read separately for each method and scope. "
              "Python dispatch, small-array work, allocation, caches, vectorized tangent operations, "
              "compiler optimization and CPU utilization contribute different terms to elapsed time. "
              "NumPy analytic and AD can therefore appear below the asymptotic arithmetic exponent "
              "at these sizes. A non-cubic/non-quadratic timing exponent is not evidence that the "
              "underlying dense algorithm has a different asymptotic complexity. JIT compilation "
              "is a separate cost and its scaling is not inferred from warm execution.", "",
              "Historical dynamics_autodiff reports used earlier dispatch, platforms and AD defaults. "
              "They cannot be combined with this run as a single scaling curve. Historical "
              "jacobian_resources selects individual outputs, whereas this run selects all joints. "
              "The saved rust_dense_dynamics run varies time derivative order as well as DOF; "
              "only equal-order rows can be used to estimate DOF scaling. "
              "This audit does not establish high-order p asymptotics or large-d asymptotics for AD.", ""]
    path.write_text("\n".join(lines).rstrip() + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dofs", nargs="+", type=int, default=[4, 8, 16, 32])
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--numerical-repeats", type=int, default=1)
    p.add_argument("--eager-repeats", type=int, default=3)
    p.add_argument("--jit-repeats", type=int, default=30)
    p.add_argument("--numerical-max-dof", type=int, default=16)
    p.add_argument("--ad-max-dof", type=int, default=16)
    p.add_argument("--output", type=Path, default=ROOT / "developer/benchmarks/results/dynamics_complexity.json")
    args = p.parse_args()
    if any(d <= 0 for d in args.dofs) or min(args.repeats, args.numerical_repeats, args.eager_repeats, args.jit_repeats) <= 0:
        p.error("DOFs and repeat counts must be positive")
    jax.config.update("jax_enable_x64", True)
    import robokots._rust_core as native
    cpu = next((line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), platform.processor())
    m = dict(utc=datetime.now(timezone.utc).isoformat(), commit=subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip(),
             platform=platform.platform(), cpu=cpu, python=platform.python_version(), numpy=np.__version__, jax=jax.__version__,
             devices=str(jax.devices()), dtype="float64", seed=SEED, gravity=GRAVITY,
             extension_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
             threads={key:os.getenv(key) for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")},
             **{key:value for key,value in vars(args).items() if key != "output"})
    result = dict(metadata=m, rows=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for derivative in (0, 2):
        for dof in args.dofs:
            result["rows"].append(evaluate(dof, derivative, args))
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            report(result, args.output.with_suffix(".md"))


if __name__ == "__main__":
    main()
