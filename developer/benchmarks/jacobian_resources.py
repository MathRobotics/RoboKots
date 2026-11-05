"""CPU time and process peak-RSS comparison in isolated sequential workers.

python -u -m developer.benchmarks.jacobian_resources
Each (quantity, derivative order, method) runs in a fresh Python process.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from robokots.kots import Kots
from robokots.outward.diff.dynamics_jax import dynamics_jax, dynamics_state_vector_jax
from .jacobian_accuracy_table import ROWS, VELOCITY_TYPES, make_model, selections
from .dynamics_autodiff_compare import error


DEFAULT_METHODS = ("numpy_full", "numpy_cached", "rust_full", "rust_cached", "numerical",
           "forward_eager", "reverse_eager", "forward_jit", "reverse_jit")
METHODS = DEFAULT_METHODS + tuple(
    f"{prefix}{direction}_{execution}"
    for prefix in ("time_", "id_time_", "cmtm_")
    for direction in ("forward", "reverse") for execution in ("eager", "jit")
)
GRAVITY = [0.3, -0.4, -9.81]
ROOT = Path(__file__).resolve().parents[2]


def peak_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 ** 2 if sys.platform == "darwin" else 1024)


def worker(args):
    jax.config.update("jax_enable_x64", True)
    devices = str(jax.devices())  # Common CPU runtime initialization before baseline.
    model = make_model(args.dof)
    kots = Kots.from_json_data(model, order=args.k + 3)
    state = selections(kots, args.k)[args.row]
    order = state.time_order
    # Same inputs as the accuracy table, including identical lower-order prefixes.
    full_motion = np.random.default_rng(args.seed).normal(scale=0.4, size=(args.dof, 7))
    motion = full_motion[:, :order].reshape(-1)
    kots = Kots.from_json_data(model, order=order)
    kots.import_motions(motion)
    kots.gravity_ = np.asarray(GRAVITY)
    backend = "rust" if args.method.startswith("rust") else "numpy"

    def update(x):
        kots.import_motions(x)
        if state.is_dynamics:
            kots.dynamics(backend=backend, gravity=GRAVITY, materialize_dict=False)
        else:
            kots.kinematics(backend=backend, materialize_dict=False)

    if args.method.endswith("cached"):
        update(motion)
        def run(x):
            return np.asarray(kots.jacobian(state))
    elif args.method.endswith("full"):
        def run(x):
            update(x)
            return np.asarray(kots.jacobian(state))
    elif args.method == "numerical":
        def run(x):
            kots.import_motions(x)
            return np.asarray(kots.jacobian(state, numerical=True))
    else:
        def value(x):
            if state.data_type in VELOCITY_TYPES:
                return dynamics_jax(kots.robot_, x, order, GRAVITY)["link_velocity"][state.owner_name][args.k]
            return dynamics_state_vector_jax(kots.robot_, x, [state], order, GRAVITY)
        if args.method.startswith("cmtm_"):
            from .cmtm_autodiff import make_cmtm_ad_value
            value = make_cmtm_ad_value(kots.robot_, state, order, GRAVITY)
        elif args.method.startswith("id_time_"):
            from .id_time_autodiff import make_id_time_ad_value
            value = make_id_time_ad_value(kots.robot_, state, order, GRAVITY)
        elif args.method.startswith("time_"):
            from .time_autodiff import make_time_ad_value
            value = make_time_ad_value(kots.robot_, state, order, GRAVITY)
        direction = args.method.removeprefix("cmtm_").removeprefix("id_time_").removeprefix("time_")
        derivative = (jax.jacfwd if direction.startswith("forward") else jax.jacrev)(value)
        if args.method.endswith("jit"):
            derivative = jax.jit(derivative)
        def run(x):
            return np.asarray(jax.block_until_ready(derivative(jnp.asarray(x))))

    baseline = peak_mib()
    start = time.perf_counter_ns()
    first = run(motion)
    first_ms = (time.perf_counter_ns() - start) / 1e6
    peak_first = peak_mib()
    for _ in range(args.warmup):
        run(motion)
    elapsed = []
    repeats = args.numerical_repeats if args.method == "numerical" else args.repeats
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = run(motion)
        elapsed.append((time.perf_counter_ns() - start) / 1e6)
    peak = peak_mib()  # Capture before validation to avoid contaminating memory results.
    np.testing.assert_allclose(result, first, atol=1e-10, rtol=1e-10)
    # Validate another runtime input, including compiled functions, outside measurements.
    other = np.random.default_rng(args.seed + 1).normal(scale=0.4, size=(args.dof, 7))[:, :order].reshape(-1)
    if args.method.endswith("cached"):
        update(other)
    other_result = run(other)
    if not np.all(np.isfinite(result)) or not np.all(np.isfinite(other_result)):
        raise AssertionError("Nonfinite Jacobian")
    output_values = {}
    if args.check_values:
        # Outside timing/memory measurements. Validate the differentiated
        # quantities themselves too, not only their input Jacobians.
        if args.method.removeprefix("cmtm_").removeprefix("id_time_").removeprefix("time_").startswith(("forward", "reverse")):
            output = jax.jit(value)
            read = lambda x: np.asarray(jax.block_until_ready(output(jnp.asarray(x))))
        else:
            def read(x):
                update(x)
                return np.asarray(kots.state_info(state))
        output_values = {"value": read(motion).tolist(), "other_value": read(other).tolist()}
    return {
        "row": args.row, "quantity": ROWS[args.row][0], "k": args.k, "method": args.method,
        "owner": state.owner_name, "motion_order": order, "shape": list(result.shape),
        "first_ms": first_ms, "median_ms": float(np.median(elapsed)), "samples_ms": elapsed,
        "repeats": repeats, "warmup": args.warmup,
        "baseline_peak_mib": baseline, "first_peak_mib": peak_first, "peak_mib": peak,
        "peak_growth_mib": max(0.0, peak - baseline),
        "post_first_peak_growth_mib": max(0.0, peak - peak_first),
        "jacobian": result.tolist(), "other_jacobian": other_result.tolist(),
        **output_values,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "jax": jax.__version__, "devices": devices},
    }


def render(metadata, results, failures):
    lines = ["# Jacobian CPU time and memory", "",
             f"Measured: {metadata['utc']}; {metadata['platform']}; CPU: {metadata['cpu']}.", "",
             f"{metadata['dof']} DOF, float64, seed {metadata['seed']}, gravity {GRAVITY}. "
             "Model, inputs, environment, revision and raw timings are in JSON. "
             "A single random motion is measured repeatedly, not independent trials.", "",
             "Every method/case uses a fresh sequential subprocess with JAX_PLATFORMS=cpu and "
             "OMP/OPENBLAS/MKL_NUM_THREADS=1. These settings do not guarantee single-threaded XLA. "
             "Process startup/import/model construction and common CPU runtime initialization are excluded. "
             "Cached analytic cases additionally exclude initial state preparation; full cases include motion "
             "import and state calculation without dictionary export. Numerical uses central differences, eps=1e-8. "
             "AD includes runtime input conversion, synchronized execution, and NumPy output conversion. "
             "Unprefixed forward/reverse AD computes analytic time derivatives directly on ordinary derivative coefficient series; "
             "eager may compute unused dynamics work. "
             "Methods prefixed time_ instead obtain velocity from FK directional JVPs, momentum rate from "
             "another JVP, and higher output derivatives from nested total-time JVPs before outer jacfwd/jacrev. "
             "Methods prefixed id_time_ start from ordinary ID(q,qdot,qddot), always motion order 3, "
             "and use total-time JVPs only for the higher torque derivatives. "
             "Methods prefixed cmtm_ instead construct explicit spatial lower block-Toeplitz CMTMs "
             "and use factorial-normalized coefficient vectors; only the outer motion Jacobian uses AD. "
             "The coefficient-series and explicit-CMTM paths are implementation representations of the same high-order algebra, "
             "not separate mathematical differentiation methods. Both use recurrences to construct transform coefficients. "
             "No trajectory is constructed: D_t f(x)=JVP(f,x,shift(x)) with x=(q,qdot,...); "
             "repeated AD differentiates the state-dependent direction too.", "",
             "First call includes lazy initialization (and tracing/compilation for JIT). "
             "Warm medians exclude the first call and configured warmups. No JIT function closes over motion. "
             "A second seeded input is checked against NumPy analytic results outside measurement.", "",
             "Peak RSS is the OS process-lifetime high-water mark, not live array size or device memory. "
             "It includes Python, libraries, compiler and allocator caches. Growth subtracts the baseline "
             "high-water mark after setup; it is NOT exact per-call allocation or a separately measured warm-only peak. "
             "Each memory measurement is one process run. Validation occurs after the memory snapshot.", "",
             "Rust labels specify the requested state backend, not a guarantee of Rust derivative kernels. "
             "The selections retain explicit local frames from the accuracy table; current Rust fast paths "
             "require frame_name=None. Nonzero gravity also restricts spatial derivative kernels. "
             "These Rust-labelled measurements therefore include Python derivative fallback, including torque.", "",
             f"Completed: {len(results)} / {metadata['expected']}; failures: {len(failures)}.", "",
             "## Forward vs reverse AD", "",
             "Speed ratio = forward median / reverse median; above 1 means reverse was faster. "
             "Peak RSS includes first call/compilation, even for JIT warm timings.", "",
             "| Quantity | k | Mode | Forward ms | Reverse ms | Speed ratio | Forward peak MiB | Reverse peak MiB |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    lookup = {(r['row'], r['k'], r['method']): r for r in results}
    for row, k in sorted({(r['row'], r['k']) for r in results}):
        for prefix, mode in ((p, m) for p in ("", "time_", "id_time_", "cmtm_") for m in ("eager", "jit")):
            forward = lookup.get((row, k, f"{prefix}forward_{mode}"))
            reverse = lookup.get((row, k, f"{prefix}reverse_{mode}"))
            if forward is not None and reverse is not None:
                ratio = forward['median_ms'] / reverse['median_ms']
                lines.append(f"| {ROWS[row][0]} | {k} | {prefix}{mode} | {forward['median_ms']:.3f} | "
                             f"{reverse['median_ms']:.3f} | {ratio:.2f} | "
                             f"{forward['peak_mib']:.1f} | {reverse['peak_mib']:.1f} |")
    lines += ["", "## All methods", "",
              "| Quantity | k | Method | First ms | Warm median ms | n | Baseline MiB | Peak MiB | Growth MiB |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['quantity']} | {r['k']} | {r['method']} | {r['first_ms']:.3f} | "
                     f"{r['median_ms']:.3f} | {r['repeats']} | {r['baseline_peak_mib']:.1f} | "
                     f"{r['peak_mib']:.1f} | {r['peak_growth_mib']:.1f} |")
    lines += ["", "## Agreement against NumPy analytic", "",
              "Errors are maximum absolute and relative Frobenius differences over the two tested inputs. "
              "NumPy analytic is a comparison reference, not an exact oracle.", "",
              "| Quantity | k | Method | Max abs | Relative Frobenius |",
              "|---|---:|---|---:|---:|"]
    for r in results:
        if "errors" in r:
            lines.append(f"| {r['quantity']} | {r['k']} | {r['method']} | "
                         f"{max(e['max_abs'] for e in r['errors']):.3e} | "
                         f"{max(e['relative_frobenius'] for e in r['errors']):.3e} |")
    if failures:
        lines += ["", "## Failures", "", "```json", json.dumps(failures, indent=2), "```"]
    if any("value_errors" in r for r in results):
        lines += ["", "## Output-value agreement", "",
                  "The time-differentiated quantities themselves are also checked at both inputs, "
                  "outside all timing/memory measurements. These are not Jacobian errors.", "",
                  "| Quantity | k | Method | Max abs | Relative Frobenius |",
                  "|---|---:|---|---:|---:|"]
        for r in results:
            if "value_errors" in r:
                lines.append(f"| {r['quantity']} | {r['k']} | {r['method']} | "
                             f"{max(e['max_abs'] for e in r['value_errors']):.3e} | "
                             f"{max(e['relative_frobenius'] for e in r['value_errors']):.3e} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dof", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--derivatives", type=int, nargs="+", choices=range(5), default=[0, 4])
    parser.add_argument("--rows", type=int, nargs="+", choices=range(8), default=[0, 1, 5, 7])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--numerical-repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--check-values", action="store_true", help="Also validate output values outside measurements")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results") / "jacobian_resources")
    parser.add_argument("--render-only", action="store_true", help="Regenerate Markdown from saved JSON without measuring")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--row", type=int, choices=range(8), default=0, help=argparse.SUPPRESS)
    parser.add_argument("--k", type=int, choices=range(5), default=0, help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=METHODS, default="numpy_full", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.render_only:
        saved = json.loads(args.output.with_suffix(".json").read_text(encoding="utf-8"))
        args.output.with_suffix(".md").write_text(
            render(saved["metadata"], saved["results"], saved["failures"]), encoding="utf-8")
        return
    if min(args.dof, args.repeats, args.numerical_repeats) < 1 or args.warmup < 0 or args.seed < 0 or not 0 < args.timeout < float("inf"):
        parser.error("invalid count, seed, warmup or timeout")
    if args.worker:
        print(json.dumps(worker(args), allow_nan=False))
        return
    if "numpy_full" not in args.methods:
        parser.error("include numpy_full for independent agreement checks")
    metadata = {
        "utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(), "dof": args.dof, "seed": args.seed,
        "derivatives": args.derivatives, "rows": args.rows, "methods": args.methods,
        "repeats": args.repeats, "numerical_repeats": args.numerical_repeats, "warmup": args.warmup,
        "check_values": args.check_values, "worker_timeout_seconds": args.timeout,
        "expected": len(args.rows) * len(args.derivatives) * len(args.methods),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "model": make_model(args.dof), "gravity": GRAVITY,
        "motions": [np.random.default_rng(s).normal(scale=0.4, size=(args.dof, 7)).tolist()
                    for s in (args.seed, args.seed + 1)],
    }
    env = dict(os.environ, JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    results, failures = [], []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for k in args.derivatives:
        for row in args.rows:
            reference = None
            methods = ["numpy_full"] + [m for m in args.methods if m != "numpy_full"]
            for method in methods:
                print(f"k={k}, {ROWS[row][0]}, {method}", flush=True)
                command = [sys.executable, "-m", "developer.benchmarks.jacobian_resources", "--worker",
                           "--dof", str(args.dof), "--seed", str(args.seed), "--row", str(row), "--k", str(k),
                           "--method", method, "--repeats", str(args.repeats),
                           "--numerical-repeats", str(args.numerical_repeats), "--warmup", str(args.warmup)]
                if args.check_values:
                    command.append("--check-values")
                try:
                    process = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True,
                                             check=True, timeout=args.timeout)
                    result = json.loads(process.stdout)
                    if method == "numpy_full":
                        reference = result
                    if reference is None:
                        raise AssertionError("Missing NumPy reference")
                    result["errors"] = [error(np.asarray(result[key]), np.asarray(reference[key]))
                                        for key in ("jacobian", "other_jacobian")]
                    if args.check_values:
                        result["value_errors"] = [error(np.asarray(result[key]), np.asarray(reference[key]))
                                                 for key in ("value", "other_value")]
                    results.append(result)
                except (subprocess.SubprocessError, ValueError, AssertionError) as exc:
                    failures.append({"row": row, "k": k, "method": method, "error": str(exc),
                                     "stderr": str(getattr(exc, "stderr", ""))[-4000:]})
                    print(f"FAILED: {failures[-1]}", flush=True)
                args.output.with_suffix(".json").write_text(json.dumps({
                    "metadata": metadata, "results": results, "failures": failures,
                }, indent=2, allow_nan=False), encoding="utf-8")
                args.output.with_suffix(".md").write_text(render(metadata, results, failures), encoding="utf-8")
    print(f"Report: {args.output.with_suffix('.md')}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
