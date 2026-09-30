"""Compare dense Rust dynamics Jacobians against an isolated source revision.

python -u -m developer.benchmarks.rust_dense_dynamics_compare --baseline-ref 932a426
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260930
GRAVITY = [.2, -.3, -9.81]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(fn, args):
    def timed(count):
        start = perf_counter_ns()
        for _ in range(count):
            fn()
        return (perf_counter_ns() - start) / count / 1e6
    first = timed(1)
    for _ in range(args.warmup):
        fn()
    samples = [timed(args.calls) for _ in range(args.samples)]
    return {"first_ms": first, "samples_ms": samples}


def worker(args):
    # The Python dispatch and extension must both belong to the chosen revision.
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(args.source))
    spec = importlib.util.spec_from_file_location("robokots._rust_core", args.extension)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    import numpy as np
    import robokots
    from robokots import outward
    from robokots.kots import Kots, StateType
    from developer.benchmarks.numpy_rust_jacobian_compare import serial_model
    assert Path(robokots.__file__).resolve().is_relative_to(args.source.resolve())

    def forbidden(*args, **kwargs):
        raise AssertionError("Rust dense benchmark must not fall back to Python derivatives")
    outward.outward_jacobian = forbidden

    cases = [(dof, family, n, "local", ()) for dof in args.dofs
             for family in ("momentum", "force", "torque") for n in range(5)]
    # Additional moving-frame and multidimensional-batch cases.
    cases += [(args.dofs[0], family, 4, "world", ()) for family in ("momentum", "force")]
    cases += [(args.dofs[0], "torque", 4, "local", (2, 1))]
    rows = []
    for dof, family, derivative, frame, shape in cases:
        order = derivative + (2 if family == "momentum" else 3)
        data = serial_model(dof)
        k = Kots.from_json_data(data, order=order, backend="rust")
        key = family if derivative == 0 else f"{family}_diff{derivative}"
        state = StateType("total_joint", "total_joint", key,
                          None if family == "torque" else frame)
        # Common q, qdot, ... across derivative orders, without strided inputs.
        rng = np.random.default_rng(SEED + dof)
        motions = [np.ascontiguousarray(rng.normal(scale=.2, size=shape + (dof, 7))[..., :order])
                   .reshape(shape + (dof * order,)) for _ in range(2)]

        def prepare(motion):
            k.import_motions(motion)
            k.dynamics(gravity=GRAVITY, materialize_dict=False)

        prepare(motions[0])
        ready = measure(lambda: k.jacobian(state), args)
        index = 0

        def full():
            nonlocal index
            index = 1 - index
            prepare(motions[index])
            return k.jacobian(state)

        including_state = measure(full, args)
        values = []
        for motion in motions:
            prepare(motion)
            value = np.asarray(k.jacobian(state))
            assert np.isfinite(value).all()
            values.append(value.tolist())
        rows.append(dict(dof=dof, family=family, derivative=derivative, frame=frame,
                         batch_shape=list(shape), motion_order=order, evaluation_order=order,
                         jacobian_shape=list(value.shape), values=values,
                         timings={"state_ready": ready, "including_state": including_state}))
        print(f"  {args.label}: d={dof} {key} {frame} batch={shape}", flush=True)
    args.output.write_text(json.dumps(dict(numpy=np.__version__, rows=rows)))


def report(path, result):
    m = result["metadata"]
    lines = ["# Rust dense dynamics Jacobian comparison", "",
             f"Measured UTC: {m['measured_at']}", "",
             f"Baseline: `{m['baseline_commit']}` (Python dispatch + release Rust extension). "
             "Optimized: current working-tree Python dispatch + installed release extension.", "",
             "Synthetic serial revolute chains with alternating axes, rotated origins and a fixed tool. "
             "All active-joint outputs; local unless specified. Momentum/force each have 6 rows per joint; "
             "torque has 1. Dense Jacobians only, float64, gravity [0.2, -0.3, -9.81], seed 20260930 + DOF. "
             "Motion order is derivative+2 for momentum, derivative+3 for force/torque; no padding.", "",
             f"{m['rounds']} fresh-process rounds, alternating variant order. Per case and scope: "
             f"one separate first call, {m['warmup']} warmups, {m['samples']} samples × {m['calls']} calls. "
             "Reported medians pool all rounds. Milliseconds per call, including the entire batch. "
             "Input/output boundary costs included; model construction and imports excluded. "
             "No JAX/JIT or numerical differentiation is timed. Python derivative fallback is blocked.", "",
             "state_ready starts with computed states and reuses warmed derivative workspaces. "
             "including_state alternates two motions and includes import, dynamics without dictionary "
             "materialization, and the Jacobian. First calls are API calls, not process startup/compilation. "
             "A compiled extension is reused throughout each process.", "",
             f"Environment: {m['cpu']}; {m['platform']}; Python {m['python']}; NumPy {m['numpy']}; {m['rustc']}. "
             "BLAS/OpenMP thread counts are set to one. Environment and raw samples are in the JSON.", "",
             "| DOF | Output | Derivative | Frame | Batch | Shape | Ready old ms | Ready new ms | Speedup | Full old ms | Full new ms | Speedup |",
             "|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in result["rows"]:
        t = r["timings"]
        cells = []
        for scope in ("state_ready", "including_state"):
            a, b = (t[v][scope]["median_ms"] for v in ("baseline", "optimized"))
            cells.extend([f"{a:.4f}", f"{b:.4f}", f"{a/b:.2f}×"])
        lines.append(f"| {r['dof']} | {r['family']} | {r['derivative']} | {r['frame']} | "
                     f"{r['batch_shape']} | {'×'.join(map(str,r['jacobian_shape']))} | " + " | ".join(cells) + " |")
    lines += ["", f"Maximum absolute difference across both motions/all rounds: {result['max_abs']:.6g}.",
              f"Maximum relative Frobenius difference: {result['relative_frobenius']:.6g}.", "",
              "Baseline outputs are a comparison reference, not an exact oracle. "
              "Independent NumPy/finite-difference regressions are separate from this timing run.", "",
              "Reproduce: `" + m["command"] + "`", ""]
    path.write_text("\n".join(lines))


def run(args):
    import numpy as np
    import robokots._rust_core as extension
    import shlex
    commit = subprocess.check_output(["git", "rev-parse", args.baseline_ref], cwd=ROOT, text=True).strip()
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", PYO3_PYTHON=sys.executable)
    results = {"baseline": [], "optimized": []}
    with tempfile.TemporaryDirectory(prefix="robokots-dense-compare-") as directory:
        scratch = Path(directory)
        baseline = scratch / "baseline"
        baseline.mkdir()
        archive = subprocess.check_output(["git", "archive", commit, "robokots"], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(baseline, filter="data")
        target = scratch / "target"
        print(f"Building isolated baseline {commit[:7]} ...", flush=True)
        build = subprocess.run(["cargo", "build", "--offline", "--release", "--lib", "--manifest-path",
                                str(baseline / "robokots/_rust/Cargo.toml")],
                               env=dict(env, CARGO_TARGET_DIR=str(target)), capture_output=True, text=True)
        if build.returncode:
            raise RuntimeError(build.stderr)
        old_extension = target / "release/librobokots_rust.so"
        new_extension = scratch / "optimized.so"
        shutil.copy2(extension.__file__, new_extension)
        hashes = {"baseline": digest(old_extension), "optimized": digest(new_extension)}
        for round_id in range(args.rounds):
            variants = ("baseline", "optimized") if round_id % 2 == 0 else ("optimized", "baseline")
            for variant in variants:
                output = scratch / f"{variant}-{round_id}.json"
                cmd = [sys.executable, str(Path(__file__).resolve()), "--source",
                       str(baseline if variant == "baseline" else ROOT), "--extension",
                       str(old_extension if variant == "baseline" else new_extension),
                       "--label", variant, "--output", str(output), "--warmup", str(args.warmup),
                       "--samples", str(args.samples), "--calls", str(args.calls),
                       "--dofs", *map(str, args.dofs)]
                print(f"Round {round_id+1}/{args.rounds}: {variant}", flush=True)
                subprocess.run(cmd, cwd=scratch, env=env, check=True)
                results[variant].append(json.loads(output.read_text()))
    rows = []
    max_abs = relative = 0.0
    for i, base in enumerate(results["baseline"][0]["rows"]):
        row = {k: v for k, v in base.items() if k not in ("values", "timings")}
        row["timings"], row["errors"] = {}, []
        for variant, rounds in results.items():
            row["timings"][variant] = {}
            for r in rounds:
                current = r["rows"][i]
                assert all(current[k] == v for k, v in row.items() if k not in ("timings", "errors"))
                for actual, expected in zip(current["values"], base["values"]):
                    a, b = np.asarray(actual), np.asarray(expected)
                    np.testing.assert_allclose(a, b, atol=5e-8, rtol=5e-9)
                    absolute = float(np.max(np.abs(a-b)))
                    rel = float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-30))
                    row["errors"].append(dict(variant=variant, max_abs=absolute, relative_frobenius=rel))
                    max_abs, relative = max(max_abs, absolute), max(relative, rel)
            for scope in ("state_ready", "including_state"):
                samples = [t for r in rounds for t in r["rows"][i]["timings"][scope]["samples_ms"]]
                row["timings"][variant][scope] = dict(
                    median_ms=statistics.median(samples), samples_ms=samples,
                    first_ms=[r["rows"][i]["timings"][scope]["first_ms"] for r in rounds],
                    round_medians_ms=[statistics.median(r["rows"][i]["timings"][scope]["samples_ms"]) for r in rounds])
        rows.append(row)
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    metadata = dict(measured_at=datetime.now(timezone.utc).isoformat(), baseline_commit=commit,
                    platform=platform.platform(), cpu=cpu, python=platform.python_version(), numpy=np.__version__,
                    rustc=subprocess.check_output(["rustc", "--version"], text=True).strip(),
                    rounds=args.rounds, warmup=args.warmup, samples=args.samples, calls=args.calls,
                    seed=SEED, gravity=GRAVITY, dofs=args.dofs, extension_sha256=hashes,
                    source_sha256={str(p.relative_to(ROOT)): digest(p) for folder in ("robokots/_rust/src", "robokots/api")
                                   for p in (ROOT/folder).iterdir() if p.suffix in (".py", ".rs")},
                    command=shlex.join([sys.executable, "-u", "-m", "developer.benchmarks.rust_dense_dynamics_compare", *sys.argv[1:]]))
    result = dict(metadata=metadata, rows=rows, max_abs=max_abs, relative_frobenius=relative)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    report(args.output.with_suffix(".md"), result)
    print(f"Report: {args.output.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="932a426")
    parser.add_argument("--dofs", nargs="+", type=int, default=[7, 16])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--label", default="optimized")
    parser.add_argument("--output", type=Path, default=ROOT / "developer/benchmarks/results/rust_dense_dynamics.json")
    args = parser.parse_args()
    if min(args.dofs + [args.rounds, args.samples, args.calls]) < 1 or args.warmup < 0:
        parser.error("DOFs/rounds/samples/calls must be positive, warmup nonnegative")
    if args.source and args.extension:
        worker(args)
    else:
        run(args)
