"""Compare a committed baseline with the current state/dictionary separation.

Run from the repository root with the existing virtualenv Python. Workers run
sequentially, sharing the same compiled Rust extension and dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
SEED = 71
GRAVITY = [0.2, -0.3, -9.81]


def worker(root, repeats, warmup):
    sys.path.insert(0, str(root))
    import numpy as np
    from robokots.kots import Kots, StateType
    from developer.benchmarks.common import build_model
    assert Path(sys.modules["robokots.kots"].__file__).resolve() == (root / "robokots/kots.py").resolve()

    rows = []
    for dof in (3, 16):
        model = (json.loads((ROOT / "tests/test_model/sample_robot.json").read_text())
                 if dof == 3 else build_model(dof, "humanoid"))
        for batch in (1, 8):
            rng = np.random.default_rng(SEED)
            motion = rng.normal(scale=0.2, size=((dof * 5,) if batch == 1 else (batch, dof * 5)))
            for backend in ("numpy", "rust"):
                for operation in ("kinematics_default", "dynamics_default", "dynamics_no_dict",
                                  "dynamics_export", "jacobian_cached"):
                    kots = Kots.from_json_data(model, order=5)
                    torque = StateType("total_joint", "total_joint", "torque_diff2")
                    tip = StateType("link", kots.link_name_list()[-1], "pos")
                    kots.import_motions(motion)
                    kots.dynamics(backend=backend, gravity=GRAVITY, materialize_dict=False)

                    def run():
                        if operation == "jacobian_cached":
                            return kots.jacobian(torque)
                        # Revision changes force recomputation even with identical input.
                        kots.import_motions(motion)
                        if operation == "kinematics_default":
                            kots.kinematics(backend=backend)
                            return kots.state_info(tip)
                        if operation == "dynamics_default":
                            kots.dynamics(backend=backend, gravity=GRAVITY)
                        else:
                            kots.dynamics(backend=backend, gravity=GRAVITY, materialize_dict=False)
                        if operation == "dynamics_export":
                            return kots.to_state_dict()
                        return kots.state_info(torque)

                    start = time.perf_counter_ns()
                    run()
                    first_ms = (time.perf_counter_ns() - start) / 1e6
                    for _ in range(warmup):
                        run()
                    samples = []
                    for _ in range(repeats):
                        start = time.perf_counter_ns()
                        value = run()
                        samples.append((time.perf_counter_ns() - start) / 1e6)
                    if isinstance(value, list) and value and isinstance(value[0], dict):
                        value = {key: np.stack([sample[key] for sample in value]) for key in value[0]}
                    if isinstance(value, dict):
                        value = np.concatenate([np.asarray(value[key]).reshape(-1) for key in sorted(value)])
                    value = np.asarray(value, dtype=float)
                    rows.append(dict(dof=dof, batch=batch, backend=backend, operation=operation,
                                     first_ms=first_ms, samples_ms=samples, shape=value.shape,
                                     value=value.reshape(-1).tolist()))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="6ec3bfb")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--worker-root", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "developer/benchmarks/results/state_dictionary")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup nonnegative")
    if args.worker_root:
        print(json.dumps(worker(args.worker_root, args.repeats, args.warmup)))
        return

    import numpy as np
    import robokots._rust_core as rust
    from importlib.metadata import version

    baseline_commit = subprocess.check_output(["git", "rev-parse", args.baseline], cwd=ROOT, text=True).strip()
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
    measurements = {"before": [], "after": []}
    # ABBA ordering reduces simple warmup/drift bias; no concurrent timed workers.
    with tempfile.TemporaryDirectory(prefix="robokots-state-baseline-") as temp:
        baseline_root = Path(temp)
        archive = subprocess.check_output(["git", "archive", baseline_commit], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(baseline_root, filter="data")
        extension = Path(rust.__file__).resolve()
        (baseline_root / "robokots" / extension.name).symlink_to(extension)
        for label in ("before", "after", "after", "before"):
            root = baseline_root if label == "before" else ROOT
            print(f"Measuring {label}: {args.repeats} repeats per case", flush=True)
            result = subprocess.check_output(
                [sys.executable, str(Path(__file__).resolve()), "--worker-root", str(root),
                 "--repeats", str(args.repeats), "--warmup", str(args.warmup)],
                cwd=root, env=env, text=True,
            )
            measurements[label].append(json.loads(result))

    rows = []
    for i, before in enumerate(measurements["before"][0]):
        after = measurements["after"][0][i]
        old, new = np.asarray(before["value"]), np.asarray(after["value"])
        assert before["shape"] == after["shape"]
        np.testing.assert_allclose(new, old, atol=1e-10, rtol=1e-10)
        delta = new - old
        row = {key: before[key] for key in ("dof", "batch", "backend", "operation", "shape")}
        for label in ("before", "after"):
            runs = [group[i] for group in measurements[label]]
            for run in runs:
                np.testing.assert_allclose(run["value"], old, atol=1e-10, rtol=1e-10)
            samples = [sample for run in runs for sample in run["samples_ms"]]
            row[label + "_samples_ms"] = samples
            row[label + "_first_ms"] = [run["first_ms"] for run in runs]
            row[label + "_median_ms"] = statistics.median(samples)
        row["speedup"] = row["before_median_ms"] / row["after_median_ms"]
        row["max_abs_error"] = float(np.max(np.abs(delta))) if delta.size else 0.0
        row["relative_frobenius_error"] = float(np.linalg.norm(delta) / max(np.linalg.norm(old), 1e-300))
        rows.append(row)

    digest = hashlib.sha256()
    for path in sorted((ROOT / "robokots").rglob("*.py")):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    metadata = dict(baseline=baseline_commit, current_source_sha256=digest.hexdigest(),
                    python=sys.version, platform=platform.platform(), cpu=cpu,
                    numpy=version("numpy"), jax=version("jax"), mathrobo=version("mathrobo"),
                    rust_extension_sha256=hashlib.sha256(Path(rust.__file__).read_bytes()).hexdigest(),
                    seed=SEED, gravity=GRAVITY, order=5, dtype="float64", repeats_per_worker=args.repeats,
                    warmup=args.warmup, worker_order=["before", "after", "after", "before"], threads=1,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(dict(metadata=metadata, rows=rows), indent=2) + "\n")
    lines = ["# State dictionary separation timing", "", f"Baseline: `{baseline_commit}`; current uncommitted source hash: `{digest.hexdigest()}`.",
             "", f"Environment: {cpu}; Python {platform.python_version()}; NumPy {metadata['numpy']}; mathrobo {metadata['mathrobo']}.",
             f"Measured: {metadata['timestamp']}. Same Rust extension and virtualenv; one BLAS/OpenMP thread.",
             "", f"Order 5, float64, seed {SEED}, world gravity {GRAVITY}. Sample 3-DOF arm and generated 16-DOF humanoid tree.",
             f"ABBA sequential workers; each case has one first call, {args.warmup} warmups and {args.repeats} measured calls per worker.",
             "First-call times (after model/state initialization) and individual samples are in the JSON. Module-import/model-construction costs are excluded.",
             "", "Default operations include motion import, state calculation and selected value retrieval. Before defaults export a dictionary; after defaults do not.",
             "`dynamics_no_dict` explicitly disables export on both revisions. `dynamics_export` includes a full dictionary export on both, including new snapshot copies.",
             "`jacobian_cached` measures the dense torque_diff2 Jacobian with state already computed. Rust dispatch may use existing NumPy fallbacks.",
             "No JAX AD/JIT timing or numerical differentiation timing is included. NumPy/Rust outputs are synchronous; Python/NumPy conversion costs are included.",
             "", "| DOF | Batch | Backend | Operation | Before ms | After ms | Before/after | Max abs error | Rel. Frobenius error |",
             "|---:|---:|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['dof']} | {row['batch']} | {row['backend']} | {row['operation']} | {row['before_median_ms']:.4f} | {row['after_median_ms']:.4f} | {row['speedup']:.2f} | {row['max_abs_error']:.2e} | {row['relative_frobenius_error']:.2e} |")
    lines += ["", "Ratios near 1 should be interpreted as similar performance, not a demonstrated improvement/regression.",
              "", "Reproduce: `.venv/bin/python -m developer.benchmarks.state_dictionary_compare`."]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(args.output.with_suffix(".md"))


if __name__ == "__main__":
    main()
