"""Alternate native binaries in fresh processes, without replacing the installed extension.

Usage: python -m developer.benchmarks.native_boundary_diagnosis
  --before /path/to/baseline.dylib --after /path/to/current.so --output results.json
"""
import argparse
import hashlib
import importlib.util
from importlib.machinery import ExtensionFileLoader
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys


def worker(binary, output, return_ops=False):
    name = "robokots._rust_core"
    spec = importlib.util.spec_from_file_location(
        name, binary, loader=ExtensionFileLoader(name, str(binary)))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    import numpy as np
    from robokots.kots import Kots, StateType
    from robokots.outward.rust.model import _model_data_from_robot
    from .core_state_layout import measure

    root = Path(__file__).resolve().parents[2]
    model = root / "tests/test_model/branched_fixed.urdf"
    k = Kots.from_urdf_file(str(model), order=4)
    data = _model_data_from_robot(k.robot_)
    rng = np.random.default_rng(812)
    x = rng.normal(scale=.2, size=k.dof()*4)
    gravity = np.array([.2, -.3, -9.81])
    states = [
        StateType("joint", "a_elbow", "acc", "world"),
        StateType("link", "a_tip", "jerk", "world"),
        StateType("link", "b_payload", "frame", "world"),
        StateType("joint", "a_shoulder", "pos", "local"),
        StateType("link", "a_tip", "force_diff1"),
        StateType("joint", "b_shoulder", "torque_diff1"),
    ]
    k._rust_compiled_robot_ = module.RustCompiledRobot.from_model_data(data)
    k.import_motions(x)
    k.dynamics(backend="rust", gravity=gravity)
    jac = k.jacobian(states)
    v = rng.normal(size=jac.shape[-1])
    w = rng.normal(size=jac.shape[-2])
    robot = k._rust_compiled_robot()
    raw = robot.create_outward_data(4)
    workspace = robot.create_selected_workspace(4)
    specs, _ = k._rust_selected_dynamics_specs(states, 4)
    motion = x.reshape(1, -1)
    direction = v.reshape(1, -1, 1)
    weight = w.reshape(1, -1, 1)
    basis = np.eye(x.size)[None, ...]
    q = np.ascontiguousarray(x[0::4])
    qdot = np.ascontiguousarray(x[1::4])
    qddot = np.ascontiguousarray(x[2::4])

    def state():
        k.import_motions(x)
        k.dynamics(backend="rust", gravity=gravity)

    operations = {
        "control/serialize_only": lambda: _model_data_from_robot(k.robot_),
        "model/compile_dict": lambda: module.RustCompiledRobot.from_model_data(data),
        "model/serialize_and_compile": lambda: module.RustCompiledRobot.from_model_data(_model_data_from_robot(k.robot_)),
        "raw/model_dof_getter": lambda: robot.dof,
        "raw/rnea": lambda: robot.rnea(q, qdot, qddot, gravity),
        "raw/compute_dynamics": lambda: raw.compute_dynamics(x, gravity),
        "public/import_and_dynamics": state,
        "raw/dense_cached": lambda: workspace.apply(motion, basis, specs, gravity),
        "raw/jvp_cached": lambda: workspace.apply(motion, direction, specs, gravity),
        "raw/vjp_cached": lambda: workspace.apply(motion, weight, specs, gravity, True),
        "public/dense_cached": lambda: k.jacobian(states),
        "public/jvp_cached": lambda: k.jacobian_mul(states, v),
        "public/vjp_cached": lambda: k.jacobian_transpose_mul(states, w),
    }
    if return_ops:
        return operations
    timings = {name: measure(fn, 20, 30, 50) for name, fn in operations.items()}
    outputs = {}
    for name in ("raw/rnea", "raw/dense_cached", "raw/jvp_cached", "raw/vjp_cached",
                 "public/dense_cached", "public/jvp_cached", "public/vjp_cached"):
        outputs[name] = np.asarray(operations[name]()).tolist()
    np.testing.assert_allclose(np.asarray(outputs["raw/dense_cached"])[0], jac, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(outputs["raw/jvp_cached"])[0, :, 0], jac @ v, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(outputs["raw/vjp_cached"])[0, :, 0], jac.T @ w, atol=1e-12, rtol=1e-12)
    output.write_text(json.dumps({
        "extension_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "numpy": np.__version__, "threads": {key: os.environ.get(key) for key in
                        ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}},
        "workload": {"model": str(model.relative_to(root)), "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                     "seed": 812, "order": 4, "gravity": gravity.tolist(), "dtype": "float64",
                     "warmup": 20, "samples": 30, "calls": 50,
                     "notes": "Single sample; cached derivative primals; raw includes PyO3 and NumPy conversion. No JIT."},
        "timings": timings, "outputs": outputs,
    }, indent=2) + "\n")


def paired(before, after, output):
    import time
    import numpy as np
    operations = {label: worker(binary.resolve(), None, return_ops=True)
                  for label, binary in (("before", before), ("after", after))}
    values = {}
    summary = {}
    # Each block alternates order; both binaries stay loaded and use separate models/workspaces.
    for key in operations["before"]:
        funcs = {label: ops[key] for label, ops in operations.items()}
        for fn in funcs.values():
            for _ in range(50):
                fn()
        samples = {"before": [], "after": []}
        for block in range(100):
            order = ("before", "after") if block % 2 == 0 else ("after", "before")
            for label in order:
                start = time.perf_counter_ns()
                for _ in range(100):
                    funcs[label]()
                samples[label].append((time.perf_counter_ns() - start) / 100 / 1e6)
        a, b = samples["before"], samples["after"]
        ratios = [(y / x - 1) * 100 for x, y in zip(a, b)]
        summary[key] = {"before_median_ms": statistics.median(a),
                        "after_median_ms": statistics.median(b),
                        "median_paired_change_percent": statistics.median(ratios),
                        "paired_change_p10_p90": np.percentile(ratios, [10, 90]).tolist(),
                        "samples_ms": samples}
        has_numeric_output = (
            key.startswith("raw/") and key not in ("raw/model_dof_getter", "raw/compute_dynamics")
            or key.startswith("public/") and key != "public/import_and_dynamics"
        )
        if has_numeric_output:
            old, new = np.asarray(funcs["before"]()), np.asarray(funcs["after"]())
            np.testing.assert_array_equal(old, new)
            values[key] = {"max_abs": float(np.max(np.abs(new - old))), "relative_frobenius": 0.0}
        print(key, summary[key]["median_paired_change_percent"], flush=True)
    output.write_text(json.dumps({
        "method": "Same process, two separately loaded binaries; independent models/workspaces. 50 warmups; 100 paired blocks x 100 calls; order alternates AB/BA.",
        "extensions": {label: hashlib.sha256(path.read_bytes()).hexdigest() for label, path in (("before", before), ("after", after))},
        "summary": summary, "accuracy_vs_before": values,
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--before-root", type=Path, help="Baseline checkout for Python code compatibility (fresh-process mode)")
    parser.add_argument("--after", type=Path)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.extension:
        worker(args.extension.resolve(), args.output)
        return
    if args.before is None or args.after is None:
        parser.error("--before and --after are required")
    if args.paired:
        paired(args.before, args.after, args.output)
        return
    sequence = ["before", "after", "after", "before"] * 2
    runs = []
    for i, label in enumerate(sequence):
        path = args.output.with_name(args.output.stem + f"_{i}_{label}.json")
        binary = args.before if label == "before" else args.after
        subprocess.run([sys.executable, "-m", __spec__.name, "--extension", str(binary.resolve()),
                        "--output", str(path)], check=True,
                       cwd=args.before_root if label == "before" else None)
        runs.append({"label": label, "file": path.name, "data": json.loads(path.read_text())})
        print(f"{i+1}/{len(sequence)} {label}", flush=True)
    reference = runs[0]["data"]
    for run in runs:
        assert run["data"]["workload"] == reference["workload"]
        assert run["data"]["outputs"] == reference["outputs"], "Output mismatch"
    summary = {}
    for key in reference["timings"]:
        groups = {label: [run["data"]["timings"][key]["median_ms"] for run in runs if run["label"] == label]
                  for label in ("before", "after")}
        summary[key] = {label: {"session_medians_ms": values, "median_ms": statistics.median(values),
                                "min_ms": min(values), "max_ms": max(values)}
                        for label, values in groups.items()}
        summary[key]["change_percent"] = (summary[key]["after"]["median_ms"] /
                                          summary[key]["before"]["median_ms"] - 1) * 100
    args.output.write_text(json.dumps({
        "sequence": sequence, "sessions": [{"label": run["label"], "file": run["file"],
                                           "extension_sha256": run["data"]["extension_sha256"]} for run in runs],
        "workload": reference["workload"], "environment": reference["environment"],
        "summary": summary, "accuracy_vs_before": {"max_abs": 0.0, "relative_frobenius": 0.0}
    }, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
