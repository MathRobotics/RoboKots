"""Compare legacy Python-model input with native-first input in the same build.

python -m developer.benchmarks.native_model_input_compare --output results.json
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time

import numpy as np
from robokots.kots import Kots, StateType
from robokots.urdf_io import load_urdf_file


def measure_pair(functions, warmup=20, samples=30, calls=20):
    first, timings = {}, {key: [] for key in functions}
    for label, fn in functions.items():
        start = time.perf_counter_ns()
        fn()
        first[label] = (time.perf_counter_ns() - start) / 1e6
        for _ in range(warmup):
            fn()
    for i in range(samples):
        labels = list(functions) if i % 2 == 0 else list(reversed(functions))
        for label in labels:
            start = time.perf_counter_ns()
            for _ in range(calls):
                functions[label]()
            timings[label].append((time.perf_counter_ns() - start) / calls / 1e6)
    result = {key: {"first_ms": first[key], "median_ms": statistics.median(values),
                    "samples_ms": values} for key, values in timings.items()}
    result["change_percent"] = (result["native"]["median_ms"] / result["legacy"]["median_ms"] - 1) * 100
    return result


def run(output):
    root = Path(__file__).resolve().parents[2]
    model = root / "tests/test_model/branched_fixed.urdf"
    data = load_urdf_file(str(model))
    gravity = np.array([.2, -.3, -9.81])
    rng = np.random.default_rng(346)
    reference = Kots.from_json_data(data, order=4)
    motion = rng.normal(scale=.2, size=reference.dof()*4)
    states = [StateType("joint", "a_elbow", "acc", "world"),
              StateType("link", "a_tip", "force_diff1", "world"),
              StateType("link", "b_payload", "pos")]
    results = {}
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "model.json"
        path.write_text(json.dumps(data))

        def create(source, backend, dynamics=False):
            if source == "dict":
                k = Kots.from_json_data(data, order=4, backend=backend)
            elif source == "json":
                k = Kots.from_json_file(path, order=4, backend=backend)
            else:
                k = Kots.from_urdf_file(str(model), order=4, backend=backend)
            k._rust_model_info()  # Both paths ready for Rust, including metadata.
            if dynamics:
                k.import_motions(motion)
                k.dynamics(backend="rust", gravity=gravity)
            return k

        for source in ("dict", "json", "urdf"):
            for dynamics in (False, True):
                name = f"{source}/" + ("construct_and_dynamics" if dynamics else "model_ready")
                results[name] = measure_pair({label: lambda backend=backend: create(source, backend, dynamics)
                                             for label, backend in (("legacy", None), ("native", "rust"))})
        models = {label: create("dict", backend, True) for label, backend in (("legacy", None), ("native", "rust"))}
        jac = models["legacy"].jacobian(states)
        direction = rng.normal(size=jac.shape[-1])
        weight = rng.normal(size=jac.shape[-2])
        accuracy = {}
        operations = {
            "state_values": lambda k: k.state_info_list(states),
            "dense_cached": lambda k: k.jacobian(states),
            "jvp_cached": lambda k: k.jacobian_mul(states, direction),
            "vjp_cached": lambda k: k.jacobian_transpose_mul(states, weight),
        }
        for name, operation in operations.items():
            functions = {label: lambda k=k: operation(k) for label, k in models.items()}
            old, new = functions["legacy"](), functions["native"]()
            np.testing.assert_allclose(old, new, atol=1e-10, rtol=1e-10)
            difference = np.asarray(new) - old
            accuracy[name] = {"max_abs": float(np.max(np.abs(difference))),
                              "relative_frobenius": float(np.linalg.norm(difference) / max(np.linalg.norm(old), 1e-300))}
            results[name] = measure_pair(functions)
        def update(k):
            k.import_motions(motion)
            return k.dynamics(backend="rust", gravity=gravity)
        results["import_and_dynamics"] = measure_pair({label: lambda k=k: update(k) for label, k in models.items()})
        assert models["native"]._python_robot_ is None
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "comparison": "Legacy and native-first input paths in the same current code/build; not different commit binaries",
        "environment": {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
                        "threads": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}},
        "workload": {"model": str(model.relative_to(root)), "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                     "order": 4, "seed": 346, "gravity": gravity.tolist(), "dtype": "float64", "batch": "single",
                     "warmup": 20, "samples": 30, "calls": 20,
                     "notes": "Alternating AB/BA blocks. Python/PyO3/NumPy conversion included. File cache warmed. First call excludes import/build. No JIT."},
        "timings": results, "accuracy": accuracy,
    }, indent=2) + "\n")
    for name, result in results.items():
        print(name, *(f"{result[label]['median_ms']*1000:.3f} us" for label in ("legacy", "native")), f"{result['change_percent']:+.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
