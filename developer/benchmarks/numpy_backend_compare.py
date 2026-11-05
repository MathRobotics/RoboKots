"""Compare NumPy-only on-demand operations with Rust on identical inputs.

Run: .venv/bin/python -m developer.benchmarks.numpy_backend_compare
This is a same-build backend comparison, not a historical before/after timing.
"""
from pathlib import Path
import json
import platform
import statistics
import time

import numpy as np

from robokots.kots import Kots

ROOT = Path(__file__).resolve().parents[2]


def main():
    seed, warmup, samples, calls = 192, 3, 9, 3
    rng = np.random.default_rng(seed)
    rows = []
    for batch in (None, 6):
        objects = {backend: Kots.from_urdf_file(
            str(ROOT / "tests/test_model/branched_fixed.urdf"), backend=backend)
            for backend in ("rust", "numpy")}
        dof = objects["rust"].dof()
        shape = () if batch is None else (batch,)
        motion = rng.normal(scale=.2, size=shape + (dof*3,))
        q, v, a = (np.ascontiguousarray(motion[..., i::3]) for i in range(3))
        gravity = [.2, -.3, -9.81]
        torque = objects["rust"].inverse_dynamics(q, v, a, gravity=gravity)
        direction = rng.normal(size=shape + (dof*2,))
        weights = np.ones(shape + (1,))
        operations = {}
        for backend, k in objects.items():
            k.import_motions(motion)
            operations[backend] = {
                "energy": k.kinetic_energy_state,
                "energy_jvp": lambda k=k: k.kinetic_energy_jacobian_mul(direction),
                "energy_vjp": lambda k=k: k.kinetic_energy_jacobian_transpose_mul(weights),
                "inverse_dynamics": lambda k=k: k.inverse_dynamics(q, v, a, gravity=gravity),
                "forward_dynamics": lambda k=k: k.forward_dynamics(q, v, torque, gravity=gravity),
            }
        for operation in operations["rust"]:
            times, outputs, first = {}, {}, {}
            for backend in objects:
                fn = operations[backend][operation]
                start = time.perf_counter_ns()
                outputs[backend] = np.asarray(fn())
                first[backend] = (time.perf_counter_ns()-start)/1e6
                for _ in range(warmup):
                    fn()
                durations = []
                for _ in range(samples):
                    start = time.perf_counter_ns()
                    for _ in range(calls):
                        fn()
                    durations.append((time.perf_counter_ns()-start)/calls/1e6)
                times[backend] = statistics.median(durations)
            delta = outputs["numpy"]-outputs["rust"]
            np.testing.assert_allclose(outputs["numpy"], outputs["rust"], atol=1e-10, rtol=1e-10)
            rows.append(dict(batch_shape=list(shape), operation=operation,
                             median_ms=times, first_call_ms=first,
                             max_abs=float(np.max(np.abs(delta))),
                             relative_frobenius=float(np.linalg.norm(delta.ravel())/max(np.linalg.norm(outputs["rust"].ravel()), 1e-30))))
    result = dict(environment=dict(platform=platform.platform(), python=platform.python_version(), numpy=np.__version__),
                  seed=seed, warmup=warmup, samples=samples, calls_per_sample=calls,
                  model="tests/test_model/branched_fixed.urdf", dtype="float64", gravity=gravity,
                  scope="same-build NumPy vs Rust; all state/primal computation and input/output conversion included; model construction excluded; no cached state or Python fallback on Rust path",
                  rows=rows)
    destination = ROOT / "developer/benchmarks/results/numpy_backend_compare.json"
    destination.write_text(json.dumps(result, indent=2)+"\n")
    lines = ["# NumPy-only / Rust comparison", "", result["scope"], "",
             f"Seed {seed}; warmup {warmup}; {samples} samples × {calls} calls; median milliseconds.", "",
             "| Batch | Operation | NumPy ms | Rust ms | Max absolute difference |",
             "|---|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['batch_shape']} | {row['operation']} | {row['median_ms']['numpy']:.6f} | {row['median_ms']['rust']:.6f} | {row['max_abs']:.3g} |")
    lines += ["", "NumPy forward dynamics builds and solves a mass matrix; Rust uses ABA.",
              "Relative Frobenius errors, first calls, and environment are in the adjacent JSON."]
    destination.with_suffix(".md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
