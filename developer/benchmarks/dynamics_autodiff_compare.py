"""Reproducible dense dynamics Jacobian timing and accuracy comparison.

Run from the repository root:
  python -u -m developer.benchmarks.dynamics_autodiff_compare
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from robokots.kots import Kots, StateType
from robokots.outward.diff.dynamics_jax import dynamics_state_vector_jax
from .jacobian_dof_sweep import build_serial_arm_model


ROOT = Path(__file__).resolve().parents[2]
GRAVITY = [0.3, -0.4, -9.81]


def measure(fn, repeats, warmup=2):
    for _ in range(warmup):
        fn()
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = fn()
        elapsed.append((time.perf_counter_ns() - start) / 1e6)
    return np.asarray(result), {
        "median_ms": float(np.median(elapsed)),
        "min_ms": float(np.min(elapsed)),
        "p90_ms": float(np.percentile(elapsed, 90)),
        "repeats": repeats,
    }


def error(a, b):
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise AssertionError("Nonfinite or incompatible Jacobians")
    return {
        "max_abs": float(np.max(np.abs(a - b))),
        "relative_frobenius": float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)),
    }


def model_factory(name, order):
    if name == "sample4":
        return Kots.from_json_file(str(ROOT / "examples/model/sample_robot.json"), order=order)
    if name == "branched3":
        return Kots.from_urdf_file(str(ROOT / "tests/test_model/branched_fixed.urdf"), order=order)
    dof = int(name.removeprefix("serial"))
    model = build_serial_arm_model(dof, [0.3, 0.2, 0.1], [0, 0, 1], 1.57)
    for i, joint in enumerate(model["joints"][1:]):
        joint["axis"] = np.eye(3)[i % 3].tolist()
    return Kots.from_json_data(model, order=order)


def evaluate(model_name, data_type, repeats):
    state = StateType("total_joint", "total_joint", data_type)
    if data_type in ("momentum", "force"):
        temp = model_factory(model_name, 3)
        state = StateType("link", temp.link_name_list()[-1], data_type)
    order = state.time_order
    kots = model_factory(model_name, order)
    # Slice a common order-5 motion so all cases use the same q, qdot, etc.
    full = np.random.default_rng(20260913 + kots.dof()).normal(scale=0.4, size=(kots.dof(), 5))
    motion = full[:, :order].reshape(-1)
    kots.import_motions(motion)
    states = kots._state_type_list(state)
    timing, values = {}, {}
    for backend in ("numpy", "rust"):
        def complete(backend=backend):
            kots.import_motions(motion)
            kots.dynamics(backend=backend, gravity=GRAVITY, materialize_dict=False)
            return kots.jacobian(state)
        values[backend], timing[backend + "_full"] = measure(complete, repeats)
        _, timing[backend + "_cached"] = measure(lambda: kots.jacobian(state), repeats)

    # Numerical reference uses the existing API (central difference, eps=1e-8).
    # Its perturbed evaluations use the existing NumPy outward implementation.
    kots.dynamics(backend="numpy", gravity=GRAVITY, materialize_dict=False)
    def numerical():
        kots.import_motions(motion)
        return kots.jacobian(state, numerical=True)
    values["numerical"], timing["numerical"] = measure(numerical, 1 if kots.dof() >= 16 else 3, warmup=0)

    def eager():
        kots.import_motions(motion)
        return kots.jacobian_autodiff(state)
    values["jax_eager"], timing["jax_eager"] = measure(eager, 3, warmup=1)

    # x is a runtime argument: do not capture the evaluated motion as a constant.
    def value(x):
        return dynamics_state_vector_jax(kots.robot_, x, states, order, GRAVITY)
    compiled = jax.jit(jax.jacfwd(value))
    def jit_run():
        return np.asarray(jax.block_until_ready(compiled(jnp.asarray(motion))))
    # Includes tracing, compilation, execution, and host conversion.
    _, timing["jax_jit_first"] = measure(jit_run, 1, warmup=0)
    values["jax_jit"], timing["jax_jit"] = measure(jit_run, max(100, repeats))
    errors = {name: error(array, values["jax_jit"]) for name, array in values.items() if name != "jax_jit"}

    # Check changing runtime inputs too, outside the timing interval.
    additional = []
    for seed in (11, 29):
        x = np.random.default_rng(seed).normal(scale=0.4, size=motion.shape)
        kots.import_motions(x)
        kots.dynamics(backend="numpy", gravity=GRAVITY, materialize_dict=False)
        reference = kots.jacobian(state)
        result = np.asarray(jax.block_until_ready(compiled(jnp.asarray(x))))
        additional.append(error(result, reference))
        np.testing.assert_allclose(result, reference, atol=1e-8, rtol=1e-9)
    np.testing.assert_allclose(values["numpy"], values["jax_jit"], atol=1e-8, rtol=1e-9)
    np.testing.assert_allclose(values["rust"], values["jax_jit"], atol=1e-8, rtol=1e-9)
    return {
        "model": model_name, "dof": kots.dof(), "output": data_type,
        "owner": state.owner_type, "motion_order": order,
        "jacobian_shape": list(values["jax_jit"].shape),
        "timing": timing, "error_vs_jax_jit": errors,
        "additional_input_errors": additional,
    }


def write_report(path, metadata, results):
    lines = [
        "# Dynamics Jacobian timing and accuracy", "",
        f"Measured: {metadata['utc']}", "",
        f"Environment: {metadata['platform']}; {metadata['cpu']}; Python {metadata['python']}; "
        f"JAX {metadata['jax']}; NumPy {metadata['numpy']}; {metadata['devices']}; float64.", "",
        "Dense Jacobians, single sample, gravity [0.3, -0.4, -9.81]. "
        "Momentum/force select the last link; torque selects all active joints. "
        "Generated serial models use alternating x/y/z joint axes.", "",
        "## Median execution time (ms)", "",
        "Full analytic timings include motion import, dynamics state computation "
        "(without dictionary materialization), and Jacobian computation. "
        "Cached timings exclude state computation. JAX JIT timings include input "
        "conversion, synchronized execution and NumPy output conversion. "
        "JIT compilation is reported separately; a compiled function must be reused. "
        "The public jacobian_autodiff() is the eager column, not the JIT column.", "",
        "NumPy/Rust labels indicate the requested dynamics-state backend. "
        "With the nonzero gravity used here, link momentum/force dense Jacobians "
        "fall back to the Python analytic path even after Rust state computation. "
        "The torque cases use Rust RNEA/CMTM derivative kernels.", "",
        "| Model | Output | Shape | NumPy full | Rust full | NumPy cached | Rust cached | Numerical | JAX eager | JAX JIT warm | JIT first (ms) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        t = item["timing"]
        times = [t[key]["median_ms"] for key in ("numpy_full", "rust_full", "numpy_cached", "rust_cached", "numerical", "jax_eager", "jax_jit")]
        shape = "×".join(map(str, item["jacobian_shape"]))
        lines.append(f"| {item['model']} | {item['output']} | {shape} | " + " | ".join(f"{x:.4f}" for x in times) + f" | {t['jax_jit_first']['median_ms']:.4f} |")
    lines.extend([
        "", "## Agreement with JAX JIT", "",
        "Errors are maximum absolute difference and relative Frobenius norm "
        "(||A−J||F / ||J||F). JAX is a comparison reference, not an exact oracle. "
        "NumPy and Rust use analytic derivatives; numerical=True uses central "
        "differences with the library default eps=1e-8.", "",
        "| Model | Output | NumPy max abs | Rust max abs | Numerical max abs | Numerical relative | NumPy relative |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for item in results:
        e = item["error_vs_jax_jit"]
        lines.append(f"| {item['model']} | {item['output']} | {e['numpy']['max_abs']:.3e} | {e['rust']['max_abs']:.3e} | {e['numerical']['max_abs']:.3e} | {e['numerical']['relative_frobenius']:.3e} | {e['numpy']['relative_frobenius']:.3e} |")
    lines.extend([
        "", "## Measurement details", "",
        f"Analytic: {metadata['repeats']} timed calls after two warmups. "
        "JAX JIT: at least 100 calls after two warmups. Eager: three calls "
        "after one warmup. Numerical: three calls, or one for 16 DOF, no warmup. "
        "JSON contains min/p90 and repeat counts. These are local machine measurements, "
        "not universal performance guarantees.", "",
        "Each compiled function was additionally checked at two different random "
        "inputs against NumPy analytic derivatives. Model creation and initial "
        "backend initialization are excluded from warm execution timings.", "",
        "Reproduce: `python -u -m developer.benchmarks.dynamics_autodiff_compare`", "",
    ])
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results") / "dynamics_autodiff")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    jax.config.update("jax_enable_x64", True)
    cpu = platform.processor() or platform.machine()
    metadata = {
        "utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
        "cpu": cpu, "python": platform.python_version(), "jax": jax.__version__,
        "numpy": np.__version__, "devices": str(jax.devices()), "repeats": args.repeats,
    }
    cases = [("sample4", key) for key in ("momentum", "force", "torque", "torque_diff1", "torque_diff2")]
    cases += [(name, key) for name in ("branched3", "serial8", "serial16") for key in ("torque", "torque_diff2")]
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(metadata), flush=True)
    for model, key in cases:
        print(f"Measuring {model} / {key} ...", flush=True)
        item = evaluate(model, key, args.repeats)
        results.append(item)
        args.output.with_suffix(".json").write_text(json.dumps({"metadata": metadata, "results": results}, indent=2))
        write_report(args.output.with_suffix(".md"), metadata, results)
        print(json.dumps(item), flush=True)
    print(f"Report: {args.output.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
