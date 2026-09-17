"""Table-style accuracy benchmark, not a reproduction of the historical paper.

Run: python -u -m developer.benchmarks.jacobian_accuracy_table
No execution-time claims: AD uses float64 jacfwd/jacrev without JIT.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from robokots.kots import Kots, StateType
from robokots.outward.diff.dynamics_jax import dynamics_jax, dynamics_state_vector_jax
from .dynamics_autodiff_compare import error
from .jacobian_dof_sweep import build_serial_arm_model


ROWS = (
    ("link velocity / local", "link", "velocity", "local"),
    ("link momentum / local", "link", "momentum", "local"),
    ("link momentum / world", "link", "momentum", "world"),
    ("joint momentum / local", "joint", "momentum", "local"),
    ("joint momentum / world", "joint", "momentum", "world"),
    ("link force / local", "link", "force", "local"),
    ("joint force / local", "joint", "force", "local"),
    ("joint torque", "joint", "torque", "local"),
)
VELOCITY_TYPES = ("vel", "acc", "jerk", "snap", "crackle")
PAIRS = (
    ("numerical", "analytic"), ("autodiff_forward", "analytic"),
    ("autodiff_reverse", "analytic"), ("numerical", "autodiff_forward"),
    ("numerical", "autodiff_reverse"), ("autodiff_reverse", "autodiff_forward"),
)


def make_model(dof):
    model = build_serial_arm_model(dof, [0.3, 0.2, 0.1], [0, 0, 1], 1.57)
    for i, joint in enumerate(model["joints"][1:]):
        joint["axis"] = np.eye(3)[i % 3].tolist()
    return model


def selections(kots, k):
    link = kots.link_name_list()[-1]
    joint = next(j.name for j in kots.robot_.joints if j.dof)
    return [StateType(owner, link if owner == "link" else joint,
                      VELOCITY_TYPES[k] if family == "velocity" else
                      family + (f"_diff{k}" if k else ""), frame)
            for _, owner, family, frame in ROWS]


def evaluate(model, motion, k, gravity):
    """One sample/derivative order; each row uses its minimal motion order."""
    kots = Kots.from_json_data(model, order=k + 3)
    kots.import_motions(motion[:, :k + 3].reshape(-1))
    kots.dynamics(backend="numpy", gravity=gravity, materialize_dict=False)
    states = selections(kots, k)
    results = []
    for (label, *_), state in zip(ROWS, states):
        order = state.time_order
        x = jnp.asarray(kots.motion(order), dtype=jnp.float64)
        if state.data_type in VELOCITY_TYPES:
            def value(x):
                return dynamics_jax(kots.robot_, x, order, gravity)["link_velocity"][state.owner_name][k]
        else:
            def value(x):
                return dynamics_state_vector_jax(kots.robot_, x, [state], order, gravity)
        values = {
            "analytic": np.asarray(kots.jacobian(state)),
            "numerical": np.asarray(kots.jacobian(state, numerical=True)),
            "autodiff_forward": np.asarray(jax.block_until_ready(jax.jacfwd(value)(x))),
            "autodiff_reverse": np.asarray(jax.block_until_ready(jax.jacrev(value)(x))),
        }
        expected = (1 if state.data_type.startswith("torque") else 6, kots.dof() * order)
        if any(v.shape != expected for v in values.values()):
            raise AssertionError(f"Unexpected Jacobian shape for {label}: expected {expected}")
        results.append({
            "row": label, "k": k, "owner": state.owner_name,
            "data_type": state.data_type, "frame": state.frame_name,
            "motion_order": order, "shape": list(expected),
            "errors": {f"{a}_vs_{b}": error(values[a], values[b]) for a, b in PAIRS},
        })
    return results


def report(metadata, results):
    lines = ["# Jacobian accuracy table", "",
             "New reproducible experiment; not a reconstruction of the historical table.", "",
             f"Measured: {metadata['utc']}", "",
             f"Environment: {metadata['platform']}; Python {metadata['python']}; "
             f"NumPy {metadata['numpy']}; JAX {metadata['jax']}; {metadata['devices']}; float64.", "",
             f"Generated {metadata['dof']}-DOF serial rigid arm, alternating x/y/z revolute axes. "
             "Last link and first actuated joint (subtree quantities). Exact model and motion are in JSON.", "",
             f"Gravity: {metadata['gravity']}; seed: {metadata['seed']}; "
             f"samples: {metadata['samples']}; motion distribution: N(0, 0.4²).", "",
             "k is the ordinary kth time derivative of the named quantity, not the order of "
             "differentiation with respect to motion. k=0 is the undifferentiated quantity. "
             "Velocity/momentum require motion order k+2; force/torque require k+3. "
             "Jacobian columns are owner-major q, qdot, ... through that required order.", "",
             "NumPy analytic vs central differences (public numerical=True, eps=1e-8) vs "
             "JAX forward/reverse AD (jacfwd/jacrev, no JIT). Dynamics AD uses the same pure function as "
             "jacobian_autodiff(); velocity AD uses its recursive velocity series. "
             "No Rust or JIT timing comparison is performed here.", "",
             "Maximum absolute difference: max(abs(A-B)). Relative Frobenius: "
             "norm(A-B)/max(norm(B), 1e-30). Each cell takes the maximum over completed samples; "
             "the second named method is the relative-error denominator, not an exact oracle.", "",
             f"Completed cases: {len(results)} / {metadata['samples'] * 8 * (metadata['max_derivative'] + 1)}.", ""]
    for a, b in PAIRS:
        pair = f"{a}_vs_{b}"
        for metric in ("max_abs", "relative_frobenius"):
            lines += [f"## {a} vs {b}: {metric}", "",
                      "| Quantity | " + " | ".join(f"k = {k}" for k in range(metadata['max_derivative'] + 1)) + " |",
                      "|---|" + "---:|" * (metadata['max_derivative'] + 1)]
            for label, *_ in ROWS:
                cells = []
                for k in range(metadata['max_derivative'] + 1):
                    values = [r['errors'][pair][metric] for r in results if r['row'] == label and r['k'] == k]
                    cells.append(f"{max(values):.3e}" if values else "pending")
                lines.append(f"| {label} | " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dof", type=int, default=7)
    parser.add_argument("--max-derivative", type=int, choices=range(5), default=4)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--gravity", type=float, nargs=3, default=[0.3, -0.4, -9.81])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results") / "jacobian_accuracy_table")
    args = parser.parse_args()
    if args.dof < 1 or args.samples < 1 or args.seed < 0 or not np.all(np.isfinite(args.gravity)):
        parser.error("dof/samples must be positive, seed nonnegative and gravity finite")
    jax.config.update("jax_enable_x64", True)
    model = make_model(args.dof)
    # Always generate seven orders: lower-order runs use identical input prefixes.
    motions = np.random.default_rng(args.seed).normal(scale=0.4, size=(args.samples, args.dof, 7))
    root = Path(__file__).resolve().parents[2]
    metadata = {
        "utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
        "python": platform.python_version(), "numpy": np.__version__, "jax": jax.__version__,
        "devices": str(jax.devices()), "dof": args.dof, "samples": args.samples,
        "seed": args.seed, "max_derivative": args.max_derivative, "gravity": args.gravity,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        "numerical_method": "central", "numerical_eps": 1e-8, "dtype": "float64",
        "autodiff": "jax.jacfwd and jax.jacrev, no JIT", "motion_layout": "sample, dof, order",
    }
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for sample, motion in enumerate(motions):
        for k in range(args.max_derivative + 1):
            print(f"Sample {sample + 1}/{args.samples}, k={k} ...", flush=True)
            results.extend(dict(item, sample=sample) for item in evaluate(model, motion, k, args.gravity))
            args.output.with_suffix(".json").write_text(json.dumps({
                "metadata": metadata, "model": model, "motions": motions.tolist(), "results": results,
            }, indent=2, allow_nan=False), encoding="utf-8")
            args.output.with_suffix(".md").write_text(report(metadata, results), encoding="utf-8")
    print(f"Report: {args.output.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
