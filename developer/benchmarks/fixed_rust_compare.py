from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np

from robokots.core.state import StateType
from robokots.kots import Kots

from .common import build_model, leaf_link_names, write_csv
from .pinocchio_compare import _optional_pinocchio, build_pinocchio_model


DEFAULT_CSV_PATH = Path(__file__).resolve().with_name("fixed_rust_compare_results.csv")

PROFILES = {
    "quick": {
        "dof_list": [16, 64],
        "batch_sizes": [1, 8],
        "repeat": 30,
        "warmup": 8,
        "public_repeat": 3,
        "public_warmup": 1,
        "include_public_jacobian": False,
        "selected_repeat": 30,
        "selected_warmup": 8,
        "mixed_selected_repeat": 2,
        "mixed_selected_warmup": 1,
        "mixed_selected_dof_list": [16],
    },
    "full": {
        "dof_list": [16, 64],
        "batch_sizes": [1, 8],
        "repeat": 100,
        "warmup": 20,
        "public_repeat": 5,
        "public_warmup": 2,
        "include_public_jacobian": True,
        "selected_repeat": 100,
        "selected_warmup": 20,
        "mixed_selected_repeat": 5,
        "mixed_selected_warmup": 1,
        "mixed_selected_dof_list": [16, 64],
    },
}

TORQUE = StateType("total_joint", "total_joint", "torque")
TORQUE_DIFF2 = StateType("total_joint", "total_joint", "torque_diff2")
MIXED_TOTAL_JOINT_STATES = [
    StateType("total_joint", "total_joint", data_type)
    for data_type in ("coord", "veloc", "accel", "jerk", "torque", "torque_diff1", "torque_diff2")
]


def _measure(fn: Callable[[], object], repeat: int, warmup: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    samples = np.zeros(repeat, dtype=float)
    for i in range(repeat):
        t0 = time.perf_counter()
        fn()
        samples[i] = (time.perf_counter() - t0) * 1e3
    return {
        "mean_ms": float(samples.mean()),
        "std_ms": float(samples.std()),
        "min_ms": float(samples.min()),
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
    }


def _motion_from_qva(q: np.ndarray, v: np.ndarray, a: np.ndarray, order: int) -> np.ndarray:
    motion = np.zeros(q.size * order, dtype=float)
    motion[0::order] = q
    motion[1::order] = v
    motion[2::order] = a
    return motion


def _make_batch_motions(
    rng: np.random.Generator,
    dof: int,
    order: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q = rng.standard_normal((batch_size, dof))
    v = rng.standard_normal((batch_size, dof))
    a = rng.standard_normal((batch_size, dof))
    motion = np.stack([_motion_from_qva(q[i], v[i], a[i], order) for i in range(batch_size)])
    return q, v, a, motion


def _pinocchio_runner(pin, model, q: np.ndarray, v: np.ndarray, a: np.ndarray):
    data = model.createData()

    def run():
        if q.ndim == 1:
            return pin.rnea(model, data, q, v, a)
        out = [pin.rnea(model, data, q[i], v[i], a[i]).copy() for i in range(q.shape[0])]
        return np.stack(out)

    return run


def _core_cases(kots: Kots, q: np.ndarray, v: np.ndarray, a: np.ndarray, motion3: np.ndarray, motion5: np.ndarray):
    batch_size = 1 if motion3.ndim == 1 else motion3.shape[0]
    if batch_size == 1:
        pin_like_data = kots._create_rust_pinocchio_like_data()
        full3 = kots._create_rust_outward_state(order=3)
        torque3 = kots._create_rust_outward_state(order=3)
        full5 = kots._create_rust_outward_state(order=5)
        return {
            "pinocchio_like": lambda: pin_like_data.compute_dynamics(q, v, a),
            "cmtm_full_order3": lambda: full3.compute_dynamics(motion3),
            "cmtm_torque_order3": lambda: torque3.compute_dynamics_minimal(motion3),
            "cmtm_full_order5": lambda: full5.compute_dynamics(motion5),
        }

    full3 = kots._create_rust_batch_outward_state(order=3, batch_shape=(batch_size,))
    torque3 = kots._create_rust_batch_outward_state(order=3, batch_shape=(batch_size,))
    full5 = kots._create_rust_batch_outward_state(order=5, batch_shape=(batch_size,))
    return {
        "pinocchio_like": lambda: kots._rust_fast_rnea(q, v, a),
        "cmtm_full_order3": lambda: full3.compute_dynamics(motion3),
        "cmtm_torque_order3": lambda: torque3.compute_dynamics_minimal(motion3),
        "cmtm_full_order5": lambda: full5.compute_dynamics(motion5),
    }


def _public_case(kots: Kots, motions: np.ndarray, op_name: str, backend: str | None):
    if op_name == "torque":
        def run():
            kots.import_motions(motions)
            kots.dynamics(order=3, backend=backend, materialize_dict=False)
            return kots.state_info(TORQUE)
        return run

    if op_name == "torque_diff2":
        def run():
            kots.import_motions(motions)
            kots.dynamics(order=5, backend=backend, materialize_dict=False)
            return kots.state_info(TORQUE_DIFF2)
        return run

    if op_name == "torque_diff2_jacobian":
        def run():
            kots.import_motions(motions)
            kots.dynamics(order=5, backend=backend, materialize_dict=False)
            return kots.jacobian(TORQUE_DIFF2)
        return run

    if op_name == "mixed_total_joint_state":
        def run():
            kots.import_motions(motions)
            kots.dynamics(order=5, backend=backend, materialize_dict=False)
            return kots.state_info_list(MIXED_TOTAL_JOINT_STATES)
        return run

    if op_name == "mixed_total_joint_jacobian":
        def run():
            kots.import_motions(motions)
            kots.dynamics(order=5, backend=backend, materialize_dict=False)
            return kots.jacobian(MIXED_TOTAL_JOINT_STATES)
        return run

    raise ValueError(f"unknown public op: {op_name}")


def _selected_link_local_states(kots: Kots) -> list[StateType]:
    link_name = leaf_link_names(kots)[-1]
    return [
        StateType("link", link_name, "momentum"),
        StateType("link", link_name, "momentum_diff1"),
        StateType("link", link_name, "force"),
    ]


def _selected_case_runner(kots: Kots, states: list[StateType], op_name: str, rhs: np.ndarray | None):
    if op_name == "STATE":
        return lambda: kots.state_info_list(states)
    if op_name == "J":
        return lambda: kots.jacobian(states)
    if op_name == "JM8":
        if rhs is None:
            raise ValueError("JM8 requires rhs")
        return lambda: kots.jacobian_mul(states, rhs)
    if op_name == "JTM4":
        if rhs is None:
            raise ValueError("JTM4 requires rhs")
        return lambda: kots.jacobian_transpose_mul(states, rhs)
    raise ValueError(f"unknown selected op: {op_name}")


def _selected_cases(kots: Kots) -> list[tuple[str, list[StateType], int, tuple[str, ...]]]:
    return [
        ("torque", [TORQUE], 3, ("J", "JM8", "JTM4")),
        ("link_local_dynamics", _selected_link_local_states(kots), 3, ("J", "JM8", "JTM4")),
        ("mixed_total_joint", MIXED_TOTAL_JOINT_STATES, 5, ("STATE", "J", "JM8", "JTM4")),
    ]


def _append_row(
    rows: list[dict[str, object]],
    *,
    profile: str,
    group: str,
    dof: int,
    batch_size: int,
    op_name: str,
    impl: str,
    stats: dict[str, float],
    baseline_mean_ms: float | None = None,
    max_error: float | None = None,
) -> None:
    ratio = None if baseline_mean_ms is None else stats["mean_ms"] / baseline_mean_ms
    rows.append(
        {
            "profile": profile,
            "group": group,
            "dof": dof,
            "batch_size": batch_size,
            "op": op_name,
            "impl": impl,
            **stats,
            "ratio_to_baseline": ratio,
            "max_error": max_error,
        }
    )


def run_fixed_suite(profile_name: str, csv_path: Path, seed: int, model_kind: str) -> list[dict[str, object]]:
    profile = PROFILES[profile_name]
    pin = _optional_pinocchio()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    print("=== Fixed RoboKots Rust Benchmark ===", flush=True)
    print(f"profile   : {profile_name}", flush=True)
    print(f"dof_list  : {profile['dof_list']}", flush=True)
    print(f"batch     : {profile['batch_sizes']}", flush=True)
    print(f"csv       : {csv_path}", flush=True)
    print(f"pinocchio : {'enabled' if pin is not None else 'disabled'}", flush=True)

    for dof in profile["dof_list"]:
        model_data = build_model(dof, model_kind)
        kots = Kots.from_json_data(model_data, order=5)
        pin_model = build_pinocchio_model(pin, model_data) if pin is not None else None

        for batch_size in profile["batch_sizes"]:
            q, v, a, motion3_batch = _make_batch_motions(rng, kots.dof(), 3, batch_size)
            _, _, _, motion5_batch = _make_batch_motions(rng, kots.dof(), 5, batch_size)
            motion5_batch[:, 0::5] = q
            motion5_batch[:, 1::5] = v
            motion5_batch[:, 2::5] = a

            q_in = q[0] if batch_size == 1 else q
            v_in = v[0] if batch_size == 1 else v
            a_in = a[0] if batch_size == 1 else a
            motion3 = motion3_batch[0] if batch_size == 1 else motion3_batch
            motion5 = motion5_batch[0] if batch_size == 1 else motion5_batch

            print(f"\n-- core dof={dof} batch={batch_size} --", flush=True)
            core_cases = _core_cases(kots, q_in, v_in, a_in, motion3, motion5)
            baseline = None
            if pin is not None:
                pin_stats = _measure(
                    _pinocchio_runner(pin, pin_model, q_in, v_in, a_in),
                    profile["repeat"],
                    profile["warmup"],
                )
                baseline = pin_stats["mean_ms"]
                _append_row(
                    rows,
                    profile=profile_name,
                    group="core",
                    dof=dof,
                    batch_size=batch_size,
                    op_name="dynamics_qva",
                    impl="pinocchio",
                    stats=pin_stats,
                    baseline_mean_ms=baseline,
                )
                print(f"pinocchio           mean={pin_stats['mean_ms']:.5f}ms", flush=True)

            pin_like_stats = None
            for impl, runner in core_cases.items():
                stats = _measure(runner, profile["repeat"], profile["warmup"])
                if impl == "pinocchio_like":
                    pin_like_stats = stats
                    if baseline is None:
                        baseline = stats["mean_ms"]
                op_name = "dynamics_order5" if impl == "cmtm_full_order5" else "dynamics_qva"
                _append_row(
                    rows,
                    profile=profile_name,
                    group="core",
                    dof=dof,
                    batch_size=batch_size,
                    op_name=op_name,
                    impl=impl,
                    stats=stats,
                    baseline_mean_ms=baseline,
                )
                base_text = "" if baseline is None else f" ratio={stats['mean_ms'] / baseline:.2f}x"
                print(f"{impl:20s} mean={stats['mean_ms']:.5f}ms{base_text}", flush=True)

            public_ops = ["torque", "torque_diff2", "mixed_total_joint_state"]
            if profile["include_public_jacobian"]:
                public_ops.extend(["torque_diff2_jacobian", "mixed_total_joint_jacobian"])

            print(f"-- public dof={dof} batch={batch_size} --", flush=True)
            for op_name in public_ops:
                op_order = 3 if op_name == "torque" else 5
                py_kots = Kots.from_json_data(model_data, order=op_order)
                rust_kots = Kots.from_json_data(model_data, order=op_order)
                op_motion = motion3 if op_order == 3 else motion5
                py_runner = _public_case(py_kots, op_motion, op_name, None)
                rust_runner = _public_case(rust_kots, op_motion, op_name, "rust")
                expected = py_runner()
                actual = rust_runner()
                max_error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
                py_stats = _measure(py_runner, profile["public_repeat"], profile["public_warmup"])
                rust_stats = _measure(rust_runner, profile["public_repeat"], profile["public_warmup"])
                _append_row(
                    rows,
                    profile=profile_name,
                    group="public",
                    dof=dof,
                    batch_size=batch_size,
                    op_name=op_name,
                    impl="robokots_python",
                    stats=py_stats,
                    baseline_mean_ms=py_stats["mean_ms"],
                    max_error=0.0,
                )
                _append_row(
                    rows,
                    profile=profile_name,
                    group="public",
                    dof=dof,
                    batch_size=batch_size,
                    op_name=op_name,
                    impl="robokots_rust",
                    stats=rust_stats,
                    baseline_mean_ms=py_stats["mean_ms"],
                    max_error=max_error,
                )
                print(
                    f"{op_name:22s} python={py_stats['mean_ms']:.3f}ms "
                    f"rust={rust_stats['mean_ms']:.3f}ms speedup={py_stats['mean_ms'] / rust_stats['mean_ms']:.2f}x "
                    f"err={max_error:.2e}",
                    flush=True,
                )

            if pin_like_stats is None:
                raise RuntimeError("pinocchio_like core benchmark did not run")

            print(f"-- selected dof={dof} batch={batch_size} --", flush=True)
            for case_name, states, case_order, case_ops in _selected_cases(kots):
                is_mixed_case = case_name == "mixed_total_joint"
                if is_mixed_case and dof not in profile["mixed_selected_dof_list"]:
                    print(f"{case_name:28s} skipped for quick profile at dof={dof}", flush=True)
                    continue
                case_motion = motion3 if case_order == 3 else motion5
                py_kots = Kots.from_json_data(model_data, order=case_order)
                rust_kots = Kots.from_json_data(model_data, order=case_order)
                py_kots.import_motions(case_motion)
                rust_kots.import_motions(case_motion)
                py_kots.dynamics(order=case_order, materialize_dict=False)
                rust_kots.dynamics(order=case_order, backend="rust", materialize_dict=False)

                selected_ops = []
                if "STATE" in case_ops:
                    selected_ops.append(
                        (
                            "STATE",
                            None,
                            py_kots.state_info_list(states),
                            rust_kots.state_info_list(states),
                        )
                    )
                if any(op in case_ops for op in ("J", "JM8", "JTM4")):
                    jacob = py_kots.jacobian(states)
                    rust_jacob = rust_kots.jacobian(states)
                    batch_shape = jacob.shape[:-2]
                    rhs_jm8 = rng.standard_normal(batch_shape + (jacob.shape[-1], 8))
                    rhs_jtm4 = rng.standard_normal(batch_shape + (jacob.shape[-2], 4))
                    if "J" in case_ops:
                        selected_ops.append(("J", None, jacob, rust_jacob))
                    if "JM8" in case_ops:
                        selected_ops.append(
                            (
                                "JM8",
                                rhs_jm8,
                                py_kots.jacobian_mul(states, rhs_jm8),
                                rust_kots.jacobian_mul(states, rhs_jm8),
                            )
                        )
                    if "JTM4" in case_ops:
                        selected_ops.append(
                            (
                                "JTM4",
                                rhs_jtm4,
                                py_kots.jacobian_transpose_mul(states, rhs_jtm4),
                                rust_kots.jacobian_transpose_mul(states, rhs_jtm4),
                            )
                        )
                for selected_op, rhs, expected, actual in selected_ops:
                    op_name = f"{case_name}_{selected_op}"
                    max_error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
                    repeat = profile["mixed_selected_repeat"] if is_mixed_case else profile["selected_repeat"]
                    warmup = profile["mixed_selected_warmup"] if is_mixed_case else profile["selected_warmup"]
                    py_stats = _measure(
                        _selected_case_runner(py_kots, states, selected_op, rhs),
                        repeat,
                        warmup,
                    )
                    rust_stats = _measure(
                        _selected_case_runner(rust_kots, states, selected_op, rhs),
                        repeat,
                        warmup,
                    )
                    _append_row(
                        rows,
                        profile=profile_name,
                        group="selected",
                        dof=dof,
                        batch_size=batch_size,
                        op_name=op_name,
                        impl="robokots_python",
                        stats=py_stats,
                        baseline_mean_ms=py_stats["mean_ms"],
                        max_error=0.0,
                    )
                    _append_row(
                        rows,
                        profile=profile_name,
                        group="selected",
                        dof=dof,
                        batch_size=batch_size,
                        op_name=op_name,
                        impl="robokots_rust",
                        stats=rust_stats,
                        baseline_mean_ms=py_stats["mean_ms"],
                        max_error=max_error,
                    )
                    print(
                        f"{op_name:28s} python={py_stats['mean_ms']:.3f}ms "
                        f"rust={rust_stats['mean_ms']:.3f}ms speedup={py_stats['mean_ms'] / rust_stats['mean_ms']:.2f}x "
                        f"err={max_error:.2e}",
                        flush=True,
                    )

    write_csv(csv_path, rows)
    metadata_path = csv_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "profile": profile_name,
                "seed": seed,
                "model_kind": model_kind,
                "profile_config": profile,
                "row_count": len(rows),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote csv : {csv_path}", flush=True)
    print(f"wrote meta: {metadata_path}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed RoboKots Rust benchmark comparisons.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="quick")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-kind", choices=["humanoid", "serial"], default="humanoid")
    args = parser.parse_args()
    run_fixed_suite(args.profile, args.csv, args.seed, args.model_kind)


if __name__ == "__main__":
    main()
