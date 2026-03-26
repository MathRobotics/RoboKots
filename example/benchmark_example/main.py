from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np

from robokots.core.state import StateType
from robokots.kots import Kots


DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "model" / "sample_robot.json"

OPS = (
    "kinematics",
    "dynamics",
    "link_diff_numerical",
    "jacobian_analytic",
    "jacobian_numerical",
    "update_cached",
    "update_recompute",
)

# Edit here for your benchmark setting.
CONFIG = {
    "model": DEFAULT_MODEL,
    "order": 5,
    "repeat": 200,
    "repeat_numerical": 10,
    "warmup": 5,
    "seed": 0,
    "ops": list(OPS),
    # Baseline mean values [ms] measured with mathrobo==0.0.1
    # under this default CONFIG (order=5, repeat=200, repeat_numerical=10, seed=0).
    "baseline_mean_ms": {
        "kinematics": 8.438,
        "dynamics": 12.970,
        "link_diff_numerical": 23.127,
        "jacobian_analytic": 10.713,
        "jacobian_numerical": 14.059,
        "update_cached": 0.059,
        "update_recompute": 8.395,
    },
    # link_diff benchmark settings
    "link_diff_data_type": "frame",
    "link_diff_link_count": 3,
}


def _select_unit(values_ms: list[float], threshold_ms: float = 0.1) -> str:
    finite_values = [abs(v) for v in values_ms if np.isfinite(v)]
    if not finite_values:
        return "ms"
    return "us" if max(finite_values) < threshold_ms else "ms"


def _format_time(value_ms: float, unit: str) -> str:
    if unit == "us":
        return f"{value_ms * 1e3:9.3f}us"
    return f"{value_ms:9.3f}ms"


def _measure(fn: Callable[[], None], repeats: int, warmup: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()

    samples = np.zeros(repeats, dtype=float)
    for i in range(repeats):
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


def _print_result(name: str, repeats: int, stats: dict[str, float]) -> None:
    unit = _select_unit(
        [
            stats["mean_ms"],
            stats["std_ms"],
            stats["p50_ms"],
            stats["p95_ms"],
            stats["min_ms"],
        ]
    )
    print(f"{name:20s} n={repeats:4d}")
    print(
        "  current  "
        f"mean={_format_time(stats['mean_ms'], unit)} "
        f"std={_format_time(stats['std_ms'], unit)} "
        f"p50={_format_time(stats['p50_ms'], unit)} "
        f"p95={_format_time(stats['p95_ms'], unit)} "
        f"min={_format_time(stats['min_ms'], unit)}",
    )


def _print_baseline_compare(name: str, stats: dict[str, float], baseline_mean_ms: dict[str, float]) -> None:
    if name not in baseline_mean_ms:
        print("  baseline mean=      (not set)")
        return
    base = float(baseline_mean_ms[name])
    if base <= 0:
        print("  baseline mean=      (invalid <= 0)")
        return
    current = stats["mean_ms"]
    delta = (current - base) / base * 100.0
    speed_ratio = base / current if current > 0 else float("inf")
    unit = _select_unit([base, current])
    print(
        "  baseline "
        f"mean={_format_time(base, unit)} "
        f"delta={delta:+8.2f}% "
        f"speed_ratio(base/current)={speed_ratio:8.3f}",
    )


def main() -> None:
    model_path = Path(CONFIG["model"]).resolve()
    order = int(CONFIG["order"])
    repeat = int(CONFIG["repeat"])
    repeat_numerical = int(CONFIG["repeat_numerical"])
    warmup = int(CONFIG["warmup"])
    seed = int(CONFIG["seed"])
    selected_ops = list(CONFIG["ops"])
    baseline_mean_ms = {str(k): float(v) for k, v in dict(CONFIG.get("baseline_mean_ms", {})).items()}
    link_diff_data_type = str(CONFIG.get("link_diff_data_type", "frame"))
    link_diff_link_count = int(CONFIG.get("link_diff_link_count", 3))

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if order < 1:
        raise ValueError("CONFIG['order'] must be >= 1")
    if repeat < 1 or repeat_numerical < 1:
        raise ValueError("CONFIG['repeat'] and CONFIG['repeat_numerical'] must be >= 1")
    if warmup < 0:
        raise ValueError("CONFIG['warmup'] must be >= 0")
    if link_diff_link_count < 1:
        raise ValueError("CONFIG['link_diff_link_count'] must be >= 1")
    invalid_ops = [name for name in selected_ops if name not in OPS]
    if invalid_ops:
        raise ValueError(f"Invalid op(s) in CONFIG['ops']: {invalid_ops}")

    kots = Kots.from_json_file(str(model_path), order=order)
    rng = np.random.default_rng(seed)
    base_motion = rng.standard_normal(kots.dof() * order)
    kots.import_motions(base_motion)

    end_link = kots.link_name_list()[-1]
    st_pos = StateType("link", end_link, "pos")
    link_names = kots.link_name_list()
    link_diff_targets = link_names[-min(len(link_names), link_diff_link_count) :]
    link_diff_direction = rng.standard_normal(kots.dof())

    print("=== RoboKots Benchmark ===")
    print(f"model      : {model_path}")
    print(f"order      : {order}")
    print(f"dof        : {kots.dof()}")
    print(f"ops        : {', '.join(selected_ops)}")
    print(f"warmup     : {warmup}")
    print(f"repeat     : {repeat}")
    print(f"repeat_num : {repeat_numerical}")
    if "link_diff_numerical" in selected_ops:
        print(f"link_diff  : type={link_diff_data_type} targets={link_diff_targets}")
    print()

    counter = {"i": 0}

    def op_kinematics() -> None:
        kots.kinematics(order=order)

    def op_dynamics() -> None:
        kots.dynamics(order=order)

    def op_link_diff_numerical() -> None:
        _ = kots.link_diff_kinematics_numerical(
            link_diff_targets,
            data_type=link_diff_data_type,
            order=order,
            update_method="poly",
            update_direction=link_diff_direction,
        )

    def op_jacobian_analytic() -> None:
        kots.kinematics(order=order)
        _ = kots.jacobian(st_pos, numerical=False)

    def op_jacobian_numerical() -> None:
        _ = kots.jacobian(st_pos, numerical=True)

    def op_update_cached() -> None:
        _ = kots.update_state_dict(order=order, is_dynamics=False)

    def op_update_recompute() -> None:
        counter["i"] += 1
        motion = base_motion.copy()
        motion[0] += 1e-6 * counter["i"]
        kots.import_motions(motion)
        _ = kots.update_state_dict(order=order, is_dynamics=False)

    op_map: dict[str, Callable[[], None]] = {
        "kinematics": op_kinematics,
        "dynamics": op_dynamics,
        "link_diff_numerical": op_link_diff_numerical,
        "jacobian_analytic": op_jacobian_analytic,
        "jacobian_numerical": op_jacobian_numerical,
        "update_cached": op_update_cached,
        "update_recompute": op_update_recompute,
    }

    if "update_cached" in selected_ops:
        kots.update_state_dict(order=order, is_dynamics=False)

    for op_name in selected_ops:
        repeats = repeat_numerical if op_name in {"jacobian_numerical", "link_diff_numerical"} else repeat
        stats = _measure(op_map[op_name], repeats=repeats, warmup=warmup)
        _print_result(op_name, repeats=repeats, stats=stats)
        _print_baseline_compare(op_name, stats=stats, baseline_mean_ms=baseline_mean_ms)


if __name__ == "__main__":
    main()
