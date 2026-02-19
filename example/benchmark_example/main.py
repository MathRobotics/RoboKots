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
    "jacobian_analytic",
    "jacobian_numerical",
    "update_cached",
    "update_recompute",
)

# Edit here for your benchmark setting.
CONFIG = {
    "model": DEFAULT_MODEL,
    "order": 3,
    "repeat": 200,
    "repeat_numerical": 10,
    "warmup": 5,
    "seed": 0,
    "ops": list(OPS),
}


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
    print(
        f"{name:20s} n={repeats:4d} "
        f"mean={stats['mean_ms']:9.3f}ms "
        f"std={stats['std_ms']:8.3f}ms "
        f"p50={stats['p50_ms']:8.3f}ms "
        f"p95={stats['p95_ms']:8.3f}ms "
        f"min={stats['min_ms']:8.3f}ms",
    )


def main() -> None:
    model_path = Path(CONFIG["model"]).resolve()
    order = int(CONFIG["order"])
    repeat = int(CONFIG["repeat"])
    repeat_numerical = int(CONFIG["repeat_numerical"])
    warmup = int(CONFIG["warmup"])
    seed = int(CONFIG["seed"])
    selected_ops = list(CONFIG["ops"])

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if order < 1:
        raise ValueError("CONFIG['order'] must be >= 1")
    if repeat < 1 or repeat_numerical < 1:
        raise ValueError("CONFIG['repeat'] and CONFIG['repeat_numerical'] must be >= 1")
    if warmup < 0:
        raise ValueError("CONFIG['warmup'] must be >= 0")
    invalid_ops = [name for name in selected_ops if name not in OPS]
    if invalid_ops:
        raise ValueError(f"Invalid op(s) in CONFIG['ops']: {invalid_ops}")

    kots = Kots.from_json_file(str(model_path), order=order)
    rng = np.random.default_rng(seed)
    base_motion = rng.standard_normal(kots.dof() * order)
    kots.import_motions(base_motion)

    end_link = kots.link_name_list()[-1]
    st_pos = StateType("link", end_link, "pos")

    print("=== RoboKots Benchmark ===")
    print(f"model      : {model_path}")
    print(f"order      : {order}")
    print(f"dof        : {kots.dof()}")
    print(f"ops        : {', '.join(selected_ops)}")
    print(f"warmup     : {warmup}")
    print(f"repeat     : {repeat}")
    print(f"repeat_num : {repeat_numerical}")
    print()

    counter = {"i": 0}

    def op_kinematics() -> None:
        kots.kinematics(order=order)

    def op_dynamics() -> None:
        kots.dynamics(order=order)

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
        "jacobian_analytic": op_jacobian_analytic,
        "jacobian_numerical": op_jacobian_numerical,
        "update_cached": op_update_cached,
        "update_recompute": op_update_recompute,
    }

    if "update_cached" in selected_ops:
        kots.update_state_dict(order=order, is_dynamics=False)

    for op_name in selected_ops:
        repeats = repeat_numerical if op_name == "jacobian_numerical" else repeat
        stats = _measure(op_map[op_name], repeats=repeats, warmup=warmup)
        _print_result(op_name, repeats=repeats, stats=stats)


if __name__ == "__main__":
    main()
