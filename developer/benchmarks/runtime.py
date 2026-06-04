from __future__ import annotations

import time
import platform
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from robokots.core.state import StateType
from robokots.kots import Kots


DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "examples" / "model" / "sample_robot.json"

OPS = (
    "kinematics",
    "kinematics_jax",
    "dynamics",
    "link_diff_numerical",
    "jacobian_analytic",
    "jacobian_numerical",
    "jacobian_mul_vector",
    "jacobian_mul_matrix",
    "jacobian_transpose_mul_vector",
    "jacobian_transpose_mul_matrix",
    "dynamics_jacobian_analytic",
    "dynamics_jacobian_numerical",
    "update_state_cached",
    "update_state_recompute",
    "update_state_dynamics_cached",
    "update_state_dynamics_recompute",
    "update_cached",
    "update_recompute",
    "update_dynamics_cached",
    "update_dynamics_recompute",
)

# Edit here for your benchmark setting.
CONFIG = {
    "model": DEFAULT_MODEL,
    "order": 5,
    "repeat": 200,
    "repeat_numerical": 10,
    "warmup": 5,
    "seed": 0,
    "rhs_cols": 8,
    "ops": list(OPS),
    # Baseline mean values [ms] for this repository/config.
    # Update these after intentional performance changes.
    "baseline_mean_ms": {
        "kinematics": 0.586,
        "kinematics_jax": 17.045,
        "dynamics": 1.550,
        "link_diff_numerical": 1.861,
        "jacobian_analytic": 1.626,
        "jacobian_numerical": 22.343,
        "dynamics_jacobian_analytic": 8.524,
        "dynamics_jacobian_numerical": 62.142,
        "update_state_cached": 0.000543,
        "update_state_recompute": 0.522,
        "update_state_dynamics_cached": 0.000453,
        "update_state_dynamics_recompute": 1.509,
        "update_cached": 0.000676,
        "update_recompute": 0.577,
        "update_dynamics_cached": 0.000561,
        "update_dynamics_recompute": 1.634,
    },
    # Use a high-order state by default so numerical Jacobian timing reflects
    # the lifted motion dimension, not only a first-order position query.
    "jacobian_data_type": "snap",
    "dynamics_jacobian_data_type": "torque_diff2",
    "dynamics_jacobian_owner_type": "joint",
    "dynamics_jacobian_owner_name": None,
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


def _version_or_missing(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except Exception:
        return "not installed"
    return str(getattr(module, "__version__", "unknown"))


def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    processor = platform.processor()
    if processor:
        return processor
    return "unknown"


def _print_environment() -> None:
    print(f"python     : {sys.version.split()[0]} ({sys.executable})")
    print(f"platform   : {platform.platform()}")
    print(f"cpu        : {_cpu_name()}")
    print(f"numpy      : {np.__version__}")
    print(f"jax        : {_version_or_missing('jax')}")
    print(f"jaxlib     : {_version_or_missing('jaxlib')}")


def main() -> None:
    model_path = Path(CONFIG["model"]).resolve()
    order = int(CONFIG["order"])
    repeat = int(CONFIG["repeat"])
    repeat_numerical = int(CONFIG["repeat_numerical"])
    warmup = int(CONFIG["warmup"])
    seed = int(CONFIG["seed"])
    selected_ops = list(CONFIG["ops"])
    baseline_mean_ms = {str(k): float(v) for k, v in dict(CONFIG.get("baseline_mean_ms", {})).items()}
    jacobian_data_type = str(CONFIG.get("jacobian_data_type", "snap"))
    dynamics_jacobian_data_type = str(CONFIG.get("dynamics_jacobian_data_type", "force_diff2"))
    dynamics_jacobian_owner_type = str(CONFIG.get("dynamics_jacobian_owner_type", "joint"))
    link_diff_data_type = str(CONFIG.get("link_diff_data_type", "frame"))
    link_diff_link_count = int(CONFIG.get("link_diff_link_count", 3))
    rhs_cols = int(CONFIG.get("rhs_cols", 8))

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
    end_joint = kots.joint_name_list()[-1]
    st_jacobian = StateType("link", end_link, jacobian_data_type)
    kots.kinematics(order=order)
    jacobian_output_dim = kots.jacobian(st_jacobian, numerical=False).shape[-2]
    jacobian_rhs_vector = rng.standard_normal(kots.dof() * st_jacobian.time_order)
    jacobian_rhs_matrix = rng.standard_normal((kots.dof() * st_jacobian.time_order, rhs_cols))
    jacobian_transpose_rhs_vector = rng.standard_normal(jacobian_output_dim)
    jacobian_transpose_rhs_matrix = rng.standard_normal((jacobian_output_dim, rhs_cols))
    dynamics_jacobian_owner_name = CONFIG.get("dynamics_jacobian_owner_name")
    if dynamics_jacobian_owner_name is None:
        dynamics_jacobian_owner_name = end_joint if dynamics_jacobian_owner_type == "joint" else end_link
    dynamics_jacobian_owner_name = str(dynamics_jacobian_owner_name)
    st_dynamics_jacobian = StateType(
        dynamics_jacobian_owner_type,
        dynamics_jacobian_owner_name,
        dynamics_jacobian_data_type,
    )
    link_names = kots.link_name_list()
    link_diff_targets = link_names[-min(len(link_names), link_diff_link_count) :]
    link_diff_direction = rng.standard_normal(kots.dof())

    print("=== RoboKots Benchmark ===")
    _print_environment()
    print(f"model      : {model_path}")
    print(f"order      : {order}")
    print(f"dof        : {kots.dof()}")
    print(f"ops        : {', '.join(selected_ops)}")
    print(f"warmup     : {warmup}")
    print(f"repeat     : {repeat}")
    print(f"repeat_num : {repeat_numerical}")
    print(f"rhs_cols   : {rhs_cols}")
    print(f"jacobian   : state={end_link}_link_{jacobian_data_type} order={st_jacobian.time_order}")
    print(
        "dyn_jacob  : "
        f"state={dynamics_jacobian_owner_name}_{dynamics_jacobian_owner_type}_{dynamics_jacobian_data_type} "
        f"order={st_dynamics_jacobian.time_order}"
    )
    if "link_diff_numerical" in selected_ops:
        print(f"link_diff  : type={link_diff_data_type} targets={link_diff_targets}")
    print()

    counter = {"i": 0}

    def op_kinematics() -> None:
        kots.kinematics(order=order)

    def op_kinematics_jax() -> None:
        kots.kinematics(order=order, backend="jax")

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
        _ = kots.jacobian(st_jacobian, numerical=False)

    def op_jacobian_numerical() -> None:
        _ = kots.jacobian(st_jacobian, numerical=True)

    def op_jacobian_mul_vector() -> None:
        kots.kinematics(order=order)
        _ = kots.jacobian_mul(st_jacobian, jacobian_rhs_vector, numerical=False)

    def op_jacobian_mul_matrix() -> None:
        kots.kinematics(order=order)
        _ = kots.jacobian_mul(st_jacobian, jacobian_rhs_matrix, numerical=False)

    def op_jacobian_transpose_mul_vector() -> None:
        kots.kinematics(order=order)
        _ = kots.jacobian_transpose_mul(st_jacobian, jacobian_transpose_rhs_vector, numerical=False)

    def op_jacobian_transpose_mul_matrix() -> None:
        kots.kinematics(order=order)
        _ = kots.jacobian_transpose_mul(st_jacobian, jacobian_transpose_rhs_matrix, numerical=False)

    def op_dynamics_jacobian_analytic() -> None:
        kots.dynamics(order=order)
        _ = kots.jacobian(st_dynamics_jacobian, numerical=False)

    def op_dynamics_jacobian_numerical() -> None:
        _ = kots.jacobian(st_dynamics_jacobian, numerical=True)

    def op_update_state_cached() -> None:
        _ = kots.update_state(order=order, is_dynamics=False)

    def op_update_state_recompute() -> None:
        counter["i"] += 1
        motion = base_motion.copy()
        motion[0] += 1e-6 * counter["i"]
        kots.import_motions(motion)
        _ = kots.update_state(order=order, is_dynamics=False)

    def op_update_state_dynamics_cached() -> None:
        _ = kots.update_state(order=order, is_dynamics=True)

    def op_update_state_dynamics_recompute() -> None:
        counter["i"] += 1
        motion = base_motion.copy()
        motion[0] += 1e-6 * counter["i"]
        kots.import_motions(motion)
        _ = kots.update_state(order=order, is_dynamics=True)

    def op_update_cached() -> None:
        _ = kots.update_state_dict(order=order, is_dynamics=False)

    def op_update_recompute() -> None:
        counter["i"] += 1
        motion = base_motion.copy()
        motion[0] += 1e-6 * counter["i"]
        kots.import_motions(motion)
        _ = kots.update_state_dict(order=order, is_dynamics=False)

    def op_update_dynamics_cached() -> None:
        _ = kots.update_state_dict(order=order, is_dynamics=True)

    def op_update_dynamics_recompute() -> None:
        counter["i"] += 1
        motion = base_motion.copy()
        motion[0] += 1e-6 * counter["i"]
        kots.import_motions(motion)
        _ = kots.update_state_dict(order=order, is_dynamics=True)

    op_map: dict[str, Callable[[], None]] = {
        "kinematics": op_kinematics,
        "kinematics_jax": op_kinematics_jax,
        "dynamics": op_dynamics,
        "link_diff_numerical": op_link_diff_numerical,
        "jacobian_analytic": op_jacobian_analytic,
        "jacobian_numerical": op_jacobian_numerical,
        "jacobian_mul_vector": op_jacobian_mul_vector,
        "jacobian_mul_matrix": op_jacobian_mul_matrix,
        "jacobian_transpose_mul_vector": op_jacobian_transpose_mul_vector,
        "jacobian_transpose_mul_matrix": op_jacobian_transpose_mul_matrix,
        "dynamics_jacobian_analytic": op_dynamics_jacobian_analytic,
        "dynamics_jacobian_numerical": op_dynamics_jacobian_numerical,
        "update_state_cached": op_update_state_cached,
        "update_state_recompute": op_update_state_recompute,
        "update_state_dynamics_cached": op_update_state_dynamics_cached,
        "update_state_dynamics_recompute": op_update_state_dynamics_recompute,
        "update_cached": op_update_cached,
        "update_recompute": op_update_recompute,
        "update_dynamics_cached": op_update_dynamics_cached,
        "update_dynamics_recompute": op_update_dynamics_recompute,
    }

    for op_name in selected_ops:
        if op_name == "update_cached":
            kots.update_state_dict(order=order, is_dynamics=False)
        elif op_name == "update_state_cached":
            kots.update_state(order=order, is_dynamics=False)
        elif op_name == "update_dynamics_cached":
            kots.update_state_dict(order=order, is_dynamics=True)
        elif op_name == "update_state_dynamics_cached":
            kots.update_state(order=order, is_dynamics=True)

        repeats = repeat_numerical if op_name in {
            "jacobian_numerical",
            "dynamics_jacobian_numerical",
            "link_diff_numerical",
        } else repeat
        stats = _measure(op_map[op_name], repeats=repeats, warmup=warmup)
        _print_result(op_name, repeats=repeats, stats=stats)
        _print_baseline_compare(op_name, stats=stats, baseline_mean_ms=baseline_mean_ms)


if __name__ == "__main__":
    main()
