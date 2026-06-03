from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Callable

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import robokots.core.models.kinematics.kinematics_jax as kinematics_jax

from robokots.core.state import StateType
from robokots.kots import Kots


COMPARE_SPECS = (
    ("vel", 2, kinematics_jax.forward_kinematics_vel),
    ("acc", 3, kinematics_jax.forward_kinematics_acc),
    ("jerk", 4, kinematics_jax.forward_kinematics_jerk),
)
DEFAULT_CSV_PATH = Path(__file__).resolve().with_name("jacobian_dof_sweep_results.csv")

CONFIG = {
    "dof_list": [1, 2, 4, 8, 16, 32, 64, 128],
    "order": 5,
    "repeat": 3,
    "repeat_numerical": 1,
    "warmup": 0,
    "seed": 0,
    "compare": ["vel", "acc", "jerk"],
    "max_dof_by_compare": {
        "vel": 128,
        "acc": 128,
        "jerk": 128,
    },
    "link_offset": [1.0, 1.0, 0.0],
    "joint_axis": [0.0, 0.0, 1.0],
    "joint_limit": 1.57,
    "csv_path": DEFAULT_CSV_PATH,
}


def build_serial_arm_model(
    dof: int,
    link_offset: list[float],
    joint_axis: list[float],
    joint_limit: float,
) -> dict:
    if dof < 1:
        raise ValueError("dof must be >= 1")

    links = [
        {"id": 0, "name": "world"},
        {
            "id": 1,
            "name": "base",
            "mass": 10.0,
            "cog": [0.0, 0.0, 0.0],
            "inertia": [0.1, 0.1, 0.1, 0.0, 0.0, 0.0],
            "geometry": "base_mesh.stl",
        },
    ]
    for i in range(1, dof + 1):
        links.append(
            {
                "id": i + 1,
                "name": f"arm{i}",
                "mass": 5.0,
                "cog": [0.0, 0.0, 0.0],
                "inertia": [0.05, 0.05, 0.02, 0.0, 0.0, 0.0],
                "geometry": "arm_mesh.stl",
            }
        )

    joints = [
        {
            "id": 0,
            "name": "root",
            "type": "fix",
            "parent_link_id": 0,
            "child_link_id": 1,
            "origin": {
                "position": [0.0, 0.0, 0.0],
                "orientation": [1.0, 0.0, 0.0, 0.0],
            },
        }
    ]
    for i in range(1, dof + 1):
        joints.append(
            {
                "id": i,
                "name": f"joint{i}",
                "type": "revolute",
                "parent_link_id": i,
                "child_link_id": i + 1,
                "axis": [float(v) for v in joint_axis],
                "limits": {"min": -float(joint_limit), "max": float(joint_limit)},
                "origin": {
                    "position": [float(v) for v in link_offset],
                    "orientation": [1.0, 0.0, 0.0, 0.0],
                },
            }
        )

    return {"links": links, "joints": joints}


def _pair_metrics(lhs: np.ndarray, rhs: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(lhs) - np.asarray(rhs)
    return float(np.max(np.abs(diff))), float(np.linalg.norm(diff))


def _measure(fn: Callable[[], np.ndarray], repeats: int, warmup: int) -> dict[str, float]:
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
    }


def _time_call(fn: Callable[[], np.ndarray]) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - t0) * 1e3


def _format_elapsed(value_ms: float) -> str:
    if abs(value_ms) < 0.1:
        return f"{value_ms * 1e3:8.3f}us"
    return f"{value_ms:8.3f}ms"


def _build_autodiff_runner(kots_jax: Kots, target_link: str, data_type: str) -> tuple[Callable[[], np.ndarray], int]:
    spec_map = {name: (required_order, func) for name, required_order, func in COMPARE_SPECS}
    if data_type not in spec_map:
        raise ValueError(f"Unsupported autodiff data_type: {data_type}")

    required_order, value_func = spec_map[data_type]
    joints = kots_jax.robot_.joints
    target_id = kots_jax.robot_.link(target_link).id
    x0 = jnp.asarray(kots_jax.motion(order=required_order))
    jac_fn = jax.jit(
        jax.jacfwd(
            lambda x, _func=value_func, _joints=joints, _target_id=target_id: _func(_joints, x)[_target_id]
        )
    )

    def run() -> np.ndarray:
        jac = jac_fn(x0)
        return np.asarray(jax.block_until_ready(jac))

    return run, required_order


def _evaluate_dof(
    dof: int,
    order: int,
    seed: int,
    repeat: int,
    repeat_numerical: int,
    warmup: int,
    link_offset: list[float],
    joint_axis: list[float],
    joint_limit: float,
    selected_specs: tuple[tuple[str, int, Callable], ...],
) -> list[dict]:
    model_data = build_serial_arm_model(
        dof=dof,
        link_offset=link_offset,
        joint_axis=joint_axis,
        joint_limit=joint_limit,
    )
    kots = Kots.from_json_data(model_data, order=order)
    kots_jax = Kots.from_json_data(model_data, order=order, lib="jax")

    rng = np.random.default_rng(seed + dof)
    motion = rng.standard_normal(order * dof)
    kots.import_motions(motion)
    kots_jax.import_motions(motion)
    target_link = kots.link_name_list()[-1]

    results = []
    for data_type, _, _ in selected_specs:
        state = StateType("link", target_link, data_type)

        def analytic_run() -> np.ndarray:
            kots.kinematics(order=order)
            return np.asarray(kots.jacobian(state))

        def numerical_run() -> np.ndarray:
            return np.asarray(kots.jacobian(state, numerical=True))

        autodiff_run, autodiff_order = _build_autodiff_runner(kots_jax, target_link, data_type)

        analytic = analytic_run()
        numerical = numerical_run()
        autodiff, autodiff_first_ms = _time_call(autodiff_run)

        if analytic.shape != numerical.shape or analytic.shape != autodiff.shape:
            raise ValueError(
                f"Jacobian shape mismatch for dof={dof}, data_type={data_type}: "
                f"analytic={analytic.shape}, numerical={numerical.shape}, autodiff={autodiff.shape}"
            )

        results.append(
            {
                "data_type": data_type,
                "dof": dof,
                "target_link": target_link,
                "required_order": autodiff_order,
                "shape": analytic.shape,
                "time": {
                    "analytic": _measure(analytic_run, repeats=repeat, warmup=warmup),
                    "numerical": _measure(numerical_run, repeats=repeat_numerical, warmup=warmup),
                    "autodiff": _measure(autodiff_run, repeats=repeat, warmup=max(warmup - 1, 0)),
                    "autodiff_first_ms": float(autodiff_first_ms),
                },
                "error": {
                    "analytic_vs_numerical": _pair_metrics(analytic, numerical),
                    "analytic_vs_autodiff": _pair_metrics(analytic, autodiff),
                    "numerical_vs_autodiff": _pair_metrics(numerical, autodiff),
                },
            }
        )

    return results


def _print_result_block(data_type: str, results: list[dict], max_dof: int) -> None:
    print(f"[{data_type}] max_dof={max_dof}")
    for item in results:
        print(
            f"  dof={item['dof']:>3d} shape={item['shape'][0]}x{item['shape'][1]} "
            f"time analytic={_format_elapsed(item['time']['analytic']['mean_ms'])} "
            f"numerical={_format_elapsed(item['time']['numerical']['mean_ms'])} "
            f"autodiff={_format_elapsed(item['time']['autodiff']['mean_ms'])} "
            f"jax_first={_format_elapsed(item['time']['autodiff_first_ms'])}"
        )
        print(
            "      err "
            f"an-num max={item['error']['analytic_vs_numerical'][0]:.3e} fro={item['error']['analytic_vs_numerical'][1]:.3e} | "
            f"an-ad max={item['error']['analytic_vs_autodiff'][0]:.3e} fro={item['error']['analytic_vs_autodiff'][1]:.3e} | "
            f"num-ad max={item['error']['numerical_vs_autodiff'][0]:.3e} fro={item['error']['numerical_vs_autodiff'][1]:.3e}"
        )
    print()


def _csv_fieldnames(selected_specs: tuple[tuple[str, int, Callable], ...]) -> list[str]:
    fieldnames = ["dof"]
    for data_type, _, _ in selected_specs:
        prefix = f"{data_type}_"
        fieldnames.extend(
            [
                prefix + "available",
                prefix + "target_link",
                prefix + "required_order",
                prefix + "shape_rows",
                prefix + "shape_cols",
                prefix + "time_analytic_mean_ms",
                prefix + "time_analytic_std_ms",
                prefix + "time_analytic_min_ms",
                prefix + "time_numerical_mean_ms",
                prefix + "time_numerical_std_ms",
                prefix + "time_numerical_min_ms",
                prefix + "time_autodiff_mean_ms",
                prefix + "time_autodiff_std_ms",
                prefix + "time_autodiff_min_ms",
                prefix + "time_autodiff_first_ms",
                prefix + "err_an_num_max_abs",
                prefix + "err_an_num_fro",
                prefix + "err_an_ad_max_abs",
                prefix + "err_an_ad_fro",
                prefix + "err_num_ad_max_abs",
                prefix + "err_num_ad_fro",
            ]
        )
    return fieldnames


def _csv_row_from_results(
    dof: int,
    results_by_type: dict[str, dict],
    selected_specs: tuple[tuple[str, int, Callable], ...],
) -> dict[str, object]:
    row: dict[str, object] = {"dof": dof}
    for data_type, _, _ in selected_specs:
        prefix = f"{data_type}_"
        item = results_by_type.get(data_type)
        row[prefix + "available"] = int(item is not None)
        if item is None:
            for key in (
                "target_link",
                "required_order",
                "shape_rows",
                "shape_cols",
                "time_analytic_mean_ms",
                "time_analytic_std_ms",
                "time_analytic_min_ms",
                "time_numerical_mean_ms",
                "time_numerical_std_ms",
                "time_numerical_min_ms",
                "time_autodiff_mean_ms",
                "time_autodiff_std_ms",
                "time_autodiff_min_ms",
                "time_autodiff_first_ms",
                "err_an_num_max_abs",
                "err_an_num_fro",
                "err_an_ad_max_abs",
                "err_an_ad_fro",
                "err_num_ad_max_abs",
                "err_num_ad_fro",
            ):
                row[prefix + key] = ""
            continue

        row[prefix + "target_link"] = item["target_link"]
        row[prefix + "required_order"] = item["required_order"]
        row[prefix + "shape_rows"] = item["shape"][0]
        row[prefix + "shape_cols"] = item["shape"][1]
        row[prefix + "time_analytic_mean_ms"] = item["time"]["analytic"]["mean_ms"]
        row[prefix + "time_analytic_std_ms"] = item["time"]["analytic"]["std_ms"]
        row[prefix + "time_analytic_min_ms"] = item["time"]["analytic"]["min_ms"]
        row[prefix + "time_numerical_mean_ms"] = item["time"]["numerical"]["mean_ms"]
        row[prefix + "time_numerical_std_ms"] = item["time"]["numerical"]["std_ms"]
        row[prefix + "time_numerical_min_ms"] = item["time"]["numerical"]["min_ms"]
        row[prefix + "time_autodiff_mean_ms"] = item["time"]["autodiff"]["mean_ms"]
        row[prefix + "time_autodiff_std_ms"] = item["time"]["autodiff"]["std_ms"]
        row[prefix + "time_autodiff_min_ms"] = item["time"]["autodiff"]["min_ms"]
        row[prefix + "time_autodiff_first_ms"] = item["time"]["autodiff_first_ms"]
        row[prefix + "err_an_num_max_abs"] = item["error"]["analytic_vs_numerical"][0]
        row[prefix + "err_an_num_fro"] = item["error"]["analytic_vs_numerical"][1]
        row[prefix + "err_an_ad_max_abs"] = item["error"]["analytic_vs_autodiff"][0]
        row[prefix + "err_an_ad_fro"] = item["error"]["analytic_vs_autodiff"][1]
        row[prefix + "err_num_ad_max_abs"] = item["error"]["numerical_vs_autodiff"][0]
        row[prefix + "err_num_ad_fro"] = item["error"]["numerical_vs_autodiff"][1]
    return row


def _write_csv(
    csv_path: Path,
    dof_list: list[int],
    grouped_results: dict[str, list[dict]],
    selected_specs: tuple[tuple[str, int, Callable], ...],
) -> None:
    results_by_dof: dict[int, dict[str, dict]] = {dof: {} for dof in dof_list}
    for data_type, _, _ in selected_specs:
        for item in grouped_results[data_type]:
            results_by_dof.setdefault(item["dof"], {})[data_type] = item

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_csv_fieldnames(selected_specs))
        writer.writeheader()
        for dof in dof_list:
            writer.writerow(_csv_row_from_results(dof, results_by_dof.get(dof, {}), selected_specs))


def main() -> None:
    dof_list = [int(v) for v in CONFIG["dof_list"]]
    order = int(CONFIG["order"])
    repeat = int(CONFIG["repeat"])
    repeat_numerical = int(CONFIG["repeat_numerical"])
    warmup = int(CONFIG["warmup"])
    seed = int(CONFIG["seed"])
    selected_compare = [str(v) for v in CONFIG.get("compare", [name for name, _, _ in COMPARE_SPECS])]
    raw_max_dof_by_compare = dict(CONFIG.get("max_dof_by_compare", {}))
    max_dof_by_compare = {str(k): int(v) for k, v in raw_max_dof_by_compare.items()}
    link_offset = [float(v) for v in CONFIG.get("link_offset", [1.0, 1.0, 0.0])]
    joint_axis = [float(v) for v in CONFIG.get("joint_axis", [0.0, 0.0, 1.0])]
    joint_limit = float(CONFIG.get("joint_limit", 1.57))
    csv_path = Path(CONFIG.get("csv_path", DEFAULT_CSV_PATH)).resolve()

    if order < 2:
        raise ValueError("CONFIG['order'] must be >= 2")
    if not dof_list or any(dof < 1 for dof in dof_list):
        raise ValueError("CONFIG['dof_list'] must contain only positive integers")
    if repeat < 1 or repeat_numerical < 1:
        raise ValueError("CONFIG['repeat'] and CONFIG['repeat_numerical'] must be >= 1")
    if warmup < 0:
        raise ValueError("CONFIG['warmup'] must be >= 0")
    if any(max_dof < 1 for max_dof in max_dof_by_compare.values()):
        raise ValueError("CONFIG['max_dof_by_compare'] must contain only positive integers")

    spec_map = {name: (required_order, func) for name, required_order, func in COMPARE_SPECS}
    invalid_compare = [name for name in selected_compare if name not in spec_map]
    invalid_max_dof = [name for name in max_dof_by_compare if name not in spec_map]
    if invalid_compare:
        raise ValueError(f"Invalid compare target(s): {invalid_compare}")
    if invalid_max_dof:
        raise ValueError(f"Invalid keys in CONFIG['max_dof_by_compare']: {invalid_max_dof}")

    selected_specs = tuple(
        spec
        for spec in COMPARE_SPECS
        if spec[0] in selected_compare and spec[1] <= order
    )
    if not selected_specs:
        raise ValueError(f"No comparison target is available for order={order}")

    default_max_dof = max(dof_list)
    grouped_results = {name: [] for name, _, _ in selected_specs}
    for dof in dof_list:
        specs_for_dof = tuple(
            spec
            for spec in selected_specs
            if dof <= max_dof_by_compare.get(spec[0], default_max_dof)
        )
        if not specs_for_dof:
            continue
        for item in _evaluate_dof(
            dof=dof,
            order=order,
            seed=seed,
            repeat=repeat,
            repeat_numerical=repeat_numerical,
            warmup=warmup,
            link_offset=link_offset,
            joint_axis=joint_axis,
            joint_limit=joint_limit,
            selected_specs=specs_for_dof,
        ):
            grouped_results[item["data_type"]].append(item)

    print("=== Jacobian DOF Sweep Example ===")
    print(f"order      : {order}")
    print(f"dof_list   : {dof_list}")
    print(f"compare    : {', '.join(name for name, _, _ in selected_specs)}")
    print(f"max_dof    : {max_dof_by_compare}")
    print(f"repeat     : {repeat}")
    print(f"repeat_num : {repeat_numerical}")
    print(f"warmup     : {warmup}")
    print(f"seed       : {seed}")
    print(f"link_offset: {link_offset}")
    print(f"joint_axis : {joint_axis}")
    print(f"csv_path   : {csv_path}")
    print("target     : last link of each generated serial chain")
    print("note       : compare types are capped by max_dof_by_compare so the sweep can reach about 100 DOF.")
    print("note       : autodiff uses kinematics_jax; jax_first includes the initial JAX trace.")
    print()

    for data_type, _, _ in selected_specs:
        _print_result_block(
            data_type,
            grouped_results[data_type],
            max_dof_by_compare.get(data_type, default_max_dof),
        )

    _write_csv(csv_path, dof_list, grouped_results, selected_specs)
    print(f"csv saved  : {csv_path}")


if __name__ == "__main__":
    main()
