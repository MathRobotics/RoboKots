from __future__ import annotations

from pathlib import Path

import numpy as np

from robokots.core.state import StateType
from robokots.kots import Kots

from .common import (
    ROBOKOTS_OPS,
    build_model,
    format_time,
    leaf_link_names,
    measure,
    robokots_operation_call,
    robokots_operation_states,
    robokots_operation_value,
    robokots_prepare_state,
    robokots_state_types,
    select_unit,
    write_csv,
)


DEFAULT_CSV_PATH = Path(__file__).resolve().with_name("robokots_baseline_results.csv")

# Developer baseline settings. Pinocchio is intentionally not needed here.
CONFIG = {
    "dof_list": [16, 32, 64],
    "order": 5,
    "batch_sizes": [4, 16],
    "repeat": 2,
    "warmup": 1,
    "seed": 0,
    "ops": ["kinematics", "dynamics", "kinematics_matvec", "dynamics_matvec"],
    "validate": True,
    "rtol": 1e-9,
    "atol": 1e-9,
    "target_count": 4,
    "model_kind": "humanoid",
    "csv_path": DEFAULT_CSV_PATH,
}


def _build_context(model_data: dict, order: int, motions: np.ndarray) -> tuple[Kots, list[Kots]]:
    batch_kots = Kots.from_json_data(model_data, order=order)
    batch_kots.import_motions(motions)

    serial_kots = []
    for motion in motions:
        kots = Kots.from_json_data(model_data, order=order)
        kots.import_motions(motion)
        serial_kots.append(kots)
    return batch_kots, serial_kots


def _serial_call(
    serial_kots: list[Kots],
    op_name: str,
    states,
    vecs: np.ndarray | None,
    order: int,
):
    values = []
    for i, kots in enumerate(serial_kots):
        vec = None if vecs is None else vecs[i]
        values.append(robokots_operation_call(kots, op_name, states, vec, order))
    return values


def _serial_value(
    serial_kots: list[Kots],
    op_name: str,
    states,
    vecs: np.ndarray | None,
    order: int,
) -> np.ndarray:
    values = []
    for i, kots in enumerate(serial_kots):
        vec = None if vecs is None else vecs[i]
        values.append(robokots_operation_value(kots, op_name, states, vec, order))
    return np.stack(values)


def _validate_batch_result(
    batch_kots: Kots,
    serial_kots: list[Kots],
    op_name: str,
    states,
    vecs: np.ndarray | None,
    order: int,
    rtol: float,
    atol: float,
) -> None:
    batch_value = robokots_operation_value(batch_kots, op_name, states, vecs, order)
    serial_value = _serial_value(serial_kots, op_name, states, vecs, order)
    np.testing.assert_allclose(batch_value, serial_value, rtol=rtol, atol=atol)


def _print_result(
    op_name: str,
    dof: int,
    batch_size: int,
    batch_stats: dict[str, float],
    serial_stats: dict[str, float],
) -> None:
    unit = select_unit(
        [
            batch_stats["mean_ms"],
            batch_stats["std_ms"],
            serial_stats["mean_ms"],
            serial_stats["std_ms"],
        ]
    )
    speedup = serial_stats["mean_ms"] / batch_stats["mean_ms"] if batch_stats["mean_ms"] > 0 else float("inf")
    print(f"{op_name:20s} dof={dof:4d} batch={batch_size:5d}", flush=True)
    print(
        "  batch   "
        f"mean={format_time(batch_stats['mean_ms'], unit)} "
        f"std={format_time(batch_stats['std_ms'], unit)} "
        f"min={format_time(batch_stats['min_ms'], unit)}",
        flush=True,
    )
    print(
        "  serial  "
        f"mean={format_time(serial_stats['mean_ms'], unit)} "
        f"std={format_time(serial_stats['std_ms'], unit)} "
        f"min={format_time(serial_stats['min_ms'], unit)} "
        f"speedup={speedup:8.2f}",
        flush=True,
    )


def _run_one(
    dof: int,
    model_kind: str,
    order: int,
    op_name: str,
    batch_size: int,
    repeat: int,
    warmup: int,
    target_count: int,
    rng: np.random.Generator,
    validate: bool,
    rtol: float,
    atol: float,
) -> dict[str, object]:
    model_data = build_model(dof, model_kind)
    template = Kots.from_json_data(model_data, order=order)
    kinematics_states, dynamics_states, target_names = robokots_state_types(template, target_count)
    states = robokots_operation_states(op_name, kinematics_states, dynamics_states)
    required_order = StateType.max_time_order(states)
    if required_order > order:
        raise ValueError(f"{op_name} requires order {required_order}, but order is {order}")

    motions = rng.standard_normal((batch_size, template.dof() * order))
    vecs = None
    if op_name.endswith("matvec"):
        vecs = rng.standard_normal((batch_size, template.dof() * required_order))

    batch_kots, serial_kots = _build_context(model_data, order, motions)
    if op_name.endswith("jacobian") or op_name.endswith("matvec"):
        robokots_prepare_state(batch_kots, op_name, order)
        for kots in serial_kots:
            robokots_prepare_state(kots, op_name, order)

    if validate:
        _validate_batch_result(batch_kots, serial_kots, op_name, states, vecs, required_order, rtol, atol)

    batch_stats = measure(
        lambda: robokots_operation_call(batch_kots, op_name, states, vecs, required_order),
        repeats=repeat,
        warmup=warmup,
    )
    serial_stats = measure(
        lambda: _serial_call(serial_kots, op_name, states, vecs, required_order),
        repeats=repeat,
        warmup=warmup,
    )
    _print_result(op_name, template.dof(), batch_size, batch_stats, serial_stats)

    return {
        "op": op_name,
        "model_kind": model_kind,
        "dof": template.dof(),
        "link_num": template.robot_.link_num,
        "joint_num": template.robot_.joint_num,
        "order": order,
        "batch_size": batch_size,
        "target_count": len(target_names),
        "target_names": " ".join(target_names),
        "required_order": required_order,
        "batch_mean_ms": batch_stats["mean_ms"],
        "batch_std_ms": batch_stats["std_ms"],
        "batch_min_ms": batch_stats["min_ms"],
        "serial_mean_ms": serial_stats["mean_ms"],
        "serial_std_ms": serial_stats["std_ms"],
        "serial_min_ms": serial_stats["min_ms"],
        "speedup": serial_stats["mean_ms"] / batch_stats["mean_ms"] if batch_stats["mean_ms"] > 0 else float("inf"),
    }


def _validate_config(config: dict) -> None:
    if int(config["order"]) < 1:
        raise ValueError("CONFIG['order'] must be >= 1")
    if int(config["repeat"]) < 1:
        raise ValueError("CONFIG['repeat'] must be >= 1")
    if int(config["warmup"]) < 0:
        raise ValueError("CONFIG['warmup'] must be >= 0")
    if int(config["target_count"]) < 1:
        raise ValueError("CONFIG['target_count'] must be >= 1")
    invalid_ops = [name for name in config["ops"] if name not in ROBOKOTS_OPS]
    if invalid_ops:
        raise ValueError(f"Invalid op(s): {invalid_ops}")


def main() -> None:
    _validate_config(CONFIG)
    dof_list = [int(dof) for dof in CONFIG["dof_list"]]
    order = int(CONFIG["order"])
    batch_sizes = [int(size) for size in CONFIG["batch_sizes"]]
    repeat = int(CONFIG["repeat"])
    warmup = int(CONFIG["warmup"])
    selected_ops = [str(name) for name in CONFIG["ops"]]
    validate = bool(CONFIG.get("validate", True))
    rtol = float(CONFIG.get("rtol", 1e-9))
    atol = float(CONFIG.get("atol", 1e-9))
    target_count = int(CONFIG["target_count"])
    model_kind = str(CONFIG.get("model_kind", "humanoid"))
    csv_path = Path(CONFIG.get("csv_path", DEFAULT_CSV_PATH)).resolve()
    rng = np.random.default_rng(int(CONFIG["seed"]))

    print("=== RoboKots Developer Baseline ===", flush=True)
    print(f"model_kind : {model_kind}", flush=True)
    print(f"dof_list   : {dof_list}", flush=True)
    print(f"order      : {order}", flush=True)
    print(f"ops        : {', '.join(selected_ops)}", flush=True)
    print(f"batch_sizes: {batch_sizes}", flush=True)
    print(f"target_cnt : {target_count}", flush=True)
    print(f"warmup     : {warmup}", flush=True)
    print(f"repeat     : {repeat}", flush=True)
    print(f"validate   : {validate}", flush=True)
    print(f"csv_path   : {csv_path}", flush=True)
    print(flush=True)

    rows = []
    for dof in dof_list:
        template = Kots.from_json_data(build_model(dof, model_kind), order=order)
        targets = leaf_link_names(template)[-min(len(leaf_link_names(template)), target_count) :]
        print(
            f"--- dof={template.dof()} links={template.robot_.link_num} "
            f"joints={template.robot_.joint_num} targets={targets} ---",
            flush=True,
        )
        for op_name in selected_ops:
            for batch_size in batch_sizes:
                rows.append(
                    _run_one(
                        dof=dof,
                        model_kind=model_kind,
                        order=order,
                        op_name=op_name,
                        batch_size=batch_size,
                        repeat=repeat,
                        warmup=warmup,
                        target_count=target_count,
                        rng=rng,
                        validate=validate,
                        rtol=rtol,
                        atol=atol,
                    )
                )
        print(flush=True)

    write_csv(csv_path, rows)
    print(f"wrote csv  : {csv_path}", flush=True)


if __name__ == "__main__":
    main()
