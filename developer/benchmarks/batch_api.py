from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np

from robokots.core.state import StateType
from robokots.kots import Kots


DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "examples" / "model" / "sample_robot.json"

OPS = (
    "kinematics",
    "dynamics",
    "kinematics_jacobian",
    "dynamics_jacobian",
    "kinematics_matvec",
    "dynamics_matvec",
)

# Edit here for your benchmark setting.
CONFIG = {
    "model": DEFAULT_MODEL,
    "order": 5,
    "batch_sizes": [16, 64],
    "repeat": 3,
    "warmup": 1,
    "seed": 0,
    "ops": list(OPS),
    "validate": True,
    "rtol": 1e-10,
    "atol": 1e-10,
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


def _print_result(op_name: str, batch_size: int, batch_stats: dict[str, float], serial_stats: dict[str, float]) -> None:
    unit = _select_unit(
        [
            batch_stats["mean_ms"],
            batch_stats["std_ms"],
            serial_stats["mean_ms"],
            serial_stats["std_ms"],
        ]
    )
    speedup = serial_stats["mean_ms"] / batch_stats["mean_ms"] if batch_stats["mean_ms"] > 0 else float("inf")
    print(f"{op_name:20s} batch={batch_size:5d}", flush=True)
    print(
        "  batch   "
        f"mean={_format_time(batch_stats['mean_ms'], unit)} "
        f"std={_format_time(batch_stats['std_ms'], unit)} "
        f"min={_format_time(batch_stats['min_ms'], unit)}",
        flush=True,
    )
    print(
        "  serial  "
        f"mean={_format_time(serial_stats['mean_ms'], unit)} "
        f"std={_format_time(serial_stats['std_ms'], unit)} "
        f"min={_format_time(serial_stats['min_ms'], unit)} "
        f"speedup={speedup:8.2f}",
        flush=True,
    )


def _validate_config(config: dict) -> None:
    model_path = Path(config["model"]).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if int(config["order"]) < 1:
        raise ValueError("CONFIG['order'] must be >= 1")
    if int(config["repeat"]) < 1:
        raise ValueError("CONFIG['repeat'] must be >= 1")
    if int(config["warmup"]) < 0:
        raise ValueError("CONFIG['warmup'] must be >= 0")
    if not config["batch_sizes"]:
        raise ValueError("CONFIG['batch_sizes'] must not be empty")
    invalid_ops = [name for name in config["ops"] if name not in OPS]
    if invalid_ops:
        raise ValueError(f"Invalid op(s) in CONFIG['ops']: {invalid_ops}")


def _state_types(kots: Kots) -> tuple[list[StateType], list[StateType]]:
    end_link = kots.link_name_list()[-1]
    end_joint = kots.joint_name_list()[-1]
    kinematics_states = [
        StateType("link", end_link, "frame"),
        StateType("link", end_link, "vel"),
        StateType("link", end_link, "acc"),
        StateType("link", end_link, "snap"),
    ]
    dynamics_states = [
        StateType("link", end_link, "momentum_diff3"),
        StateType("link", end_link, "momentum_diff3", "world"),
        StateType("link", end_link, "force_diff2"),
        StateType("joint", end_joint, "momentum_diff3"),
        StateType("joint", end_joint, "momentum_diff3", "world"),
        StateType("joint", end_joint, "force_diff2"),
        StateType("joint", end_joint, "torque_diff2"),
    ]
    return kinematics_states, dynamics_states


def _operation_states(op_name: str, kinematics_states: list[StateType], dynamics_states: list[StateType]) -> list[StateType]:
    return dynamics_states if op_name.startswith("dynamics") else kinematics_states


def _operation_is_dynamics(op_name: str) -> bool:
    return op_name.startswith("dynamics")


def _operation_needs_state(op_name: str) -> bool:
    return op_name.endswith("jacobian") or op_name.endswith("matvec")


def _prepare_state(kots: Kots, op_name: str, order: int) -> None:
    if _operation_is_dynamics(op_name):
        kots.dynamics(order=order)
    else:
        kots.kinematics(order=order)


def _operation_call(kots: Kots, op_name: str, states: list[StateType], vec: np.ndarray | None, order: int):
    if op_name == "kinematics":
        return kots.kinematics(order=order)
    if op_name == "dynamics":
        return kots.dynamics(order=order)
    if op_name.endswith("jacobian"):
        return kots.jacobian(states)
    if op_name.endswith("matvec"):
        if vec is None:
            raise ValueError(f"{op_name} requires vec")
        return kots.jacobian_mul(states, vec)
    raise ValueError(f"Invalid op_name: {op_name}")


def _operation_value(kots: Kots, op_name: str, states: list[StateType], vec: np.ndarray | None, order: int):
    value = _operation_call(kots, op_name, states, vec, order)
    if value is not None:
        return value
    # State builders return None; compare a representative state instead.
    return kots.state_info(states[-1])


def _build_context(model_path: Path, order: int, motions: np.ndarray) -> tuple[Kots, list[Kots]]:
    batch_kots = Kots.from_json_file(str(model_path), order=order)
    batch_kots.import_motions(motions)

    serial_kots = []
    for motion in motions:
        kots = Kots.from_json_file(str(model_path), order=order)
        kots.import_motions(motion)
        serial_kots.append(kots)
    return batch_kots, serial_kots


def _serial_call(
    serial_kots: list[Kots],
    op_name: str,
    states: list[StateType],
    vecs: np.ndarray | None,
    order: int,
):
    values = []
    for i, kots in enumerate(serial_kots):
        vec = None if vecs is None else vecs[i]
        values.append(_operation_call(kots, op_name, states, vec, order))
    return values


def _serial_value(
    serial_kots: list[Kots],
    op_name: str,
    states: list[StateType],
    vecs: np.ndarray | None,
    order: int,
) -> np.ndarray:
    values = []
    for i, kots in enumerate(serial_kots):
        vec = None if vecs is None else vecs[i]
        values.append(_operation_value(kots, op_name, states, vec, order))
    return np.stack(values)


def _validate_batch_result(
    batch_kots: Kots,
    serial_kots: list[Kots],
    op_name: str,
    states: list[StateType],
    vecs: np.ndarray | None,
    order: int,
    rtol: float,
    atol: float,
) -> None:
    batch_value = _operation_value(batch_kots, op_name, states, vecs, order)
    serial_value = _serial_value(serial_kots, op_name, states, vecs, order)
    np.testing.assert_allclose(batch_value, serial_value, rtol=rtol, atol=atol)


def _run_one(
    model_path: Path,
    order: int,
    op_name: str,
    batch_size: int,
    repeat: int,
    warmup: int,
    rng: np.random.Generator,
    validate: bool,
    rtol: float,
    atol: float,
) -> None:
    template = Kots.from_json_file(str(model_path), order=order)
    kinematics_states, dynamics_states = _state_types(template)
    states = _operation_states(op_name, kinematics_states, dynamics_states)
    required_order = StateType.max_time_order(states)
    if required_order > order:
        raise ValueError(f"{op_name} requires order {required_order}, but CONFIG['order'] is {order}")

    motions = rng.standard_normal((batch_size, template.dof() * order))
    vecs = None
    if op_name.endswith("matvec"):
        vecs = rng.standard_normal((batch_size, template.dof() * required_order))

    batch_kots, serial_kots = _build_context(model_path, order, motions)
    if _operation_needs_state(op_name):
        _prepare_state(batch_kots, op_name, order)
        for kots in serial_kots:
            _prepare_state(kots, op_name, order)

    if validate:
        _validate_batch_result(batch_kots, serial_kots, op_name, states, vecs, required_order, rtol=rtol, atol=atol)

    batch_stats = _measure(
        lambda: _operation_call(batch_kots, op_name, states, vecs, required_order),
        repeats=repeat,
        warmup=warmup,
    )
    serial_stats = _measure(
        lambda: _serial_call(serial_kots, op_name, states, vecs, required_order),
        repeats=repeat,
        warmup=warmup,
    )
    _print_result(op_name, batch_size, batch_stats, serial_stats)


def main() -> None:
    _validate_config(CONFIG)
    model_path = Path(CONFIG["model"]).resolve()
    order = int(CONFIG["order"])
    repeat = int(CONFIG["repeat"])
    warmup = int(CONFIG["warmup"])
    rng = np.random.default_rng(int(CONFIG["seed"]))
    batch_sizes = [int(size) for size in CONFIG["batch_sizes"]]
    selected_ops = [str(name) for name in CONFIG["ops"]]
    validate = bool(CONFIG.get("validate", True))
    rtol = float(CONFIG.get("rtol", 1e-10))
    atol = float(CONFIG.get("atol", 1e-10))

    template = Kots.from_json_file(str(model_path), order=order)
    print("=== RoboKots Batch Benchmark ===", flush=True)
    print(f"model      : {model_path}", flush=True)
    print(f"order      : {order}", flush=True)
    print(f"dof        : {template.dof()}", flush=True)
    print(f"ops        : {', '.join(selected_ops)}", flush=True)
    print(f"batch_sizes: {batch_sizes}", flush=True)
    print(f"warmup     : {warmup}", flush=True)
    print(f"repeat     : {repeat}", flush=True)
    print(f"validate   : {validate}", flush=True)
    print(flush=True)

    for op_name in selected_ops:
        for batch_size in batch_sizes:
            _run_one(
                model_path=model_path,
                order=order,
                op_name=op_name,
                batch_size=batch_size,
                repeat=repeat,
                warmup=warmup,
                rng=rng,
                validate=validate,
                rtol=rtol,
                atol=atol,
            )


if __name__ == "__main__":
    main()
