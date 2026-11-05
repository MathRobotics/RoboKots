from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from robokots.kots import Kots
from robokots.outward.state import build_dynamics_outward_state, build_kinematics_outward_state

from .common import build_model, format_time, measure, select_unit, write_csv
from .pinocchio_compare import _optional_pinocchio, build_pinocchio_model


DEFAULT_CSV_PATH = Path(__file__).resolve().with_name("rust_cmtm_compare_results.csv")

CONFIG = {
    "dof_list": [16, 64],
    "kinematics_orders": [3, 5],
    "dynamics_orders": [1, 3],
    "batch_sizes": [1, 8, 64],
    "repeat": 20,
    "warmup": 3,
    "seed": 0,
    "model_kind": "humanoid",
    "validate": True,
    "include_rust_data": True,
    "rtol": 1e-8,
    "atol": 1e-8,
    "csv_path": DEFAULT_CSV_PATH,
}


def _optional_rust_backend():
    try:
        from robokots._rust import RustCompiledRobot
    except ImportError:
        return None
    return RustCompiledRobot


def _motion_from_qva(q: np.ndarray, v: np.ndarray, a: np.ndarray, order: int) -> np.ndarray:
    if order < 3:
        raise ValueError("q/v/a motion construction requires order >= 3")
    motion = np.zeros(q.size * order, dtype=float)
    for i in range(q.size):
        motion[i * order] = q[i]
        motion[i * order + 1] = v[i]
        motion[i * order + 2] = a[i]
    return motion


def _qva_from_motion(motion: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blocks = np.asarray(motion, dtype=float).reshape(-1, order)
    return blocks[:, 0], blocks[:, 1], blocks[:, 2]


def _pinocchio_loop(pin, model, motions: np.ndarray, order: int, op_name: str):
    data = model.createData()

    def run():
        for motion in motions:
            q, v, a = _qva_from_motion(motion, order)
            if op_name == "kinematics_cmtm":
                pin.forwardKinematics(model, data, q, v, a)
            elif op_name == "dynamics_cmtm":
                pin.rnea(model, data, q, v, a)
            else:
                raise ValueError(f"invalid op_name: {op_name}")

    return run


def _python_runner(robot, motions: np.ndarray, op_name: str, order: int):
    if motions.shape[0] == 1:
        motion = motions[0]
        if op_name == "kinematics_cmtm":
            return lambda: build_kinematics_outward_state(robot, motion, order)
        if op_name == "dynamics_cmtm":
            return lambda: build_dynamics_outward_state(robot, motion, order - 2)
    if op_name == "kinematics_cmtm":
        return lambda: build_kinematics_outward_state(robot, motions, order)
    if op_name == "dynamics_cmtm":
        return lambda: build_dynamics_outward_state(robot, motions, order - 2)
    raise ValueError(f"invalid op_name: {op_name}")


def _rust_runner(rust_robot, motions: np.ndarray, op_name: str, order: int):
    if motions.shape[0] == 1:
        motion = motions[0]
        if op_name == "kinematics_cmtm":
            return lambda: rust_robot.kinematics_cmtm(motion, order)
        if op_name == "dynamics_cmtm":
            return lambda: rust_robot.dynamics_outward_cmtm(motion, order - 2)
    if op_name == "kinematics_cmtm":
        return lambda: rust_robot.kinematics_cmtm_batch(motions, order)
    if op_name == "dynamics_cmtm":
        return lambda: rust_robot.dynamics_outward_cmtm_batch(motions, order - 2)
    raise ValueError(f"invalid op_name: {op_name}")


def _rust_data_runner(data, motions: np.ndarray, op_name: str, order: int):
    if op_name == "kinematics_cmtm":
        return lambda: data.compute_kinematics(motions[0] if motions.shape[0] == 1 else motions)
    if op_name == "dynamics_cmtm":
        return lambda: data.compute_dynamics(motions[0] if motions.shape[0] == 1 else motions)
    raise ValueError(f"invalid op_name: {op_name}")


def _max_abs_tuple_error(left, right) -> float:
    return max(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) for a, b in zip(left, right))


def _validate_kinematics(robot, rust_robot, motions: np.ndarray, order: int, rtol: float, atol: float) -> float:
    if motions.shape[0] == 1:
        py_state = build_kinematics_outward_state(robot, motions[0], order)
        rust_out = rust_robot.kinematics_cmtm(motions[0], order)
        py_out = _kinematics_state_arrays(robot, py_state, order)
    else:
        rust_out = rust_robot.kinematics_cmtm_batch(motions, order)
        py_out = tuple(
            np.stack(
                [
                    _kinematics_state_arrays(robot, build_kinematics_outward_state(robot, motion, order), order)[i]
                    for motion in motions
                ]
            )
            for i in range(4)
        )
    for actual, expected in zip(rust_out, py_out):
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    return _max_abs_tuple_error(rust_out, py_out)


def _validate_dynamics(robot, rust_robot, motions: np.ndarray, dynamics_order: int, rtol: float, atol: float) -> float:
    order = dynamics_order + 2
    if motions.shape[0] == 1:
        py_state = build_dynamics_outward_state(robot, motions[0], dynamics_order)
        rust_out = rust_robot.dynamics_outward_cmtm(motions[0], dynamics_order)
        py_out = (
            *_kinematics_state_arrays(robot, py_state, order),
            *_dynamics_state_arrays(robot, py_state, dynamics_order),
        )
    else:
        rust_out = rust_robot.dynamics_outward_cmtm_batch(motions, dynamics_order)
        py_states = [
            build_dynamics_outward_state(robot, motion, dynamics_order)
            for motion in motions
        ]
        py_out = tuple(
            np.stack(
                [
                    (
                        *_kinematics_state_arrays(
                            robot,
                            py_state,
                            order,
                        ),
                        *_dynamics_state_arrays(
                            robot,
                            py_state,
                            dynamics_order,
                        ),
                    )[i]
                    for py_state in py_states
                ]
            )
            for i in range(9)
        )
    for actual, expected in zip(rust_out, py_out):
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    _ = order
    return _max_abs_tuple_error(rust_out, py_out)


def _kinematics_state_arrays(robot, state, order: int):
    link_mat = []
    link_vecs = []
    for link in robot.links:
        cmtm = state.link_cmtm[link.name]
        link_mat.append(np.asarray(cmtm.elem_mat()))
        link_vecs.append(np.asarray(cmtm.vecs()))
    joint_mat = []
    joint_vecs = []
    for joint in robot.joints:
        cmtm = state.joint_cmtm[joint.name]
        joint_mat.append(np.asarray(cmtm.elem_mat()))
        joint_vecs.append(np.asarray(cmtm.vecs()))
    if order == 1:
        link_vecs = [np.zeros((0, 6)) for _ in robot.links]
        joint_vecs = [np.zeros((0, 6)) for _ in robot.joints]
    return np.stack(link_mat), np.stack(link_vecs), np.stack(joint_mat), np.stack(joint_vecs)


def _dynamics_state_arrays(robot, state, dynamics_order: int):
    link_momentum = []
    link_force = []
    for link in robot.links:
        link_momentum.append(np.asarray(state.link_momentum[link.name].vecs()))
        if dynamics_order > 0:
            link_force.append(np.asarray(state.link_force[link.name].vecs()))
    joint_momentum = []
    joint_force = []
    joint_torque = np.zeros((len(robot.joints), dynamics_order, 1), dtype=float)
    for i, joint in enumerate(robot.joints):
        joint_momentum.append(np.asarray(state.joint_momentum[joint.name].vecs()))
        if dynamics_order > 0:
            joint_force.append(np.asarray(state.joint_force[joint.name].vecs()))
            torque = state.joint_torque.get(joint.name)
            if torque is not None:
                joint_torque[i] = np.asarray(torque).reshape(dynamics_order, joint.dof)
    if dynamics_order == 0:
        link_force = [np.zeros((0, 6)) for _ in robot.links]
        joint_force = [np.zeros((0, 6)) for _ in robot.joints]
    return (
        np.stack(link_momentum),
        np.stack(link_force),
        np.stack(joint_momentum),
        np.stack(joint_force),
        joint_torque,
    )


def _available_pinocchio(op_name: str, order: int) -> bool:
    return order == 3 and op_name in {"kinematics_cmtm", "dynamics_cmtm"}


def _print_result(
    op_name: str,
    dof: int,
    order: int,
    batch_size: int,
    py_stats: dict[str, float],
    rust_stats: dict[str, float],
    rust_data_stats: dict[str, float] | None,
    pin_stats: dict[str, float] | None,
    max_error: float,
) -> None:
    values = [py_stats["mean_ms"], rust_stats["mean_ms"]]
    if rust_data_stats is not None:
        values.append(rust_data_stats["mean_ms"])
    if pin_stats is not None:
        values.append(pin_stats["mean_ms"])
    unit = select_unit(values)
    speedup_py = py_stats["mean_ms"] / rust_stats["mean_ms"] if rust_stats["mean_ms"] > 0 else float("inf")
    print(f"{op_name:16s} dof={dof:4d} order={order:2d} batch={batch_size:4d}", flush=True)
    print(
        "  python    "
        f"mean={format_time(py_stats['mean_ms'], unit)} "
        f"std={format_time(py_stats['std_ms'], unit)} "
        f"min={format_time(py_stats['min_ms'], unit)}",
        flush=True,
    )
    print(
        "  rust      "
        f"mean={format_time(rust_stats['mean_ms'], unit)} "
        f"std={format_time(rust_stats['std_ms'], unit)} "
        f"min={format_time(rust_stats['min_ms'], unit)} "
        f"speedup(py/rust)={speedup_py:8.2f} "
        f"max_err={max_error:.3e}",
        flush=True,
    )
    if rust_data_stats is not None:
        speedup_py_data = py_stats["mean_ms"] / rust_data_stats["mean_ms"] if rust_data_stats["mean_ms"] > 0 else float("inf")
        print(
            "  rust_data "
            f"mean={format_time(rust_data_stats['mean_ms'], unit)} "
            f"std={format_time(rust_data_stats['std_ms'], unit)} "
            f"min={format_time(rust_data_stats['min_ms'], unit)} "
            f"speedup(py/rust_data)={speedup_py_data:8.2f}",
            flush=True,
        )
    if pin_stats is not None:
        ratio_rust_pin = rust_stats["mean_ms"] / pin_stats["mean_ms"] if pin_stats["mean_ms"] > 0 else float("inf")
        print(
            "  pinocchio "
            f"mean={format_time(pin_stats['mean_ms'], unit)} "
            f"std={format_time(pin_stats['std_ms'], unit)} "
            f"min={format_time(pin_stats['min_ms'], unit)} "
            f"ratio(rust/pinocchio)={ratio_rust_pin:8.2f}",
            flush=True,
        )


def _run_one(
    *,
    pin,
    rust_backend,
    dof: int,
    model_kind: str,
    op_name: str,
    order: int,
    batch_size: int,
    repeat: int,
    warmup: int,
    rng: np.random.Generator,
    validate: bool,
    include_rust_data: bool,
    rtol: float,
    atol: float,
) -> dict[str, object]:
    model_data = build_model(dof, model_kind)
    kots = Kots.from_json_data(model_data, order=max(order, 3))
    robot = kots.robot_
    rust_robot = rust_backend.from_model_data(model_data)
    rust_data = (
        kots._create_rust_outward_state(order=order)
        if batch_size == 1
        else kots._create_rust_batch_outward_state(order=order, batch_shape=(batch_size,))
    ) if include_rust_data else None
    pin_model = build_pinocchio_model(pin, model_data) if pin is not None else None

    motions = rng.standard_normal((batch_size, rust_robot.dof * order))
    if _available_pinocchio(op_name, order):
        q = rng.standard_normal(rust_robot.dof)
        v = rng.standard_normal(rust_robot.dof)
        a = rng.standard_normal(rust_robot.dof)
        base_motion = _motion_from_qva(q, v, a, order)
        motions = np.vstack([base_motion if i == 0 else rng.standard_normal(rust_robot.dof * order) for i in range(batch_size)])
        if batch_size > 1:
            for i in range(1, batch_size):
                q = rng.standard_normal(rust_robot.dof)
                v = rng.standard_normal(rust_robot.dof)
                a = rng.standard_normal(rust_robot.dof)
                motions[i] = _motion_from_qva(q, v, a, order)

    if validate:
        if op_name == "kinematics_cmtm":
            max_error = _validate_kinematics(robot, rust_robot, motions, order, rtol, atol)
        else:
            max_error = _validate_dynamics(robot, rust_robot, motions, order - 2, rtol, atol)
    else:
        max_error = float("nan")

    py_stats = measure(_python_runner(robot, motions, op_name, order), repeats=repeat, warmup=warmup)
    rust_stats = measure(_rust_runner(rust_robot, motions, op_name, order), repeats=repeat, warmup=warmup)
    rust_data_stats = (
        measure(_rust_data_runner(rust_data, motions, op_name, order), repeats=repeat, warmup=warmup)
        if rust_data is not None
        else None
    )
    pin_stats = None
    if pin_model is not None and _available_pinocchio(op_name, order):
        pin_stats = measure(_pinocchio_loop(pin, pin_model, motions, order, op_name), repeats=repeat, warmup=warmup)

    _print_result(op_name, rust_robot.dof, order, batch_size, py_stats, rust_stats, rust_data_stats, pin_stats, max_error)

    return {
        "op": op_name,
        "model_kind": model_kind,
        "dof": rust_robot.dof,
        "link_num": rust_robot.link_num,
        "joint_num": rust_robot.joint_num,
        "order": order,
        "dynamics_order": order - 2 if op_name == "dynamics_cmtm" else "",
        "batch_size": batch_size,
        "python_mean_ms": py_stats["mean_ms"],
        "python_std_ms": py_stats["std_ms"],
        "python_min_ms": py_stats["min_ms"],
        "rust_mean_ms": rust_stats["mean_ms"],
        "rust_std_ms": rust_stats["std_ms"],
        "rust_min_ms": rust_stats["min_ms"],
        "rust_data_mean_ms": rust_data_stats["mean_ms"] if rust_data_stats is not None else "",
        "rust_data_std_ms": rust_data_stats["std_ms"] if rust_data_stats is not None else "",
        "rust_data_min_ms": rust_data_stats["min_ms"] if rust_data_stats is not None else "",
        "pinocchio_mean_ms": pin_stats["mean_ms"] if pin_stats is not None else "",
        "pinocchio_std_ms": pin_stats["std_ms"] if pin_stats is not None else "",
        "pinocchio_min_ms": pin_stats["min_ms"] if pin_stats is not None else "",
        "speedup_python_over_rust": (
            py_stats["mean_ms"] / rust_stats["mean_ms"] if rust_stats["mean_ms"] > 0 else float("inf")
        ),
        "speedup_python_over_rust_data": (
            py_stats["mean_ms"] / rust_data_stats["mean_ms"]
            if rust_data_stats is not None and rust_data_stats["mean_ms"] > 0
            else ""
        ),
        "ratio_rust_over_pinocchio": (
            rust_stats["mean_ms"] / pin_stats["mean_ms"]
            if pin_stats is not None and pin_stats["mean_ms"] > 0
            else ""
        ),
        "max_error_rust_vs_python": max_error,
        "note": "Pinocchio columns are category comparisons only and are present for order=3 rows.",
    }


def _validate_config(config: dict) -> None:
    if int(config["repeat"]) < 1:
        raise ValueError("CONFIG['repeat'] must be >= 1")
    if int(config["warmup"]) < 0:
        raise ValueError("CONFIG['warmup'] must be >= 0")
    if any(int(order) < 1 for order in config["kinematics_orders"]):
        raise ValueError("kinematics orders must be >= 1")
    if any(int(order) < 0 for order in config["dynamics_orders"]):
        raise ValueError("dynamics orders must be >= 0")
    if any(int(size) < 1 for size in config["batch_sizes"]):
        raise ValueError("batch sizes must be >= 1")


def main() -> None:
    _validate_config(CONFIG)
    rust_backend = _optional_rust_backend()
    if rust_backend is None:
        print("RoboKots Rust backend is not available; build robokots/_rust first.", flush=True)
        return

    pin = _optional_pinocchio()
    dof_list = [int(dof) for dof in CONFIG["dof_list"]]
    kinematics_orders = [int(order) for order in CONFIG["kinematics_orders"]]
    dynamics_orders = [int(order) for order in CONFIG["dynamics_orders"]]
    batch_sizes = [int(size) for size in CONFIG["batch_sizes"]]
    repeat = int(CONFIG["repeat"])
    warmup = int(CONFIG["warmup"])
    model_kind = str(CONFIG["model_kind"])
    validate = bool(CONFIG.get("validate", True))
    include_rust_data = bool(CONFIG.get("include_rust_data", True))
    rtol = float(CONFIG.get("rtol", 1e-8))
    atol = float(CONFIG.get("atol", 1e-8))
    csv_path = Path(CONFIG.get("csv_path", DEFAULT_CSV_PATH)).resolve()
    rng = np.random.default_rng(int(CONFIG["seed"]))

    print("=== Rust CMTM RoboKots Benchmark ===", flush=True)
    print(f"model_kind       : {model_kind}", flush=True)
    print(f"dof_list         : {dof_list}", flush=True)
    print(f"kinematics_orders: {kinematics_orders}", flush=True)
    print(f"dynamics_orders  : {dynamics_orders}", flush=True)
    print(f"batch_sizes      : {batch_sizes}", flush=True)
    print(f"warmup           : {warmup}", flush=True)
    print(f"repeat           : {repeat}", flush=True)
    print(f"validate         : {validate}", flush=True)
    print(f"rust_data        : {'enabled' if include_rust_data else 'disabled'}", flush=True)
    print(f"pinocchio        : {'enabled' if pin is not None else 'disabled'}", flush=True)
    print(f"csv_path         : {csv_path}", flush=True)
    print("note             : Pinocchio is shown only for order=3 category comparisons.", flush=True)
    print(flush=True)

    rows = []
    for dof in dof_list:
        print(f"--- dof={dof} ---", flush=True)
        for batch_size in batch_sizes:
            for order in kinematics_orders:
                rows.append(
                    _run_one(
                        pin=pin,
                        rust_backend=rust_backend,
                        dof=dof,
                        model_kind=model_kind,
                        op_name="kinematics_cmtm",
                        order=order,
                        batch_size=batch_size,
                        repeat=repeat,
                        warmup=warmup,
                        rng=rng,
                        validate=validate,
                        include_rust_data=include_rust_data,
                        rtol=rtol,
                        atol=atol,
                    )
                )
            for dynamics_order in dynamics_orders:
                rows.append(
                    _run_one(
                        pin=pin,
                        rust_backend=rust_backend,
                        dof=dof,
                        model_kind=model_kind,
                        op_name="dynamics_cmtm",
                        order=dynamics_order + 2,
                        batch_size=batch_size,
                        repeat=repeat,
                        warmup=warmup,
                        rng=rng,
                        validate=validate,
                        include_rust_data=include_rust_data,
                        rtol=rtol,
                        atol=atol,
                    )
                )
        print(flush=True)

    write_csv(csv_path, rows)
    print(f"wrote csv        : {csv_path}", flush=True)


if __name__ == "__main__":
    main()
