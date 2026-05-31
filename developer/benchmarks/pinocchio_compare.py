from __future__ import annotations

from pathlib import Path

import numpy as np

from robokots.core.state import StateType
from robokots.kots import Kots

from .common import build_model, format_time, leaf_link_names, measure, select_unit, write_csv


DEFAULT_CSV_PATH = Path(__file__).resolve().with_name("pinocchio_compare_results.csv")

# Pinocchio is intentionally optional. This script skips cleanly if it is not
# installed in the active environment.
CONFIG = {
    "dof_list": [16, 32, 64],
    "order": 3,
    "repeat": 10,
    "warmup": 2,
    "seed": 0,
    "model_kind": "humanoid",
    "ops": ["kinematics", "dynamics", "joint_jacobians"],
    "csv_path": DEFAULT_CSV_PATH,
}


def _optional_pinocchio():
    try:
        import pinocchio as pin
    except ImportError:
        return None
    return pin


def _axis_joint_model(pin, axis: np.ndarray):
    axis = np.asarray(axis, dtype=float)
    if np.allclose(axis, [1.0, 0.0, 0.0]):
        return pin.JointModelRX()
    if np.allclose(axis, [0.0, 1.0, 0.0]):
        return pin.JointModelRY()
    if np.allclose(axis, [0.0, 0.0, 1.0]):
        return pin.JointModelRZ()
    return pin.JointModelRevoluteUnaligned(axis)


def _placement_from_origin(pin, origin: dict):
    position = np.asarray(origin.get("position", [0.0, 0.0, 0.0]), dtype=float)
    quat_values = np.asarray(origin.get("orientation", [1.0, 0.0, 0.0, 0.0]), dtype=float)
    quat = pin.Quaternion(quat_values[0], quat_values[1], quat_values[2], quat_values[3])
    return pin.SE3(quat.toRotationMatrix(), position)


def _inertia_from_link(pin, link: dict):
    mass = float(link.get("mass", 0.0))
    cog = np.asarray(link.get("cog", [0.0, 0.0, 0.0]), dtype=float)
    inertia_values = np.asarray(link.get("inertia", [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), dtype=float)
    inertia = np.array(
        [
            [inertia_values[0], inertia_values[3], inertia_values[4]],
            [inertia_values[3], inertia_values[1], inertia_values[5]],
            [inertia_values[4], inertia_values[5], inertia_values[2]],
        ],
        dtype=float,
    )
    return pin.Inertia(mass, cog, inertia)


def build_pinocchio_model(pin, model_data: dict):
    model = pin.Model()
    link_to_joint_id = {0: 0}

    for joint in model_data["joints"]:
        parent_link_id = int(joint["parent_link_id"])
        child_link_id = int(joint["child_link_id"])
        parent_joint_id = link_to_joint_id[parent_link_id]
        placement = _placement_from_origin(pin, joint.get("origin", {}))

        if joint["type"] == "fix":
            link_to_joint_id[child_link_id] = parent_joint_id
            continue

        if joint["type"] != "revolute":
            raise ValueError(f"Unsupported joint type for Pinocchio comparison: {joint['type']}")

        joint_model = _axis_joint_model(pin, joint.get("axis", [0.0, 0.0, 1.0]))
        joint_id = model.addJoint(parent_joint_id, joint_model, placement, joint["name"])
        child_link = model_data["links"][child_link_id]
        model.appendBodyToJoint(joint_id, _inertia_from_link(pin, child_link), pin.SE3.Identity())
        link_to_joint_id[child_link_id] = joint_id

    return model


def _motion_from_qva(q: np.ndarray, v: np.ndarray, a: np.ndarray, order: int) -> np.ndarray:
    if order < 3:
        raise ValueError("RoboKots comparison order must be >= 3")
    parts = []
    for i in range(q.size):
        block = np.zeros(order, dtype=float)
        block[0] = q[i]
        block[1] = v[i]
        block[2] = a[i]
        parts.append(block)
    return np.concatenate(parts)


def _pinocchio_runner(pin, model, q, v, a, op_name: str):
    data = model.createData()
    if op_name == "kinematics":
        return lambda: pin.forwardKinematics(model, data, q, v, a)
    if op_name == "dynamics":
        return lambda: pin.rnea(model, data, q, v, a)
    if op_name == "joint_jacobians":
        return lambda: pin.computeJointJacobians(model, data, q)
    raise ValueError(f"Invalid Pinocchio op: {op_name}")


def _robokots_runner(kots: Kots, target_link: str, op_name: str, order: int):
    if op_name == "kinematics":
        return lambda: kots.kinematics(order=order)
    if op_name == "dynamics":
        return lambda: kots.dynamics(order=order)
    if op_name == "joint_jacobians":
        state = StateType("link", target_link, "acc")

        def run():
            kots.kinematics(order=order)
            return kots.jacobian(state)

        return run
    raise ValueError(f"Invalid RoboKots comparison op: {op_name}")


def _print_result(dof: int, op_name: str, pin_stats: dict[str, float], robokots_stats: dict[str, float]) -> None:
    unit = select_unit([pin_stats["mean_ms"], robokots_stats["mean_ms"]])
    ratio = robokots_stats["mean_ms"] / pin_stats["mean_ms"] if pin_stats["mean_ms"] > 0 else float("inf")
    print(f"{op_name:16s} dof={dof:4d}", flush=True)
    print(
        "  pinocchio "
        f"mean={format_time(pin_stats['mean_ms'], unit)} "
        f"std={format_time(pin_stats['std_ms'], unit)} "
        f"min={format_time(pin_stats['min_ms'], unit)}",
        flush=True,
    )
    print(
        "  robokots  "
        f"mean={format_time(robokots_stats['mean_ms'], unit)} "
        f"std={format_time(robokots_stats['std_ms'], unit)} "
        f"min={format_time(robokots_stats['min_ms'], unit)} "
        f"ratio(robokots/pinocchio)={ratio:8.2f}",
        flush=True,
    )


def _validate_config(config: dict) -> None:
    if int(config["order"]) < 3:
        raise ValueError("CONFIG['order'] must be >= 3")
    if int(config["repeat"]) < 1:
        raise ValueError("CONFIG['repeat'] must be >= 1")
    if int(config["warmup"]) < 0:
        raise ValueError("CONFIG['warmup'] must be >= 0")
    valid_ops = {"kinematics", "dynamics", "joint_jacobians"}
    invalid_ops = [op for op in config["ops"] if op not in valid_ops]
    if invalid_ops:
        raise ValueError(f"Invalid op(s): {invalid_ops}")


def main() -> None:
    _validate_config(CONFIG)
    pin = _optional_pinocchio()
    if pin is None:
        print("Pinocchio is not installed; skipping optional comparison.", flush=True)
        print("Install it in your developer environment separately, then rerun this module.", flush=True)
        return

    dof_list = [int(dof) for dof in CONFIG["dof_list"]]
    order = int(CONFIG["order"])
    repeat = int(CONFIG["repeat"])
    warmup = int(CONFIG["warmup"])
    selected_ops = [str(op) for op in CONFIG["ops"]]
    model_kind = str(CONFIG["model_kind"])
    csv_path = Path(CONFIG.get("csv_path", DEFAULT_CSV_PATH)).resolve()
    rng = np.random.default_rng(int(CONFIG["seed"]))

    print("=== RoboKots vs Pinocchio Developer Benchmark ===", flush=True)
    print(f"model_kind : {model_kind}", flush=True)
    print(f"dof_list   : {dof_list}", flush=True)
    print(f"order      : {order}", flush=True)
    print(f"ops        : {', '.join(selected_ops)}", flush=True)
    print(f"warmup     : {warmup}", flush=True)
    print(f"repeat     : {repeat}", flush=True)
    print(f"csv_path   : {csv_path}", flush=True)
    print("note       : compares runtime categories, not identical output semantics.", flush=True)
    print(flush=True)

    rows = []
    for dof in dof_list:
        model_data = build_model(dof, model_kind)
        pin_model = build_pinocchio_model(pin, model_data)
        kots = Kots.from_json_data(model_data, order=order)
        target_link = leaf_link_names(kots)[-1]

        q = rng.standard_normal(pin_model.nq)
        v = rng.standard_normal(pin_model.nv)
        a = rng.standard_normal(pin_model.nv)
        kots.import_motions(_motion_from_qva(q, v, a, order))

        print(
            f"--- dof={kots.dof()} pin_nq={pin_model.nq} "
            f"links={kots.robot_.link_num} joints={kots.robot_.joint_num} target={target_link} ---",
            flush=True,
        )
        for op_name in selected_ops:
            pin_stats = measure(_pinocchio_runner(pin, pin_model, q, v, a, op_name), repeats=repeat, warmup=warmup)
            robokots_stats = measure(_robokots_runner(kots, target_link, op_name, order), repeats=repeat, warmup=warmup)
            _print_result(dof, op_name, pin_stats, robokots_stats)
            rows.append(
                {
                    "op": op_name,
                    "model_kind": model_kind,
                    "dof": dof,
                    "order": order,
                    "pinocchio_nq": pin_model.nq,
                    "pinocchio_nv": pin_model.nv,
                    "robokots_mean_ms": robokots_stats["mean_ms"],
                    "robokots_std_ms": robokots_stats["std_ms"],
                    "robokots_min_ms": robokots_stats["min_ms"],
                    "pinocchio_mean_ms": pin_stats["mean_ms"],
                    "pinocchio_std_ms": pin_stats["std_ms"],
                    "pinocchio_min_ms": pin_stats["min_ms"],
                    "ratio_robokots_over_pinocchio": (
                        robokots_stats["mean_ms"] / pin_stats["mean_ms"]
                        if pin_stats["mean_ms"] > 0
                        else float("inf")
                    ),
                    "note": "runtime category comparison; outputs are not exactly equivalent",
                }
            )
        print(flush=True)

    write_csv(csv_path, rows)
    print(f"wrote csv  : {csv_path}", flush=True)


if __name__ == "__main__":
    main()
