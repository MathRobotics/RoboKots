from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Callable

import numpy as np

from robokots.core.state import StateType
from robokots.kots import Kots


ROBOKOTS_OPS = (
    "kinematics",
    "dynamics",
    "kinematics_jacobian",
    "dynamics_jacobian",
    "kinematics_matvec",
    "dynamics_matvec",
)


def select_unit(values_ms: list[float], threshold_ms: float = 0.1) -> str:
    finite_values = [abs(v) for v in values_ms if np.isfinite(v)]
    if not finite_values:
        return "ms"
    return "us" if max(finite_values) < threshold_ms else "ms"


def format_time(value_ms: float, unit: str) -> str:
    if unit == "us":
        return f"{value_ms * 1e3:9.3f}us"
    return f"{value_ms:9.3f}ms"


def measure(fn: Callable[[], object], repeats: int, warmup: int) -> dict[str, float]:
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


def write_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def add_revolute_chain(
    links: list[dict],
    joints: list[dict],
    parent_link_id: int,
    prefix: str,
    count: int,
    axes: tuple[tuple[float, float, float], ...],
    offset: tuple[float, float, float],
) -> int:
    current_parent = parent_link_id
    for i in range(count):
        link_id = len(links)
        joint_id = len(joints)
        links.append(
            {
                "id": link_id,
                "name": f"{prefix}{i + 1}",
                "mass": 4.0,
                "cog": [0.0, 0.0, 0.0],
                "inertia": [0.05, 0.05, 0.02, 0.0, 0.0, 0.0],
                "geometry": "generated_link.stl",
            }
        )
        joints.append(
            {
                "id": joint_id,
                "name": f"{prefix}_joint{i + 1}",
                "type": "revolute",
                "parent_link_id": current_parent,
                "child_link_id": link_id,
                "axis": [float(v) for v in axes[i % len(axes)]],
                "limits": {"min": -1.57, "max": 1.57},
                "origin": {
                    "position": [float(v) for v in offset],
                    "orientation": [1.0, 0.0, 0.0, 0.0],
                },
            }
        )
        current_parent = link_id
    return current_parent


def build_humanoid_like_model(dof: int) -> dict:
    if dof < 1:
        raise ValueError("dof must be >= 1")

    links = [
        {"id": 0, "name": "world"},
        {
            "id": 1,
            "name": "pelvis",
            "mass": 12.0,
            "cog": [0.0, 0.0, 0.0],
            "inertia": [0.2, 0.2, 0.2, 0.0, 0.0, 0.0],
            "geometry": "generated_pelvis.stl",
        },
    ]
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

    axes = (
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
    )
    remaining = dof

    def add(prefix: str, count: int, parent: int, offset: tuple[float, float, float]) -> int:
        nonlocal remaining
        used = min(count, remaining)
        remaining -= used
        if used == 0:
            return parent
        return add_revolute_chain(links, joints, parent, prefix, used, axes, offset)

    spine_end = add("spine", 3, 1, (0.0, 0.0, 0.25))
    add("neck", 2, spine_end, (0.0, 0.0, 0.18))
    add("left_arm", 7, spine_end, (0.18, 0.22, 0.0))
    add("right_arm", 7, spine_end, (-0.18, 0.22, 0.0))
    add("left_leg", 6, 1, (0.12, -0.08, -0.28))
    add("right_leg", 6, 1, (-0.12, -0.08, -0.28))
    add("extra", remaining, spine_end, (0.0, 0.18, 0.0))
    return {"links": links, "joints": joints}


def build_serial_chain_model(dof: int) -> dict:
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
            "geometry": "generated_base.stl",
        },
    ]
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
    add_revolute_chain(
        links,
        joints,
        parent_link_id=1,
        prefix="chain",
        count=dof,
        axes=((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
        offset=(0.2, 0.0, 0.05),
    )
    return {"links": links, "joints": joints}


def build_model(dof: int, model_kind: str) -> dict:
    if model_kind == "humanoid":
        return build_humanoid_like_model(dof)
    if model_kind == "serial":
        return build_serial_chain_model(dof)
    raise ValueError("model_kind must be 'humanoid' or 'serial'")


def leaf_link_names(kots: Kots) -> list[str]:
    return [
        link.name
        for link in kots.robot_.links
        if link.name != "world" and not link.child_joint_ids
    ]


def parent_joint_name(kots: Kots, link_name: str) -> str:
    link = kots.robot_.link(link_name)
    if link is None or not link.parent_joint_ids:
        raise ValueError(f"Link has no parent joint: {link_name}")
    return kots.robot_.joints[link.parent_joint_ids[0]].name


def robokots_state_types(kots: Kots, target_count: int) -> tuple[list[StateType], list[StateType], list[str]]:
    leaf_names = leaf_link_names(kots)
    if not leaf_names:
        raise ValueError("Generated model has no leaf links")
    target_names = leaf_names[-min(len(leaf_names), target_count) :]

    kinematics_states = []
    dynamics_states = []
    for link_name in target_names:
        joint_name = parent_joint_name(kots, link_name)
        kinematics_states.extend(
            [
                StateType("link", link_name, "frame"),
                StateType("link", link_name, "vel"),
                StateType("link", link_name, "acc"),
                StateType("link", link_name, "snap"),
            ]
        )
        dynamics_states.extend(
            [
                StateType("link", link_name, "momentum_diff3"),
                StateType("link", link_name, "momentum_diff3", "world"),
                StateType("link", link_name, "force_diff2"),
                StateType("joint", joint_name, "momentum_diff3"),
                StateType("joint", joint_name, "momentum_diff3", "world"),
                StateType("joint", joint_name, "force_diff2"),
                StateType("joint", joint_name, "torque_diff2"),
            ]
        )
    return kinematics_states, dynamics_states, target_names


def robokots_operation_states(
    op_name: str,
    kinematics_states: list[StateType],
    dynamics_states: list[StateType],
) -> list[StateType]:
    return dynamics_states if op_name.startswith("dynamics") else kinematics_states


def robokots_prepare_state(kots: Kots, op_name: str, order: int) -> None:
    if op_name.startswith("dynamics"):
        kots.dynamics(order=order)
    else:
        kots.kinematics(order=order)


def robokots_operation_call(
    kots: Kots,
    op_name: str,
    states: list[StateType],
    vec: np.ndarray | None,
    order: int,
):
    if op_name == "kinematics":
        return kots.kinematics(order=order)
    if op_name == "dynamics":
        return kots.dynamics(order=order)
    if op_name.endswith("jacobian"):
        return kots.jacobian(states)
    if op_name.endswith("matvec"):
        if vec is None:
            raise ValueError(f"{op_name} requires vec")
        return kots.jacobian_matvec(states, vec)
    raise ValueError(f"Invalid op_name: {op_name}")


def robokots_operation_value(
    kots: Kots,
    op_name: str,
    states: list[StateType],
    vec: np.ndarray | None,
    order: int,
):
    value = robokots_operation_call(kots, op_name, states, vec, order)
    if value is not None:
        return value
    return kots.state_info(states[-1])
