from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import build_model, format_time, measure, select_unit, write_csv
from .pinocchio_compare import _optional_pinocchio, build_pinocchio_model
from robokots.kots import Kots


DEFAULT_CSV_PATH = Path(__file__).resolve().with_name("fast_minimal_compare_results.csv")

CONFIG = {
    "dof_list": [16, 32, 64],
    "repeat": 100,
    "warmup": 10,
    "seed": 0,
    "model_kind": "humanoid",
    "ops": ["kinematics", "dynamics", "joint_jacobians"],
    "include_rust": True,
    "include_rust_data": True,
    "include_kots_fast": True,
    "csv_path": DEFAULT_CSV_PATH,
}


@dataclass(frozen=True)
class CompiledRobot:
    link_num: int
    joint_num: int
    dof: int
    parent_link: np.ndarray
    child_link: np.ndarray
    q_index: np.ndarray
    axis: np.ndarray
    origin_R: np.ndarray
    origin_p: np.ndarray
    link_inertia: np.ndarray
    link_ancestors: tuple[np.ndarray, ...]


@dataclass
class FastWorkspace:
    R: np.ndarray
    p: np.ndarray
    w: np.ndarray
    lin_v: np.ndarray
    alpha: np.ndarray
    lin_a: np.ndarray
    forces: np.ndarray
    tau: np.ndarray
    jac: np.ndarray
    active_axes: np.ndarray
    active_points: np.ndarray


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=float,
    )


def _rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        return np.eye(3)
    a = axis / norm
    k = _skew(a)
    s = np.sin(angle)
    c = np.cos(angle)
    return np.eye(3) + s * k + (1.0 - c) * (k @ k)


def _quat_to_rot(quat_values) -> np.ndarray:
    q = np.asarray(quat_values, dtype=float)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _spatial_inertia(link: dict) -> np.ndarray:
    mass = float(link.get("mass", 0.0))
    cog = np.asarray(link.get("cog", [0.0, 0.0, 0.0]), dtype=float)
    iv = np.asarray(link.get("inertia", [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), dtype=float)
    inertia = np.array(
        [
            [iv[0], iv[3], iv[4]],
            [iv[3], iv[1], iv[5]],
            [iv[4], iv[5], iv[2]],
        ],
        dtype=float,
    )
    c_hat = _skew(cog)
    mat = np.zeros((6, 6), dtype=float)
    mat[:3, :3] = inertia - mass * c_hat @ c_hat
    mat[:3, 3:] = -mass * c_hat
    mat[3:, :3] = mass * c_hat
    mat[3:, 3:] = mass * np.eye(3)
    return mat


def compile_model(model_data: dict) -> CompiledRobot:
    joints = model_data["joints"]
    links = model_data["links"]
    parent_link = np.zeros(len(joints), dtype=int)
    child_link = np.zeros(len(joints), dtype=int)
    q_index = np.full(len(joints), -1, dtype=int)
    axis = np.zeros((len(joints), 3), dtype=float)
    origin_R = np.zeros((len(joints), 3, 3), dtype=float)
    origin_p = np.zeros((len(joints), 3), dtype=float)

    dof = 0
    link_ancestor_lists: list[list[int]] = [[] for _ in links]
    for i, joint in enumerate(joints):
        parent_link[i] = int(joint["parent_link_id"])
        child_link[i] = int(joint["child_link_id"])
        origin = joint.get("origin", {})
        origin_R[i] = _quat_to_rot(origin.get("orientation", [1.0, 0.0, 0.0, 0.0]))
        origin_p[i] = np.asarray(origin.get("position", [0.0, 0.0, 0.0]), dtype=float)
        if joint["type"] == "fix":
            axis[i] = np.array([1.0, 0.0, 0.0])
            link_ancestor_lists[child_link[i]] = list(link_ancestor_lists[parent_link[i]])
            continue
        if joint["type"] != "revolute":
            raise ValueError("minimal fast benchmark supports fixed/revolute joints only")
        raw_axis = np.asarray(joint.get("axis", [0.0, 0.0, 1.0]), dtype=float)
        axis[i] = raw_axis / np.linalg.norm(raw_axis)
        q_index[i] = dof
        link_ancestor_lists[child_link[i]] = [*link_ancestor_lists[parent_link[i]], dof]
        dof += 1

    link_inertia = np.stack([_spatial_inertia(link) for link in links])
    link_ancestors = tuple(np.asarray(items, dtype=int) for items in link_ancestor_lists)
    return CompiledRobot(
        link_num=len(links),
        joint_num=len(joints),
        dof=dof,
        parent_link=parent_link,
        child_link=child_link,
        q_index=q_index,
        axis=axis,
        origin_R=origin_R,
        origin_p=origin_p,
        link_inertia=link_inertia,
        link_ancestors=link_ancestors,
    )


def make_workspace(robot: CompiledRobot) -> FastWorkspace:
    return FastWorkspace(
        R=np.zeros((robot.link_num, 3, 3), dtype=float),
        p=np.zeros((robot.link_num, 3), dtype=float),
        w=np.zeros((robot.link_num, 3), dtype=float),
        lin_v=np.zeros((robot.link_num, 3), dtype=float),
        alpha=np.zeros((robot.link_num, 3), dtype=float),
        lin_a=np.zeros((robot.link_num, 3), dtype=float),
        forces=np.zeros((robot.link_num, 6), dtype=float),
        tau=np.zeros(robot.dof, dtype=float),
        jac=np.zeros((robot.link_num, 6, robot.dof), dtype=float),
        active_axes=np.zeros((robot.dof, 3), dtype=float),
        active_points=np.zeros((robot.dof, 3), dtype=float),
    )


def fast_forward_kinematics(
    robot: CompiledRobot,
    q: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    workspace: FastWorkspace | None = None,
):
    ws = workspace if workspace is not None else make_workspace(robot)
    R = ws.R
    p = ws.p
    w = ws.w
    lin_v = ws.lin_v
    alpha = ws.alpha
    lin_a = ws.lin_a
    R.fill(0.0)
    p.fill(0.0)
    w.fill(0.0)
    lin_v.fill(0.0)
    alpha.fill(0.0)
    lin_a.fill(0.0)
    R[0] = np.eye(3)

    for j in range(robot.joint_num):
        parent = robot.parent_link[j]
        child = robot.child_link[j]
        joint_R0 = R[parent] @ robot.origin_R[j]
        joint_p = p[parent] + R[parent] @ robot.origin_p[j]
        qi = robot.q_index[j]
        if qi >= 0:
            axis_world = joint_R0 @ robot.axis[j]
            Rj = _rot_axis(robot.axis[j], q[qi])
            R[child] = joint_R0 @ Rj
            p[child] = joint_p
            r = joint_p - p[parent]
            w[child] = w[parent] + axis_world * v[qi]
            lin_v[child] = lin_v[parent] + np.cross(w[parent], r)
            alpha[child] = alpha[parent] + axis_world * a[qi] + np.cross(w[parent], axis_world * v[qi])
            lin_a[child] = lin_a[parent] + np.cross(alpha[parent], r) + np.cross(w[parent], np.cross(w[parent], r))
        else:
            R[child] = joint_R0
            p[child] = joint_p
            r = joint_p - p[parent]
            w[child] = w[parent]
            lin_v[child] = lin_v[parent] + np.cross(w[parent], r)
            alpha[child] = alpha[parent]
            lin_a[child] = lin_a[parent] + np.cross(alpha[parent], r) + np.cross(w[parent], np.cross(w[parent], r))

    return R, p, w, lin_v, alpha, lin_a


def _spatial_cross_force(vel: np.ndarray, force: np.ndarray) -> np.ndarray:
    w = vel[:3]
    v = vel[3:]
    n = force[:3]
    f = force[3:]
    return np.concatenate([np.cross(w, n) + np.cross(v, f), np.cross(w, f)])


def fast_rnea(
    robot: CompiledRobot,
    q: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    workspace: FastWorkspace | None = None,
) -> np.ndarray:
    ws = workspace if workspace is not None else make_workspace(robot)
    R, p, w, lin_v, alpha, lin_a = fast_forward_kinematics(robot, q, v, a, ws)
    forces = ws.forces
    tau = ws.tau
    forces.fill(0.0)
    tau.fill(0.0)

    for link_id in range(1, robot.link_num):
        rot_t = R[link_id].T
        spatial_v = np.concatenate([rot_t @ w[link_id], rot_t @ lin_v[link_id]])
        spatial_a = np.concatenate(
            [
                rot_t @ alpha[link_id],
                rot_t @ lin_a[link_id] - np.cross(spatial_v[:3], spatial_v[3:]),
            ]
        )
        force = np.empty(6, dtype=float)
        force_local = robot.link_inertia[link_id] @ spatial_a
        momentum = robot.link_inertia[link_id] @ spatial_v
        force_local = force_local + _spatial_cross_force(spatial_v, momentum)
        force[:3] = R[link_id] @ force_local[:3]
        force[3:] = R[link_id] @ force_local[3:]
        forces[link_id] = force

    for j in range(robot.joint_num - 1, -1, -1):
        parent = robot.parent_link[j]
        child = robot.child_link[j]
        qi = robot.q_index[j]
        if qi >= 0:
            axis_world = R[parent] @ robot.origin_R[j] @ robot.axis[j]
            tau[qi] = axis_world @ forces[child, :3]
        rel = p[child] - p[parent]
        forces[parent, :3] += forces[child, :3] + np.cross(rel, forces[child, 3:])
        forces[parent, 3:] += forces[child, 3:]
    return tau


def fast_joint_jacobians(
    robot: CompiledRobot,
    q: np.ndarray,
    workspace: FastWorkspace | None = None,
) -> np.ndarray:
    ws = workspace if workspace is not None else make_workspace(robot)
    R, p, *_ = fast_forward_kinematics(robot, q, np.zeros_like(q), np.zeros_like(q), ws)
    jac = ws.jac
    active_axes = ws.active_axes
    active_points = ws.active_points
    jac.fill(0.0)

    for j in range(robot.joint_num):
        qi = robot.q_index[j]
        if qi < 0:
            continue
        parent = robot.parent_link[j]
        active_axes[qi] = R[parent] @ robot.origin_R[j] @ robot.axis[j]
        active_points[qi] = p[parent] + R[parent] @ robot.origin_p[j]

    for link_id in range(robot.link_num):
        for qi in robot.link_ancestors[link_id]:
            axis = active_axes[qi]
            jac[link_id, :3, qi] = axis
            jac[link_id, 3:, qi] = np.cross(axis, p[link_id] - active_points[qi])
    return jac


def _pinocchio_runner(pin, model, q, v, a, op_name: str):
    data = model.createData()
    if op_name == "kinematics":
        return lambda: pin.forwardKinematics(model, data, q, v, a)
    if op_name == "dynamics":
        return lambda: pin.rnea(model, data, q, v, a)
    if op_name == "joint_jacobians":
        return lambda: pin.computeJointJacobians(model, data, q)
    raise ValueError(f"invalid op: {op_name}")


def _optional_rust_backend():
    try:
        from robokots._rust import RustCompiledRobot
    except ImportError:
        return None
    return RustCompiledRobot


def _fast_runner(robot: CompiledRobot, q: np.ndarray, v: np.ndarray, a: np.ndarray, op_name: str):
    workspace = make_workspace(robot)
    if op_name == "kinematics":
        return lambda: fast_forward_kinematics(robot, q, v, a, workspace)
    if op_name == "dynamics":
        return lambda: fast_rnea(robot, q, v, a, workspace)
    if op_name == "joint_jacobians":
        return lambda: fast_joint_jacobians(robot, q, workspace)
    raise ValueError(f"invalid op: {op_name}")


def _rust_runner(robot, q: np.ndarray, v: np.ndarray, a: np.ndarray, op_name: str):
    if op_name == "kinematics":
        return lambda: robot.forward_kinematics(q, v, a)
    if op_name == "dynamics":
        return lambda: robot.rnea(q, v, a)
    if op_name == "joint_jacobians":
        return lambda: robot.joint_jacobians(q)
    raise ValueError(f"invalid op: {op_name}")


def _rust_data_runner(data, q: np.ndarray, v: np.ndarray, a: np.ndarray, op_name: str):
    if op_name == "kinematics":
        return lambda: data.compute_kinematics(q, v, a)
    if op_name == "dynamics":
        return lambda: data.compute_dynamics(q, v, a)
    if op_name == "joint_jacobians":
        return lambda: data.compute_joint_jacobians(q)
    raise ValueError(f"invalid op: {op_name}")


def _kots_fast_runner(kots: Kots, q: np.ndarray, v: np.ndarray, a: np.ndarray, op_name: str):
    if op_name == "kinematics":
        return lambda: kots._rust_fast_forward_kinematics(q, v, a)
    if op_name == "dynamics":
        return lambda: kots._rust_fast_rnea(q, v, a)
    if op_name == "joint_jacobians":
        return lambda: kots._rust_fast_joint_jacobians(q)
    raise ValueError(f"invalid op: {op_name}")


def _print_result(
    dof: int,
    op_name: str,
    pin_stats: dict[str, float],
    fast_stats: dict[str, float],
    rust_stats: dict[str, float] | None = None,
    rust_data_stats: dict[str, float] | None = None,
    kots_fast_stats: dict[str, float] | None = None,
) -> None:
    timing_values = [pin_stats["mean_ms"], fast_stats["mean_ms"]]
    if rust_stats is not None:
        timing_values.append(rust_stats["mean_ms"])
    if rust_data_stats is not None:
        timing_values.append(rust_data_stats["mean_ms"])
    if kots_fast_stats is not None:
        timing_values.append(kots_fast_stats["mean_ms"])
    unit = select_unit(timing_values)
    ratio = fast_stats["mean_ms"] / pin_stats["mean_ms"] if pin_stats["mean_ms"] > 0 else float("inf")
    print(f"{op_name:16s} dof={dof:4d}", flush=True)
    print(
        "  pinocchio "
        f"mean={format_time(pin_stats['mean_ms'], unit)} "
        f"std={format_time(pin_stats['std_ms'], unit)} "
        f"min={format_time(pin_stats['min_ms'], unit)}",
        flush=True,
    )
    print(
        "  fast      "
        f"mean={format_time(fast_stats['mean_ms'], unit)} "
        f"std={format_time(fast_stats['std_ms'], unit)} "
        f"min={format_time(fast_stats['min_ms'], unit)} "
        f"ratio(fast/pinocchio)={ratio:8.2f}",
        flush=True,
    )
    if rust_stats is not None:
        rust_ratio = rust_stats["mean_ms"] / pin_stats["mean_ms"] if pin_stats["mean_ms"] > 0 else float("inf")
        rust_vs_fast = fast_stats["mean_ms"] / rust_stats["mean_ms"] if rust_stats["mean_ms"] > 0 else float("inf")
        print(
            "  rust      "
            f"mean={format_time(rust_stats['mean_ms'], unit)} "
            f"std={format_time(rust_stats['std_ms'], unit)} "
            f"min={format_time(rust_stats['min_ms'], unit)} "
            f"ratio(rust/pinocchio)={rust_ratio:8.2f} "
            f"speedup(fast/rust)={rust_vs_fast:8.2f}",
            flush=True,
        )
    if rust_data_stats is not None:
        rust_data_ratio = rust_data_stats["mean_ms"] / pin_stats["mean_ms"] if pin_stats["mean_ms"] > 0 else float("inf")
        print(
            "  rust_data "
            f"mean={format_time(rust_data_stats['mean_ms'], unit)} "
            f"std={format_time(rust_data_stats['std_ms'], unit)} "
            f"min={format_time(rust_data_stats['min_ms'], unit)} "
            f"ratio(rust_data/pinocchio)={rust_data_ratio:8.2f}",
            flush=True,
        )
    if kots_fast_stats is not None:
        kots_fast_ratio = (
            kots_fast_stats["mean_ms"] / pin_stats["mean_ms"] if pin_stats["mean_ms"] > 0 else float("inf")
        )
        kots_fast_vs_rust = (
            kots_fast_stats["mean_ms"] / rust_stats["mean_ms"]
            if rust_stats is not None and rust_stats["mean_ms"] > 0
            else float("inf")
        )
        print(
            "  kots_fast "
            f"mean={format_time(kots_fast_stats['mean_ms'], unit)} "
            f"std={format_time(kots_fast_stats['std_ms'], unit)} "
            f"min={format_time(kots_fast_stats['min_ms'], unit)} "
            f"ratio(kots_fast/pinocchio)={kots_fast_ratio:8.2f} "
            f"overhead(kots_fast/rust)={kots_fast_vs_rust:8.2f}",
            flush=True,
        )


def main() -> None:
    pin = _optional_pinocchio()
    if pin is None:
        print("Pinocchio is not installed; skipping optional comparison.", flush=True)
        return

    dof_list = [int(dof) for dof in CONFIG["dof_list"]]
    repeat = int(CONFIG["repeat"])
    warmup = int(CONFIG["warmup"])
    selected_ops = [str(op) for op in CONFIG["ops"]]
    include_rust = bool(CONFIG.get("include_rust", True))
    include_rust_data = bool(CONFIG.get("include_rust_data", True))
    include_kots_fast = bool(CONFIG.get("include_kots_fast", True))
    rust_backend = _optional_rust_backend() if include_rust else None
    model_kind = str(CONFIG["model_kind"])
    csv_path = Path(CONFIG.get("csv_path", DEFAULT_CSV_PATH)).resolve()
    rng = np.random.default_rng(int(CONFIG["seed"]))

    print("=== Minimal Fast RoboKots vs Pinocchio Benchmark ===", flush=True)
    print(f"model_kind : {model_kind}", flush=True)
    print(f"dof_list   : {dof_list}", flush=True)
    print(f"ops        : {', '.join(selected_ops)}", flush=True)
    print(f"warmup     : {warmup}", flush=True)
    print(f"repeat     : {repeat}", flush=True)
    print(f"rust       : {'enabled' if rust_backend is not None else 'disabled'}", flush=True)
    print(f"rust_data  : {'enabled' if rust_backend is not None and include_rust_data else 'disabled'}", flush=True)
    print(f"kots_fast  : {'enabled' if include_kots_fast else 'disabled'}", flush=True)
    print(f"csv_path   : {csv_path}", flush=True)
    print("note       : q/v/a array-only kernels; no CMTM/state-dict materialization.", flush=True)
    print(flush=True)

    rows = []
    for dof in dof_list:
        model_data = build_model(dof, model_kind)
        pin_model = build_pinocchio_model(pin, model_data)
        fast_model = compile_model(model_data)
        rust_model = rust_backend.from_model_data(model_data) if rust_backend is not None else None
        rust_data = rust_model.create_fast_data() if rust_model is not None and include_rust_data else None
        kots_fast = Kots.from_json_data(model_data, order=3) if include_kots_fast else None
        q = rng.standard_normal(pin_model.nq)
        v = rng.standard_normal(pin_model.nv)
        a = rng.standard_normal(pin_model.nv)
        print(
            f"--- dof={fast_model.dof} pin_nq={pin_model.nq} "
            f"links={fast_model.link_num} joints={fast_model.joint_num} ---",
            flush=True,
        )
        for op_name in selected_ops:
            pin_stats = measure(_pinocchio_runner(pin, pin_model, q, v, a, op_name), repeats=repeat, warmup=warmup)
            fast_stats = measure(_fast_runner(fast_model, q, v, a, op_name), repeats=repeat, warmup=warmup)
            rust_stats = (
                measure(_rust_runner(rust_model, q, v, a, op_name), repeats=repeat, warmup=warmup)
                if rust_model is not None
                else None
            )
            rust_data_stats = (
                measure(_rust_data_runner(rust_data, q, v, a, op_name), repeats=repeat, warmup=warmup)
                if rust_data is not None
                else None
            )
            kots_fast_stats = (
                measure(_kots_fast_runner(kots_fast, q, v, a, op_name), repeats=repeat, warmup=warmup)
                if kots_fast is not None
                else None
            )
            _print_result(dof, op_name, pin_stats, fast_stats, rust_stats, rust_data_stats, kots_fast_stats)
            rows.append(
                {
                    "op": op_name,
                    "model_kind": model_kind,
                    "dof": dof,
                    "pinocchio_nq": pin_model.nq,
                    "pinocchio_nv": pin_model.nv,
                    "fast_mean_ms": fast_stats["mean_ms"],
                    "fast_std_ms": fast_stats["std_ms"],
                    "fast_min_ms": fast_stats["min_ms"],
                    "pinocchio_mean_ms": pin_stats["mean_ms"],
                    "pinocchio_std_ms": pin_stats["std_ms"],
                    "pinocchio_min_ms": pin_stats["min_ms"],
                    "rust_mean_ms": rust_stats["mean_ms"] if rust_stats is not None else "",
                    "rust_std_ms": rust_stats["std_ms"] if rust_stats is not None else "",
                    "rust_min_ms": rust_stats["min_ms"] if rust_stats is not None else "",
                    "rust_data_mean_ms": rust_data_stats["mean_ms"] if rust_data_stats is not None else "",
                    "rust_data_std_ms": rust_data_stats["std_ms"] if rust_data_stats is not None else "",
                    "rust_data_min_ms": rust_data_stats["min_ms"] if rust_data_stats is not None else "",
                    "kots_fast_mean_ms": kots_fast_stats["mean_ms"] if kots_fast_stats is not None else "",
                    "kots_fast_std_ms": kots_fast_stats["std_ms"] if kots_fast_stats is not None else "",
                    "kots_fast_min_ms": kots_fast_stats["min_ms"] if kots_fast_stats is not None else "",
                    "ratio_fast_over_pinocchio": (
                        fast_stats["mean_ms"] / pin_stats["mean_ms"] if pin_stats["mean_ms"] > 0 else float("inf")
                    ),
                    "ratio_rust_over_pinocchio": (
                        rust_stats["mean_ms"] / pin_stats["mean_ms"]
                        if rust_stats is not None and pin_stats["mean_ms"] > 0
                        else ""
                    ),
                    "ratio_kots_fast_over_pinocchio": (
                        kots_fast_stats["mean_ms"] / pin_stats["mean_ms"]
                        if kots_fast_stats is not None and pin_stats["mean_ms"] > 0
                        else ""
                    ),
                    "ratio_rust_data_over_pinocchio": (
                        rust_data_stats["mean_ms"] / pin_stats["mean_ms"]
                        if rust_data_stats is not None and pin_stats["mean_ms"] > 0
                        else ""
                    ),
                    "overhead_kots_fast_over_rust": (
                        kots_fast_stats["mean_ms"] / rust_stats["mean_ms"]
                        if kots_fast_stats is not None and rust_stats is not None and rust_stats["mean_ms"] > 0
                        else ""
                    ),
                    "speedup_fast_over_rust": (
                        fast_stats["mean_ms"] / rust_stats["mean_ms"]
                        if rust_stats is not None and rust_stats["mean_ms"] > 0
                        else ""
                    ),
                    "note": "minimal q/v/a benchmark; not RoboKots public API semantics",
                }
            )
        print(flush=True)

    write_csv(csv_path, rows)
    print(f"wrote csv  : {csv_path}", flush=True)


if __name__ == "__main__":
    main()
