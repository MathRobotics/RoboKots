from __future__ import annotations

from pathlib import Path

import numpy as np

from robokots.core.state import StateType
from robokots.kots import Kots

from .common import build_model, format_time, measure, select_unit, write_csv
from .pinocchio_compare import _optional_pinocchio, build_pinocchio_model


DEFAULT_CSV_PATH = Path(__file__).resolve().with_name("outward_api_compare_results.csv")

CONFIG = {
    "dof_list": [16, 64],
    "batch_sizes": [1, 8],
    "repeat": 10,
    "warmup": 3,
    "seed": 0,
    "model_kind": "humanoid",
    "ops": ["torque", "torque_diff2", "torque_diff2_jacobian"],
    "validate": True,
    "rtol": 1e-8,
    "atol": 1e-8,
    "csv_path": DEFAULT_CSV_PATH,
}


TORQUE = StateType("total_joint", "total_joint", "torque")
TORQUE_DIFF2 = StateType("total_joint", "total_joint", "torque_diff2")


def _motion_from_qva(q: np.ndarray, v: np.ndarray, a: np.ndarray, order: int) -> np.ndarray:
    motion = np.zeros(q.size * order, dtype=float)
    for i in range(q.size):
        motion[i * order] = q[i]
        motion[i * order + 1] = v[i]
        motion[i * order + 2] = a[i]
    return motion


def _qva_from_motion(motion: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blocks = np.asarray(motion, dtype=float).reshape(-1, order)
    return blocks[:, 0], blocks[:, 1], blocks[:, 2]


def _make_motions(rng: np.random.Generator, dof: int, order: int, batch_size: int) -> np.ndarray:
    motions = rng.standard_normal((batch_size, dof * order))
    if order >= 3:
        for i in range(batch_size):
            q = rng.standard_normal(dof)
            v = rng.standard_normal(dof)
            a = rng.standard_normal(dof)
            motions[i] = _motion_from_qva(q, v, a, order)
    return motions[0] if batch_size == 1 else motions


def _public_runner(kots: Kots, motions: np.ndarray, op_name: str, backend: str | None):
    if op_name == "torque":
        def run():
            kots.import_motions(motions)
            kots.dynamics(order=3, backend=backend, materialize_dict=False)
            return kots.state_info(TORQUE)
        return run

    if op_name == "torque_diff2":
        def run():
            kots.import_motions(motions)
            kots.dynamics(order=5, backend=backend, materialize_dict=False)
            return kots.state_info(TORQUE_DIFF2)
        return run

    if op_name == "torque_diff2_jacobian":
        def run():
            kots.import_motions(motions)
            kots.dynamics(order=5, backend=backend, materialize_dict=False)
            return kots.jacobian(TORQUE_DIFF2)
        return run

    raise ValueError(f"invalid op_name: {op_name}")


def _pinocchio_runner(pin, model, motions: np.ndarray, order: int):
    data = model.createData()
    motion_batch = motions.reshape((1, -1)) if motions.ndim == 1 else motions

    def run():
        out = []
        for motion in motion_batch:
            q, v, a = _qva_from_motion(motion, order)
            out.append(pin.rnea(model, data, q, v, a).copy())
        return out[0] if motions.ndim == 1 else np.stack(out)

    return run


def _validate_outputs(py_run, rust_run, rtol: float, atol: float) -> float:
    expected = py_run()
    actual = rust_run()
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def _print_result(
    op_name: str,
    dof: int,
    batch_size: int,
    py_stats: dict[str, float],
    rust_stats: dict[str, float],
    pin_stats: dict[str, float] | None,
    max_error: float,
) -> None:
    values = [py_stats["mean_ms"], rust_stats["mean_ms"]]
    if pin_stats is not None:
        values.append(pin_stats["mean_ms"])
    unit = select_unit(values)
    speedup = py_stats["mean_ms"] / rust_stats["mean_ms"] if rust_stats["mean_ms"] > 0 else float("inf")
    print(f"{op_name:22s} dof={dof:4d} batch={batch_size:4d}", flush=True)
    print(
        "  robokots  "
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
        f"speedup={speedup:8.2f} "
        f"max_err={max_error:.3e}",
        flush=True,
    )
    if pin_stats is not None:
        ratio = rust_stats["mean_ms"] / pin_stats["mean_ms"] if pin_stats["mean_ms"] > 0 else float("inf")
        print(
            "  pinocchio "
            f"mean={format_time(pin_stats['mean_ms'], unit)} "
            f"std={format_time(pin_stats['std_ms'], unit)} "
            f"min={format_time(pin_stats['min_ms'], unit)} "
            f"ratio(rust/pinocchio)={ratio:8.2f}",
            flush=True,
        )


def _run_one(
    *,
    pin,
    dof: int,
    batch_size: int,
    op_name: str,
    model_kind: str,
    repeat: int,
    warmup: int,
    validate: bool,
    rtol: float,
    atol: float,
    rng: np.random.Generator,
) -> dict[str, object]:
    order = 3 if op_name == "torque" else 5
    model_data = build_model(dof, model_kind)
    py_kots = Kots.from_json_data(model_data, order=order)
    rust_kots = Kots.from_json_data(model_data, order=order)
    motions = _make_motions(rng, py_kots.dof(), order, batch_size)
    py_run = _public_runner(py_kots, motions, op_name, backend=None)
    rust_run = _public_runner(rust_kots, motions, op_name, backend="rust")

    max_error = _validate_outputs(py_run, rust_run, rtol, atol) if validate else float("nan")

    py_stats = measure(py_run, repeats=repeat, warmup=warmup)
    rust_stats = measure(rust_run, repeats=repeat, warmup=warmup)
    pin_stats = None
    if pin is not None and op_name == "torque":
        pin_model = build_pinocchio_model(pin, model_data)
        pin_stats = measure(_pinocchio_runner(pin, pin_model, motions, order), repeats=repeat, warmup=warmup)

    _print_result(op_name, py_kots.dof(), batch_size, py_stats, rust_stats, pin_stats, max_error)

    return {
        "op": op_name,
        "model_kind": model_kind,
        "dof": py_kots.dof(),
        "batch_size": batch_size,
        "order": order,
        "robokots_mean_ms": py_stats["mean_ms"],
        "robokots_std_ms": py_stats["std_ms"],
        "robokots_min_ms": py_stats["min_ms"],
        "rust_mean_ms": rust_stats["mean_ms"],
        "rust_std_ms": rust_stats["std_ms"],
        "rust_min_ms": rust_stats["min_ms"],
        "pinocchio_mean_ms": pin_stats["mean_ms"] if pin_stats is not None else "",
        "pinocchio_std_ms": pin_stats["std_ms"] if pin_stats is not None else "",
        "pinocchio_min_ms": pin_stats["min_ms"] if pin_stats is not None else "",
        "speedup_robokots_over_rust": (
            py_stats["mean_ms"] / rust_stats["mean_ms"] if rust_stats["mean_ms"] > 0 else float("inf")
        ),
        "ratio_rust_over_pinocchio": (
            rust_stats["mean_ms"] / pin_stats["mean_ms"]
            if pin_stats is not None and pin_stats["mean_ms"] > 0
            else ""
        ),
        "max_error": max_error,
    }


def main() -> None:
    pin = _optional_pinocchio()
    dof_list = [int(dof) for dof in CONFIG["dof_list"]]
    batch_sizes = [int(size) for size in CONFIG["batch_sizes"]]
    ops = [str(op) for op in CONFIG["ops"]]
    repeat = int(CONFIG["repeat"])
    warmup = int(CONFIG["warmup"])
    model_kind = str(CONFIG["model_kind"])
    validate = bool(CONFIG.get("validate", True))
    rtol = float(CONFIG.get("rtol", 1e-8))
    atol = float(CONFIG.get("atol", 1e-8))
    csv_path = Path(CONFIG.get("csv_path", DEFAULT_CSV_PATH)).resolve()
    rng = np.random.default_rng(int(CONFIG["seed"]))

    print("=== RoboKots Public Outward API Benchmark ===", flush=True)
    print(f"model_kind : {model_kind}", flush=True)
    print(f"dof_list   : {dof_list}", flush=True)
    print(f"batch_sizes: {batch_sizes}", flush=True)
    print(f"ops        : {', '.join(ops)}", flush=True)
    print(f"warmup     : {warmup}", flush=True)
    print(f"repeat     : {repeat}", flush=True)
    print(f"validate   : {validate}", flush=True)
    print(f"pinocchio  : {'enabled' if pin is not None else 'disabled'}", flush=True)
    print(f"csv_path   : {csv_path}", flush=True)
    print(flush=True)

    rows = []
    for dof in dof_list:
        print(f"--- dof={dof} ---", flush=True)
        for batch_size in batch_sizes:
            for op_name in ops:
                rows.append(
                    _run_one(
                        pin=pin,
                        dof=dof,
                        batch_size=batch_size,
                        op_name=op_name,
                        model_kind=model_kind,
                        repeat=repeat,
                        warmup=warmup,
                        validate=validate,
                        rtol=rtol,
                        atol=atol,
                        rng=rng,
                    )
                )
        print(flush=True)

    write_csv(csv_path, rows)
    print(f"wrote csv  : {csv_path}", flush=True)


if __name__ == "__main__":
    main()
