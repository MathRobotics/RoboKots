from __future__ import annotations

from pathlib import Path

import numpy as np

from robokots.core.state import StateType
from robokots.kots import Kots

from .common import (
    build_model,
    format_time,
    measure,
    robokots_prepare_state,
    robokots_state_types,
    select_unit,
    write_csv,
)


DEFAULT_CSV_PATH = Path(__file__).resolve().with_name("jacobian_transpose_matvec_compare_results.csv")

CONFIG = {
    "dof_list": [16, 32, 64],
    "order": 5,
    "batch_sizes": [1, 8],
    "repeat": 3,
    "warmup": 1,
    "seed": 0,
    "modes": ["kinematics", "dynamics"],
    "target_count": 4,
    "model_kind": "humanoid",
    "validate": True,
    "rtol": 1e-9,
    "atol": 1e-9,
    "csv_path": DEFAULT_CSV_PATH,
}


def _operation_states(kots: Kots, mode: str, target_count: int) -> list[StateType]:
    kinematics_states, dynamics_states, _ = robokots_state_types(kots, target_count)
    if mode == "kinematics":
        return kinematics_states
    if mode == "dynamics":
        return dynamics_states
    raise ValueError("mode must be 'kinematics' or 'dynamics'")


def _explicit_transpose_matvec(kots: Kots, states: list[StateType], vec: np.ndarray) -> np.ndarray:
    jacob = kots.jacobian(states)
    return (np.swapaxes(jacob, -1, -2) @ vec[..., None])[..., 0]


def _print_result(
    mode: str,
    dof: int,
    batch_size: int,
    target_count: int,
    direct_stats: dict[str, float],
    explicit_stats: dict[str, float],
    error_norm: float,
) -> None:
    unit = select_unit(
        [
            direct_stats["mean_ms"],
            direct_stats["std_ms"],
            explicit_stats["mean_ms"],
            explicit_stats["std_ms"],
        ]
    )
    speedup = explicit_stats["mean_ms"] / direct_stats["mean_ms"] if direct_stats["mean_ms"] > 0 else float("inf")
    print(f"{mode:10s} dof={dof:4d} batch={batch_size:4d} targets={target_count:3d}", flush=True)
    print(
        "  direct   "
        f"mean={format_time(direct_stats['mean_ms'], unit)} "
        f"std={format_time(direct_stats['std_ms'], unit)} "
        f"min={format_time(direct_stats['min_ms'], unit)}",
        flush=True,
    )
    print(
        "  explicit "
        f"mean={format_time(explicit_stats['mean_ms'], unit)} "
        f"std={format_time(explicit_stats['std_ms'], unit)} "
        f"min={format_time(explicit_stats['min_ms'], unit)} "
        f"speedup={speedup:8.2f} "
        f"err={error_norm:.3e}",
        flush=True,
    )


def _run_one(
    dof: int,
    mode: str,
    order: int,
    batch_size: int,
    target_count: int,
    model_kind: str,
    rng: np.random.Generator,
    repeat: int,
    warmup: int,
    validate: bool,
    rtol: float,
    atol: float,
) -> dict[str, object]:
    model_data = build_model(dof, model_kind)
    kots = Kots.from_json_data(model_data, order=order)
    states = _operation_states(kots, mode, target_count)
    required_order = StateType.max_time_order(states)
    if required_order > order:
        raise ValueError(f"{mode} requires order {required_order}, but order is {order}")

    if batch_size == 1:
        motion = rng.standard_normal(kots.dof() * order)
    else:
        motion = rng.standard_normal((batch_size, kots.dof() * order))
    kots.import_motions(motion)
    robokots_prepare_state(kots, mode, order)

    jacob = kots.jacobian(states)
    vec = rng.standard_normal(jacob.shape[:-2] + (jacob.shape[-2],))
    direct = kots.jacobian_transpose_mul(states, vec)
    explicit = (np.swapaxes(jacob, -1, -2) @ vec[..., None])[..., 0]
    error_norm = float(np.linalg.norm(direct - explicit))

    if validate:
        np.testing.assert_allclose(direct, explicit, rtol=rtol, atol=atol)

    direct_stats = measure(
        lambda: kots.jacobian_transpose_mul(states, vec),
        repeats=repeat,
        warmup=warmup,
    )
    explicit_stats = measure(
        lambda: _explicit_transpose_matvec(kots, states, vec),
        repeats=repeat,
        warmup=warmup,
    )
    _print_result(mode, kots.dof(), batch_size, len(states), direct_stats, explicit_stats, error_norm)

    return {
        "mode": mode,
        "model_kind": model_kind,
        "dof": kots.dof(),
        "batch_size": batch_size,
        "link_num": kots.robot_.link_num,
        "joint_num": kots.robot_.joint_num,
        "order": order,
        "target_count": len(states),
        "required_order": required_order,
        "jacobian_rows": jacob.shape[-2],
        "jacobian_cols": jacob.shape[-1],
        "direct_mean_ms": direct_stats["mean_ms"],
        "direct_std_ms": direct_stats["std_ms"],
        "direct_min_ms": direct_stats["min_ms"],
        "explicit_mean_ms": explicit_stats["mean_ms"],
        "explicit_std_ms": explicit_stats["std_ms"],
        "explicit_min_ms": explicit_stats["min_ms"],
        "speedup": explicit_stats["mean_ms"] / direct_stats["mean_ms"] if direct_stats["mean_ms"] > 0 else float("inf"),
        "error_norm": error_norm,
    }


def _validate_config(config: dict) -> None:
    if int(config["order"]) < 1:
        raise ValueError("CONFIG['order'] must be >= 1")
    if int(config["repeat"]) < 1:
        raise ValueError("CONFIG['repeat'] must be >= 1")
    if int(config["warmup"]) < 0:
        raise ValueError("CONFIG['warmup'] must be >= 0")
    if any(int(value) < 1 for value in config["batch_sizes"]):
        raise ValueError("CONFIG['batch_sizes'] values must be >= 1")
    invalid_modes = [mode for mode in config["modes"] if mode not in ("kinematics", "dynamics")]
    if invalid_modes:
        raise ValueError(f"Invalid mode(s): {invalid_modes}")


def main() -> None:
    _validate_config(CONFIG)
    rng = np.random.default_rng(int(CONFIG["seed"]))
    rows = []

    for dof in [int(value) for value in CONFIG["dof_list"]]:
        for batch_size in [int(value) for value in CONFIG["batch_sizes"]]:
            for mode in CONFIG["modes"]:
                rows.append(
                    _run_one(
                        dof=dof,
                        mode=mode,
                        order=int(CONFIG["order"]),
                        batch_size=batch_size,
                        target_count=int(CONFIG["target_count"]),
                        model_kind=str(CONFIG["model_kind"]),
                        rng=rng,
                        repeat=int(CONFIG["repeat"]),
                        warmup=int(CONFIG["warmup"]),
                        validate=bool(CONFIG["validate"]),
                        rtol=float(CONFIG["rtol"]),
                        atol=float(CONFIG["atol"]),
                    )
                )

    write_csv(Path(CONFIG["csv_path"]), rows)
    print(f"Wrote {CONFIG['csv_path']}", flush=True)


if __name__ == "__main__":
    main()
