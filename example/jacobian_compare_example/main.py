from __future__ import annotations

from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import robokots.core.models.kinematics.kinematics_jax as kinematics_jax

from robokots.core.state import StateType
from robokots.kots import Kots


DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "model" / "sample_robot.json"

CONFIG = {
    "model": DEFAULT_MODEL,
    "order": 5,
    "seed": 0,
    "target_link": None,
}

COMPARE_SPECS = (
    ("vel", 2, kinematics_jax.forward_kinematics_vel),
    ("acc", 3, kinematics_jax.forward_kinematics_acc),
    ("jerk", 4, kinematics_jax.forward_kinematics_jerk)
)


def _pair_metrics(lhs: np.ndarray, rhs: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(lhs) - np.asarray(rhs)
    return float(np.max(np.abs(diff))), float(np.linalg.norm(diff))


def _autodiff_link_jacobian(kots_jax: Kots, target_link: str, data_type: str) -> np.ndarray:
    spec_map = {name: (required_order, func) for name, required_order, func in COMPARE_SPECS}
    if data_type not in spec_map:
        raise ValueError(f"Unsupported autodiff data_type: {data_type}")

    required_order, value_func = spec_map[data_type]
    if kots_jax.order() < required_order:
        raise ValueError(
            f"order={kots_jax.order()} is too small for {data_type}. "
            f"Need at least order={required_order}."
        )

    joints = kots_jax.robot_.joints
    target_id = kots_jax.robot_.link(target_link).id
    x0 = jnp.asarray(kots_jax.motion(order=required_order))

    def wrapped(x, _func=value_func, _joints=joints, _target_id=target_id):
        return _func(_joints, x)[_target_id]

    return np.asarray(jax.jacfwd(wrapped)(x0))


def _print_comparison(data_type: str, analytic: np.ndarray, numerical: np.ndarray, autodiff: np.ndarray) -> None:
    an_num = _pair_metrics(analytic, numerical)
    an_ad = _pair_metrics(analytic, autodiff)
    num_ad = _pair_metrics(numerical, autodiff)

    print(f"[{data_type}] shape={analytic.shape}")
    print(f"  analytic vs numerical : max_abs={an_num[0]:.3e} fro={an_num[1]:.3e}")
    print(f"  analytic vs autodiff  : max_abs={an_ad[0]:.3e} fro={an_ad[1]:.3e}")
    print(f"  numerical vs autodiff : max_abs={num_ad[0]:.3e} fro={num_ad[1]:.3e}")


def main() -> None:
    model_path = Path(CONFIG["model"]).resolve()
    order = int(CONFIG["order"])
    seed = int(CONFIG["seed"])

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if order < 2:
        raise ValueError("CONFIG['order'] must be >= 2")

    kots = Kots.from_json_file(str(model_path), order=order)
    kots_jax = Kots.from_json_file(str(model_path), order=order, lib="jax")

    rng = np.random.default_rng(seed)
    motion = rng.standard_normal(kots.order() * kots.dof())
    kots.import_motions(motion)
    kots_jax.import_motions(motion)
    kots.kinematics()

    target_link = CONFIG.get("target_link") or kots.link_name_list()[-1]
    available_specs = [(name, required_order) for name, required_order, _ in COMPARE_SPECS if required_order <= order]
    if not available_specs:
        raise ValueError(f"No comparison target is available for order={order}")

    print("=== Jacobian Comparison Example ===")
    print(f"model      : {model_path}")
    print(f"order      : {order}")
    print(f"dof        : {kots.dof()}")
    print(f"target     : {target_link}")
    print("compare    : analytic / numerical / autodiff(jax)")
    print("note       : manifold-valued frame Jacobian is omitted here; compare vector-valued kinematics only.")
    print()

    for data_type, required_order in available_specs:
        state = StateType("link", target_link, data_type)
        analytic = kots.jacobian(state)
        numerical = kots.jacobian(state, numerical=True)
        autodiff = _autodiff_link_jacobian(kots_jax, target_link, data_type)
        _print_comparison(data_type, analytic, numerical, autodiff)
        print(f"  motion_order={required_order}")
        print()


if __name__ == "__main__":
    main()
