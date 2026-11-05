"""Compare a TOML-perturbed model with the nominal two-DoF arm.

Run inside this example directory: uv run python main.py
"""
import argparse
import json
from pathlib import Path

import numpy as np

from robokots import PerturbationSpec, apply_perturbation
from robokots.kots import Kots, StateType


EXAMPLE_DIR = Path(__file__).resolve().parent
MODEL_PATH = EXAMPLE_DIR.parent / "model" / "2dof_arm.json"
# Fixed evaluation inputs, independent of the perturbation seed.
MOTION = np.array([[0.3, 0.2, 0.1], [-0.4, -0.1, 0.2]])
# In-plane component makes gravity torques visible for this planar arm.
GRAVITY = np.array([0.0, -9.81, 0.0])


def evaluate(model):
    model.import_motion_array(MOTION.copy())
    model.kinematics(backend="numpy")
    state = StateType("link", "arm2", "pos", "world")
    return {
        "arm2_origin_world_m": np.asarray(model.state_info(state)),
        "position_jacobian": np.asarray(model.jacobian(state)),
        "joint_torque_Nm": model.inverse_dynamics(
            MOTION[:, 0], MOTION[:, 1], MOTION[:, 2], gravity=GRAVITY, backend="numpy"),
    }


def compare_outputs(actual, expected):
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12,
                                   err_msg=key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXAMPLE_DIR / "perturbation.toml",
                        help="TOML path (explicit relative paths use the current directory)")
    args = parser.parse_args()
    spec = PerturbationSpec.from_toml_file(args.config)
    nominal = Kots.from_json_file(str(MODEL_PATH), backend="numpy")
    original = nominal.robot_.to_dict()
    perturbed, report = apply_perturbation(nominal, spec, return_report=True)
    nominal_values = evaluate(nominal)
    perturbed_values = evaluate(perturbed)

    if nominal.robot_.to_dict() != original:
        raise AssertionError("公称モデルが変更されています")
    print(f"設定: {args.config.resolve()}")
    print(f"モデル: {MODEL_PATH}")
    print(f"seed: {spec.seed}; backend: numpy; gravity: {GRAVITY.tolist()} m/s²")
    print("motion [q, dq, ddq]:", MOTION.tolist())
    print("\nリンク質量 [kg]: 公称 -> 摂動後")
    for before, after in zip(nominal.robot_.links, perturbed.robot_.links):
        print(f"  {before.name}: {before.mass:.8f} -> {after.mass:.8f}")
    print("\n関節原点の距離 [m]: 公称 -> 摂動後（親リンク座標系）")
    for before, after in zip(nominal.robot_.joints, perturbed.robot_.joints):
        print(f"  {before.name}: {np.linalg.norm(before.origin.pos()):.8f}"
              f" -> {np.linalg.norm(after.origin.pos()):.8f}")
    print("\n同じmotionでの計算結果:")
    for key, value in nominal_values.items():
        other = perturbed_values[key]
        if not np.all(np.isfinite(other)):
            raise AssertionError(f"非有限の計算結果: {key}")
        print(f"  {key}")
        print("    公称:", np.array2string(value, precision=8))
        print("    摂動:", np.array2string(other, precision=8))
        print(f"    最大絶対差: {np.max(np.abs(other - value)):.6e}")

    zero = apply_perturbation(nominal, PerturbationSpec(seed=42))
    compare_outputs(evaluate(zero), nominal_values)
    print("\nPASS: 公称モデル非破壊、ゼロ摂動で計算結果一致")
    if spec.seed is not None:
        repeated = apply_perturbation(nominal, spec)
        if repeated.robot_.to_dict() != perturbed.robot_.to_dict():
            raise AssertionError("同じseedでモデルを再現できません")
        compare_outputs(evaluate(repeated), perturbed_values)
        print("PASS: 同じseedでモデル・計算結果を再現")
    else:
        print("SKIP: seed未指定のため同一seedでの再現確認")

    # JSON serialization also checks that report data can be stored for replay.
    restored = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    replay = Kots.from_json_data(restored["model_data"], backend="numpy")
    compare_outputs(evaluate(replay), perturbed_values)
    print("PASS: レポートのJSON往復からモデル・計算結果を再現")
    print("\n明示ルールのサンプル:", json.dumps(restored["rule_changes"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
