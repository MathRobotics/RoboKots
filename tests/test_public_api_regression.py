"""Golden public-API regression checks for facade refactors.

Values here are intentionally literal, rather than being computed through a
second RoboKots path: changes to cache/backend dispatch must preserve these
observable results for the fixed sample model and input below.
"""
from pathlib import Path

import numpy as np
import pytest

from robokots.core.state import StateType
from robokots.kots import Kots


MODEL_PATH = Path(__file__).with_name("test_model") / "sample_robot.json"
GRAVITY = np.array([0.2, -0.3, -9.81])
MOTION = np.array([
    0.11, -0.23, 0.37, -0.19, 0.29,
    -0.41, 0.43, -0.31, 0.17, -0.07,
    0.13, -0.17, 0.05, -0.11, 0.19,
])
GOLDEN = {
    "arm3_pos": np.array([3.1350344929064677, 2.7635506812581383, 0.0]),
    "arm3_vel": np.array([0.0, 0.0, 0.03, 0.11221899976220485, 0.06675825562516549, 0.0]),
    "joint3_torque": np.array([0.0022]),
    "joint2_torque_d1": np.array([-0.372065118314642]),
    "joint3_torque_jacobian": np.array([[
        0.0, 0.0, 0.02, 0.0, 0.0, 0.02, 0.0, 0.0, 0.02,
    ]]),
    "inverse_torque": np.array([19.299102449892413, 6.321984036062365, 0.0022]),
    "forward_acceleration": np.array([0.37, -0.31, 0.05]),
}


@pytest.mark.parametrize("backend", ["numpy", "rust"])
def test_public_state_api_matches_golden_fixture(backend):
    if backend == "rust":
        pytest.importorskip("robokots._rust")
    kots = Kots.from_json_file(str(MODEL_PATH), order=5)
    kots.import_motions(MOTION)
    kots.dynamics(order=5, backend=backend, gravity=GRAVITY, materialize_dict=False)

    np.testing.assert_allclose(kots.state_info(StateType("link", "arm3", "pos")), GOLDEN["arm3_pos"], atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(kots.state_info(StateType("link", "arm3", "vel")), GOLDEN["arm3_vel"], atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(kots.state_info(StateType("joint", "joint3", "torque")), GOLDEN["joint3_torque"], atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(kots.state_info(StateType("joint", "joint2", "torque_diff1")), GOLDEN["joint2_torque_d1"], atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(kots.jacobian(StateType("joint", "joint3", "torque")), GOLDEN["joint3_torque_jacobian"], atol=2e-11, rtol=2e-11)


def test_public_inward_api_matches_golden_fixture():
    pytest.importorskip("robokots._rust")
    kots = Kots.from_json_file(str(MODEL_PATH), order=5)
    q, v, acceleration = MOTION[0::5], MOTION[1::5], MOTION[2::5]
    torque = kots.inverse_dynamics(q, v, acceleration, gravity=GRAVITY)
    np.testing.assert_allclose(torque, GOLDEN["inverse_torque"], atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(kots.forward_dynamics(q, v, torque, gravity=GRAVITY), GOLDEN["forward_acceleration"], atol=2e-11, rtol=2e-11)
    cache = kots.create_inward_cache().prepare(q, v, GRAVITY)
    np.testing.assert_allclose(cache.forward_dynamics(torque), GOLDEN["forward_acceleration"], atol=2e-11, rtol=2e-11)
