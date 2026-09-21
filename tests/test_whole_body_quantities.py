import numpy as np

from robokots import outward as outward_api
from robokots.core.state.spec import StateType
from robokots.kots import Kots


MODEL_PATH = "tests/test_model/sample_robot.json"


def _make_kots(q, qdot):
    kots = Kots.from_json_file(MODEL_PATH, order=3)
    motion = np.zeros((kots.dof(), 3))
    motion[:, 0] = q
    motion[:, 1] = qdot
    kots.import_motion_array(motion)
    return kots


def test_center_of_mass_and_jacobian_match_central_difference():
    q = np.array([0.2, -0.3, 0.4])
    qdot = np.array([0.1, 0.2, -0.1])
    kots = _make_kots(q, qdot)

    jacobian = kots.center_of_mass_jacobian()
    assert jacobian.shape == (3, kots.dof())

    eps = 2e-6
    for index in range(kots.dof()):
        delta = np.zeros(kots.dof())
        delta[index] = eps
        expected = (_make_kots(q + delta, qdot).center_of_mass()
                    - _make_kots(q - delta, qdot).center_of_mass()) / (2.0 * eps)
        np.testing.assert_allclose(jacobian[:, index], expected, atol=1e-6, rtol=1e-6)


def test_angular_momentum_about_com_and_jacobian_match_central_difference():
    q = np.array([0.2, -0.3, 0.4])
    qdot = np.array([0.1, 0.2, -0.1])
    kots = _make_kots(q, qdot)

    state = outward_api.build_dynamics_outward_state(kots.robot_, kots.motion(order=2), dynamics_order=0)
    total_momentum = sum(
        (outward_api.get_value(kots.robot_, state, StateType("link", link.name, "momentum", "world"))
         for link in kots.robot_.links),
        start=np.zeros(6),
    )
    np.testing.assert_allclose(kots.angular_momentum(), total_momentum[:3])
    np.testing.assert_allclose(
        kots.angular_momentum("com"),
        total_momentum[:3] - np.cross(kots.center_of_mass(), total_momentum[3:]),
    )

    jacobian = kots.angular_momentum_jacobian("com")
    assert jacobian.shape == (3, 2 * kots.dof())

    eps = 2e-6
    for index in range(kots.dof()):
        delta = np.zeros(kots.dof())
        delta[index] = eps
        expected_q = (_make_kots(q + delta, qdot).angular_momentum("com")
                      - _make_kots(q - delta, qdot).angular_momentum("com")) / (2.0 * eps)
        np.testing.assert_allclose(jacobian[:, 2 * index], expected_q, atol=1e-5, rtol=1e-5)

        expected_v = (_make_kots(q, qdot + delta).angular_momentum("com")
                      - _make_kots(q, qdot - delta).angular_momentum("com")) / (2.0 * eps)
        np.testing.assert_allclose(jacobian[:, 2 * index + 1], expected_v, atol=1e-5, rtol=1e-5)
