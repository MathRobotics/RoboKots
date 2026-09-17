from pathlib import Path

import jax
import numpy as np
import pytest

from developer.benchmarks import id_time_autodiff as module
from developer.benchmarks.jacobian_accuracy_table import make_model
from developer.benchmarks.time_autodiff import make_time_ad_value
from robokots.kots import Kots, StateType


@pytest.mark.parametrize("k", [0, 1, 4])
def test_id_time_ad_values_and_jacobians(k, monkeypatch):
    jax.config.update("jax_enable_x64", True)
    order = k + 3
    robot = Kots.from_json_data(make_model(2), order=order)
    state = StateType("joint", "joint1", "torque" + (f"_diff{k}" if k else ""))
    gravity = [0.3, -0.4, -9.81]
    original = module.dynamics_state_vector_jax
    calls = []
    def ordinary_only(model, motion, states, order, gravity):
        assert order == 3 and motion.shape == (model.dof * 3,)
        assert all(s.data_type == "torque" for s in states)
        calls.append(order)
        return original(model, motion, states, order, gravity)
    monkeypatch.setattr(module, "dynamics_state_vector_jax", ordinary_only)
    value = module.make_id_time_ad_value(robot.robot_, state, order, gravity)
    forward = jax.jit(jax.jacfwd(value))
    reverse = jax.jit(jax.jacrev(value))
    for x in (np.zeros(2 * order), np.random.default_rng(35).normal(scale=.3, size=2 * order)):
        robot.import_motions(x)
        robot.dynamics(gravity=gravity)
        np.testing.assert_allclose(value(x), robot.state_info(state), atol=1e-9, rtol=1e-9)
        reference = robot.jacobian(state)
        np.testing.assert_allclose(forward(x), reference, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(reverse(x), reference, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(forward(x), robot.jacobian(state, numerical=True), atol=2e-5, rtol=2e-5)
        if k <= 1:
            fk_value = make_time_ad_value(robot.robot_, state, order, gravity)
            np.testing.assert_allclose(value(x), fk_value(x), atol=1e-9, rtol=1e-9)
    assert calls
    jax.clear_caches()


def test_id_time_ad_branched_prismatic_and_validation(tmp_path):
    jax.config.update("jax_enable_x64", True)
    source = Path(__file__).parent / "test_model/branched_fixed.urdf"
    path = tmp_path / "prismatic.urdf"
    path.write_text(source.read_text().replace('name="a_elbow" type="revolute"', 'name="a_elbow" type="prismatic"'))
    robot = Kots.from_urdf_file(str(path), order=4)
    state = StateType("joint", "a_shoulder", "torque_diff1")
    x = np.random.default_rng(5).normal(scale=.3, size=robot.dof() * 4)
    robot.import_motions(x)
    robot.dynamics(gravity=[0, 0, -9.81])
    value = module.make_id_time_ad_value(robot.robot_, state, 4, [0, 0, -9.81])
    np.testing.assert_allclose(value(x), robot.state_info(state), atol=1e-10)
    np.testing.assert_allclose(jax.jacfwd(value)(x), robot.jacobian(state), atol=1e-10)
    with pytest.raises(ValueError, match="Insufficient"):
        module.make_id_time_ad_value(robot.robot_, state, 3)
    with pytest.raises(ValueError, match="requires one joint"):
        module.make_id_time_ad_value(robot.robot_, StateType("link", "a_tip", "force"), 3)
    with pytest.raises(ValueError, match="motion must"):
        value(np.zeros(1))
