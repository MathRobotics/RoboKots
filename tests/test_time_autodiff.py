from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from developer.benchmarks.jacobian_accuracy_table import make_model, selections
from developer.benchmarks.time_autodiff import make_time_ad_value, make_fk_value, total_time_derivative
from robokots.kots import Kots, StateType


def test_total_derivative_includes_changing_direction():
    jax.config.update("jax_enable_x64", True)
    motion = jnp.asarray([[2., 3., 5., 7.]])
    function = lambda m: m[0, 0] ** 2
    first = total_time_derivative(function)
    second = total_time_derivative(first)
    third = total_time_derivative(second)
    np.testing.assert_allclose(first(motion), 2 * 2 * 3)
    np.testing.assert_allclose(second(motion), 2 * 3**2 + 2 * 2 * 5)
    np.testing.assert_allclose(third(motion), 6 * 3 * 5 + 2 * 2 * 7)


@pytest.mark.parametrize("zero", [False, True])
def test_time_ad_values_all_rows_and_world_frame(zero):
    jax.config.update("jax_enable_x64", True)
    kots = Kots.from_json_data(make_model(2), order=4)
    x = np.zeros(8) if zero else np.random.default_rng(7).normal(scale=0.3, size=8)
    kots.import_motions(x)
    gravity = [0.3, -0.4, -9.81]
    kots.dynamics(gravity=gravity)
    states = selections(kots, 1) + [StateType("link", kots.link_name_list()[-1], "force_diff1", "world")]
    for state in states:
        actual = make_time_ad_value(kots.robot_, state, 4, gravity)(jnp.asarray(x))
        np.testing.assert_allclose(actual, kots.state_info(state), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("data_type,order", [("torque_diff2", 5), ("torque_diff4", 7), ("crackle", 6)])
def test_time_ad_high_derivative_jacobians(data_type, order):
    jax.config.update("jax_enable_x64", True)
    kots = Kots.from_json_data(make_model(2), order=order)
    owner = "joint" if data_type.startswith("torque") else "link"
    name = "joint1" if owner == "joint" else kots.link_name_list()[-1]
    state = StateType(owner, name, data_type)
    gravity = [0.3, -0.4, -9.81]
    function = make_time_ad_value(kots.robot_, state, order, gravity)
    forward = jax.jit(jax.jacfwd(function))
    reverse = jax.jit(jax.jacrev(function))
    for x in (np.zeros(2 * order), np.random.default_rng(13).normal(scale=.2, size=2 * order)):
        kots.import_motions(x)
        kots.dynamics(gravity=gravity)
        reference = kots.jacobian(state)
        np.testing.assert_allclose(forward(x), reference, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(reverse(x), reference, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(forward(x), kots.jacobian(state, numerical=True), atol=2e-5, rtol=2e-5)
    jax.clear_caches()


def test_fk_pose_time_derivatives_and_elementwise_jacobian():
    jax.config.update("jax_enable_x64", True)
    kots = Kots.from_json_data(make_model(2), order=3)
    x = np.random.default_rng(18).normal(scale=.2, size=6)
    name = kots.link_name_list()[-1]
    def reference(x):
        kots.import_motions(x)
        kots.kinematics()
        return np.asarray(kots.state_info(StateType("link", name, "frame")).mat())
    pose = reference(x)
    def hat(vector):
        wx, wy, wz = vector[:3]
        result = np.zeros((4, 4))
        result[:3, :3] = [[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]
        result[:3, 3] = vector[3:]
        return result
    velocity = hat(kots.state_info(StateType("link", name, "vel")))
    acceleration = hat(kots.state_info(StateType("link", name, "acc")))
    np.testing.assert_allclose(make_fk_value(kots.robot_, name, 0, 3)(x), pose, atol=1e-12)
    np.testing.assert_allclose(make_fk_value(kots.robot_, name, 1, 3)(x), pose @ velocity, atol=1e-12)
    np.testing.assert_allclose(make_fk_value(kots.robot_, name, 2, 3)(x), pose @ (velocity @ velocity + acceleration), atol=1e-12)
    function = make_fk_value(kots.robot_, name, 0, 3)
    direction = np.random.default_rng(21).normal(size=6)
    eps = 1e-6
    numerical = (reference(x + eps * direction) - reference(x - eps * direction)) / (2 * eps)
    jac = jax.jacfwd(function)(x)
    assert jac.shape == (4, 4, 6)
    np.testing.assert_allclose(jac, jax.jacrev(function)(x), atol=1e-12)
    np.testing.assert_allclose(jac @ direction, numerical, atol=1e-9)


def test_time_ad_branched_prismatic_and_rejections(tmp_path):
    jax.config.update("jax_enable_x64", True)
    path = tmp_path / "prismatic.urdf"
    source = Path(__file__).parent / "test_model/branched_fixed.urdf"
    path.write_text(source.read_text().replace('name="a_elbow" type="revolute"', 'name="a_elbow" type="prismatic"'))
    kots = Kots.from_urdf_file(str(path), order=3)
    x = np.random.default_rng(8).normal(scale=.3, size=kots.dof() * 3)
    kots.import_motions(x)
    kots.dynamics(gravity=[0, 0, -9.81])
    state = StateType("joint", "a_shoulder", "torque")
    fn = make_time_ad_value(kots.robot_, state, 3, [0, 0, -9.81])
    np.testing.assert_allclose(jax.jacfwd(fn)(x), kots.jacobian(state), atol=1e-10)
    with pytest.raises(ValueError, match="Insufficient"):
        make_time_ad_value(kots.robot_, state, 2)
    with pytest.raises(ValueError, match="motion must"):
        fn(np.zeros(1))
    with pytest.raises(ValueError, match="gravity"):
        make_time_ad_value(kots.robot_, state, 3, [0, 0])
