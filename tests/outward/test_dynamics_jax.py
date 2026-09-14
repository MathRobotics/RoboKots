from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from robokots.kots import Kots, StateType
from robokots.outward.diff.dynamics_jax import dynamics_jax, dynamics_state_vector_jax


MODELS = Path(__file__).resolve().parents[1] / "test_model"


def _states(kots, order):
    states = []
    for owner, names in (
        ("link", kots.link_name_list()),
        ("joint", [joint.name for joint in kots.robot_.joints]),
    ):
        for name in names:
            for family, base in (("momentum", 2), ("force", 3)):
                for n in range(order - base + 1):
                    key = family if n == 0 else f"{family}_diff{n}"
                    states.extend(StateType(owner, name, key, frame) for frame in (None, "world"))
    for joint in kots.robot_.joints:
        if joint.dof:
            states.extend(StateType("joint", joint.name, key) for key in ("torque", "torque_diff1", "torque_diff2"))
    return states


@pytest.mark.parametrize("model", ["serial_zero", "branched", "prismatic"])
def test_dynamics_autodiff_values_and_jacobians(model, tmp_path):
    order = 5
    if model == "serial_zero":
        kots = Kots.from_json_file(str(MODELS / "sample_robot.json"), order=order)
        motion = np.zeros(kots.dof() * order)
        gravity = np.zeros(3)
    else:
        path = MODELS / "branched_fixed.urdf"
        if model == "prismatic":
            path = tmp_path / "prismatic.urdf"
            path.write_text((MODELS / "branched_fixed.urdf").read_text().replace(
                'name="a_elbow" type="revolute"', 'name="a_elbow" type="prismatic"'
            ))
        kots = Kots.from_urdf_file(str(path), order=order)
        motion = np.random.default_rng(23).normal(scale=0.4, size=kots.dof() * order)
        gravity = np.array([0.3, -0.4, -9.81])
    kots.import_motions(motion)
    kots.dynamics(backend="numpy", gravity=gravity)
    states = _states(kots, order)
    values = dynamics_state_vector_jax(kots.robot_, jnp.asarray(motion), states, order, gravity)
    expected = np.concatenate([np.asarray(kots.state_info(state)).reshape(-1) for state in states])
    np.testing.assert_allclose(values, expected, atol=2e-10, rtol=2e-10)
    autodiff = kots.jacobian_autodiff(states)
    assert np.all(np.isfinite(autodiff))
    np.testing.assert_allclose(autodiff, kots.jacobian(states), atol=2e-9, rtol=2e-9)

    # Independent directional finite difference of the existing NumPy values.
    direction = np.random.default_rng(45).normal(size=motion.shape)
    eps = 1e-6
    def reference(x):
        kots.import_motions(x)
        kots.dynamics(backend="numpy", gravity=gravity)
        return np.concatenate([np.asarray(kots.state_info(state)).reshape(-1) for state in states])
    difference = (reference(motion + eps * direction) - reference(motion - eps * direction)) / (2 * eps)
    np.testing.assert_allclose(autodiff @ direction, difference, atol=2e-7, rtol=2e-7)


def test_dynamics_autodiff_batch_list_and_momentum_only():
    kots = Kots.from_urdf_file(str(MODELS / "branched_fixed.urdf"), order=3)
    motion = np.random.default_rng(9).normal(scale=0.3, size=(2, 1, kots.dof() * 3))
    kots.import_motions(motion)
    kots.dynamics(backend="numpy", gravity=[0, 0, -9.81])
    state = StateType("total_joint", "total_joint", "torque")
    parts = kots.jacobian_autodiff(state, list_output=True)
    actual = np.concatenate(parts, axis=-2)
    assert actual.shape == (2, 1, kots.dof(), kots.dof() * 3)
    np.testing.assert_allclose(actual, kots.jacobian(state), atol=2e-10, rtol=2e-10)
    momentum = StateType("link", "a_tip", "momentum", "world")
    np.testing.assert_allclose(kots.jacobian_autodiff(momentum), kots.jacobian(momentum), atol=2e-10, rtol=2e-10)
    mixed = [
        StateType("link", "a_tip", "force", "world"),
        StateType("joint", "a_shoulder", "force", "world"),
        state,
    ]
    np.testing.assert_allclose(kots.jacobian_autodiff(mixed), kots.jacobian(mixed), atol=2e-10, rtol=2e-10)


def test_dynamics_jax_jit_forward_and_reverse_ad():
    kots = Kots.from_urdf_file(str(MODELS / "branched_fixed.urdf"), order=3)
    motion = jnp.zeros(kots.dof() * 3)
    states = [StateType("joint", "a_shoulder", "torque")]
    def value(x):
        return dynamics_state_vector_jax(kots.robot_, x, states, 3, [0.3, -0.4, -9.81])
    forward = jax.jit(jax.jacfwd(value))(motion)
    reverse = jax.jit(jax.jacrev(value))(motion)
    assert np.all(np.isfinite(forward))
    np.testing.assert_allclose(forward, reverse, atol=2e-10, rtol=2e-10)


def test_dynamics_jax_rejects_unsupported_inputs():
    soft = Kots.from_json_file(str(MODELS / "soft_rod.json"), order=3)
    with pytest.raises(NotImplementedError, match="rigid links"):
        dynamics_jax(soft.robot_, np.zeros(soft.dof() * 3))
    kots = Kots.from_json_file(str(MODELS / "sample_robot.json"), order=3)
    with pytest.raises(ValueError, match="motions must have shape"):
        dynamics_jax(kots.robot_, np.zeros(1))
    with pytest.raises(ValueError, match="gravity must have shape"):
        dynamics_jax(kots.robot_, np.zeros(kots.dof() * 3), gravity=[0, 0])
    with pytest.raises(NotImplementedError, match="does not support"):
        kots.jacobian_autodiff(StateType("link", kots.link_name_list()[-1], "vel"))
    with pytest.raises(ValueError, match="requires at least one"):
        kots.jacobian_autodiff([])
    kots.robot_.joints[-1].type = "spherical"
    with pytest.raises(NotImplementedError, match="fixed, revolute and prismatic"):
        dynamics_jax(kots.robot_, np.zeros(kots.dof() * 3))
