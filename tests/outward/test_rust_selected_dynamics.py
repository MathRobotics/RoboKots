"""Selected Rust outputs reuse the torque recurrence without dense products."""
from pathlib import Path

import numpy as np
import pytest

from robokots import outward as outward_api
from robokots.kots import Kots, StateType

MODELS = Path(__file__).resolve().parents[1] / "test_model"


def make_kots(order):
    pytest.importorskip("robokots._rust")
    return Kots.from_urdf_file(str(MODELS / "branched_fixed.urdf"), order=order)


def selected_states(kots, order):
    states = []
    for owner, names in (("link", kots.link_name_list()), ("joint", kots.joint_name_list())):
        for name in names:
            for family, base in (("momentum", 2), ("force", 3)):
                for n in range(order - base + 1):
                    key = family if n == 0 else f"{family}_diff{n}"
                    for frame in (None, "world"):
                        states.append(StateType(owner, name, key, frame))
    states.extend([
        StateType("total_joint", "total_joint", "torque"),
        StateType("total_joint", "total_joint", f"torque_diff{order - 3}" if order > 3 else "torque"),
        # Repeated rows and explicit local frame must accumulate in the VJP.
        StateType("link", "a_tip", "force", "world"),
        StateType("link", "a_tip", "force", "local"),
    ])
    return states


@pytest.mark.parametrize("order,batch_shape,zero_pose,gravity", [
    (3, (), False, [0, 0, 0]),
    (3, (2, 1), False, [0.3, -0.4, -9.81]),
    (5, (), True, [0.3, -0.4, -9.81]),
    (5, (2, 1), False, [0.3, -0.4, -9.81]),
])
def test_selected_dynamics_jacobian_and_direct_products(monkeypatch, order, batch_shape, zero_pose, gravity):
    rust = make_kots(order)
    reference = make_kots(order)
    rng = np.random.default_rng(493)
    motion = rng.normal(scale=0.3, size=batch_shape + (rust.dof() * order,))
    if zero_pose:
        motion[..., 0::order] = 0
    states = selected_states(rust, order)
    for kots, backend in ((reference, "numpy"), (rust, "rust")):
        kots.import_motions(motion)
        kots.dynamics(backend=backend, gravity=gravity)
    expected = reference.jacobian(states)

    def forbidden(*args, **kwargs):
        raise AssertionError("selected dynamics must not use Python derivatives or a dense product")

    for name in ("outward_jacobian", "outward_jacobian_matvec", "outward_jacobian_matmul_rhs", "outward_jacobian_transpose_matvec"):
        monkeypatch.setattr(outward_api, name, forbidden)
    actual = rust.jacobian(states)
    np.testing.assert_allclose(actual, expected, atol=3e-9, rtol=3e-9)
    parts = rust.jacobian(states, list_output=True)
    np.testing.assert_allclose(np.concatenate(parts, axis=-2), expected, atol=3e-9, rtol=3e-9)
    assert np.all(np.isfinite(actual))
    monkeypatch.setattr(rust, "_jacobian_from_state", forbidden)
    input_dim, output_dim = expected.shape[-1], expected.shape[-2]
    directions = rng.normal(size=batch_shape + (input_dim, 2))
    weights = rng.normal(size=batch_shape + (output_dim, 2))
    jvp = rust.jacobian_mul(states, directions)
    vjp = rust.jacobian_transpose_mul(states, weights)
    np.testing.assert_allclose(jvp, expected @ directions, atol=3e-9, rtol=3e-9)
    np.testing.assert_allclose(vjp, np.swapaxes(expected, -1, -2) @ weights, atol=3e-9, rtol=3e-9)
    # Vector RHS, including a broadcast direction across multiple batch axes.
    direction = rng.normal(size=input_dim)
    np.testing.assert_allclose(rust.jacobian_mul(states, direction), (expected @ direction[..., None])[..., 0], atol=3e-9, rtol=3e-9)
    np.testing.assert_allclose(rust.jacobian_transpose_mul(states, weights[..., 0]), vjp[..., 0], atol=3e-9, rtol=3e-9)
    parts = rust.jacobian_mul(states, directions, list_output=True)
    np.testing.assert_allclose(np.concatenate(parts, axis=-2), jvp, atol=3e-9, rtol=3e-9)
    # Independent central difference of NumPy state values, including dX * F.
    def value(x):
        reference.import_motions(x)
        reference.dynamics(backend="numpy", gravity=gravity)
        return np.concatenate([np.asarray(reference.state_info(st)).reshape(batch_shape + (-1,)) for st in reference._state_type_list(states)], axis=-1)
    eps = 1e-6
    difference = (value(motion + eps * directions[..., 0]) - value(motion - eps * directions[..., 0])) / (2 * eps)
    np.testing.assert_allclose(jvp[..., 0], difference, atol=5e-7, rtol=5e-7)
    np.testing.assert_allclose(
        np.sum(directions[..., 0] * vjp[..., 0], axis=-1),
        np.sum(difference * weights[..., 0], axis=-1), atol=2e-6, rtol=5e-7,
    )


def test_selected_world_force_matches_jax_and_fused_vjp(monkeypatch):
    kots = make_kots(5)
    rng = np.random.default_rng(194)
    kots.import_motions(rng.normal(scale=0.3, size=kots.dof() * 5))
    kots.dynamics(backend="rust", gravity=[0.3, -0.4, -9.81])
    states = [
        StateType("joint", "a_shoulder", "force_diff2", "world"),
        StateType("link", "a_tip", "force_diff1", "world"),
        StateType("joint", "b_payload_fixed", "momentum_diff2", "local"),
        StateType("total_joint", "total_joint", "torque_diff2"),
    ]
    expected = kots.jacobian_autodiff(states)
    np.testing.assert_allclose(kots.jacobian(states), expected, atol=2e-9, rtol=2e-9)
    weights = rng.normal(size=(expected.shape[0], 2))
    def forbidden(*args, **kwargs):
        raise AssertionError("dense fallback")
    monkeypatch.setattr(kots, "_jacobian_from_state", forbidden)
    fused = kots.jacobian_transpose_mul_many([
        (states[:2], weights[:12]), (states[2:], weights[12:]),
    ])
    np.testing.assert_allclose(fused, expected.T @ weights, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("kernel", ["dynamics_selected_tangent_batch", "dynamics_selected_transpose_batch"])
def test_selected_dynamics_rejects_invalid_output_descriptors(kernel):
    kots = make_kots(3)
    robot = kots._rust_compiled_robot()
    method = getattr(robot, kernel)
    motion = np.zeros((1, kots.dof() * 3))
    rhs = np.zeros((1, kots.dof() * 3, 1))
    for output in [(2, 0, 1, 0, False), (0, 999, 1, 0, False), (0, 0, 4, 0, False), (0, 0, 1, 1, False), (1, 0, 2, 0, True)]:
        with pytest.raises(ValueError, match="invalid dynamics output"):
            method(motion, rhs, [output], 1)
    with pytest.raises(ValueError, match="shape"):
        method(motion, np.zeros((2, 6, 1)), [(0, 0, 1, 0, False)], 1)
    with pytest.raises(ValueError, match="dynamics_order"):
        method(motion, rhs, [(0, 0, 0, 0, False)], 0)
