from pathlib import Path

import numpy as np
import pytest

from robokots import outward
from robokots.kots import Kots, StateType


MODEL = Path(__file__).resolve().parents[1] / "test_model/branched_fixed.urdf"


def make_kots(backend, batch_shape):
    if backend == "rust":
        pytest.importorskip("robokots._rust", exc_type=ImportError)
    kots = Kots.from_urdf_file(str(MODEL), order=3, backend=backend)
    motion = np.random.default_rng(912).normal(scale=.2, size=batch_shape + (kots.dof() * 3,))
    kots.import_motions(motion)
    kots.dynamics(gravity=[.2, -.3, -9.81])
    return kots


@pytest.mark.parametrize("backend", ["numpy", "rust"])
@pytest.mark.parametrize("batch_shape", [(), (2, 3)])
@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
@pytest.mark.parametrize("quantity", ["energy", "torque", "velocity"])
def test_real_rhs_products_match_dense(backend, batch_shape, matrix, dtype, quantity):
    kots = make_kots(backend, batch_shape)
    state = {
        "energy": StateType("total_body", "total_body", "kinetic_energy"),
        "torque": StateType("total_joint", "total_joint", "torque"),
        "velocity": StateType("link", "a_tip", "vel"),
    }[quantity]
    jacobian = kots.jacobian(state)
    # Shared inputs exercise broadcasting across multiple motion batch axes.
    columns = (2,) if matrix else ()
    tangent = np.arange(1, np.prod((jacobian.shape[-1],) + columns) + 1, dtype=dtype).reshape((jacobian.shape[-1],) + columns)
    cotangent = np.arange(1, np.prod((jacobian.shape[-2],) + columns) + 1, dtype=dtype).reshape((jacobian.shape[-2],) + columns)
    expected_jvp = jacobian @ tangent if matrix else (jacobian @ tangent[..., None])[..., 0]
    transpose = jacobian.swapaxes(-1, -2)
    expected_vjp = transpose @ cotangent if matrix else (transpose @ cotangent[..., None])[..., 0]
    np.testing.assert_allclose(kots.jacobian_mul(state, tangent), expected_jvp, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(kots.jacobian_transpose_mul(state, cotangent), expected_vjp, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(kots.jacobian_transpose_mul_many([(state, cotangent)]), expected_vjp,
                               atol=1e-10, rtol=1e-10)
    if quantity == "energy":
        np.testing.assert_allclose(kots.kinetic_energy_jacobian_mul(tangent), expected_jvp, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(kots.kinetic_energy_jacobian_transpose_mul(cotangent), expected_vjp,
                                   atol=1e-10, rtol=1e-10)
    elif backend == "numpy" and not batch_shape:
        states = kots._state_type_list(state)
        order = StateType.max_time_order(states)
        apply = outward.outward_jacobian_matmul_rhs if matrix else outward.outward_jacobian_matvec
        np.testing.assert_allclose(apply(kots.robot_, kots.outward_state_, states, tangent, order), expected_jvp,
                                   atol=1e-10, rtol=1e-10)
        transpose_apply = outward.outward_jacobian_transpose_matmul_rhs if matrix else outward.outward_jacobian_transpose_matvec
        np.testing.assert_allclose(transpose_apply(
            kots.robot_, kots.outward_state_, states, cotangent, order), expected_vjp,
            atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("backend", ["numpy", "rust"])
@pytest.mark.parametrize("batch_shape", [(), (2, 3)])
@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("shared_rhs", [False, True])
def test_squared_power_terms_preserve_batch_axes(backend, batch_shape, matrix, shared_rhs):
    kots = make_kots(backend, batch_shape)
    state = StateType("total_joint", "total_joint", "torque")
    torque = kots.state_info(state)
    velocity = kots.motion(2).reshape(batch_shape + (kots.dof(), 2))[..., :, 1]
    power = np.sum(torque * velocity, axis=-1)
    shape = (() if shared_rhs else batch_shape) + (1,) + ((2,) if matrix else ())
    rhs = np.arange(1, np.prod(shape) + 1).reshape(shape)
    result = kots.squared_power_torque_vjp_terms(state, rhs)
    rhs = np.broadcast_to(rhs, batch_shape + (1,) + ((2,) if matrix else ()))
    scale = 2 * power[..., None] * rhs[..., 0, :] if matrix else 2 * power * rhs[..., 0]
    expected_torque = velocity[..., :, None] * scale[..., None, :] if matrix else velocity * scale[..., None]
    expected_velocity = torque[..., :, None] * scale[..., None, :] if matrix else torque * scale[..., None]
    np.testing.assert_allclose(result["power"], power)
    assert result["torque_request"][1].shape == expected_torque.shape
    np.testing.assert_allclose(result["torque_request"][1], expected_torque)
    expected_motion = np.zeros(batch_shape + (kots.dof(), 2) + ((2,) if matrix else ()))
    if matrix:
        expected_motion[..., :, 1, :] = expected_velocity
    else:
        expected_motion[..., :, 1] = expected_velocity
    np.testing.assert_allclose(result["motion_vjp_order2"], expected_motion.reshape(
        batch_shape + (kots.dof() * 2,) + ((2,) if matrix else ())))


@pytest.mark.parametrize("batch_shape", [(), (2, 3)])
@pytest.mark.parametrize("matrix", [False, True])
def test_numerical_jvp_list_output_splits_state_rows(batch_shape, matrix):
    kots = make_kots("numpy", batch_shape)
    states = [StateType("link", "a_tip", "pos"), StateType("joint", "a_shoulder", "torque")]
    jacobian = kots.jacobian(states, numerical=True, eps=1e-6)
    shape = (jacobian.shape[-1],) + ((2,) if matrix else ())
    rhs = np.ones(shape)
    expected = jacobian @ rhs if matrix else (jacobian @ rhs[..., None])[..., 0]
    parts = kots.jacobian_mul(states, rhs, numerical=True, list_output=True, eps=1e-6)
    assert len(parts) == 2
    assert parts[0].shape == batch_shape + (3,) + ((2,) if matrix else ())
    assert parts[1].shape == batch_shape + (1,) + ((2,) if matrix else ())
    np.testing.assert_allclose(np.concatenate(parts, axis=-2 if matrix else -1), expected)
