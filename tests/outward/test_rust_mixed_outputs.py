"""Mixed local spatial/dynamics outputs use a single Rust derivative path."""
from pathlib import Path

import numpy as np
import pytest

from robokots import outward
from robokots.kots import Kots, StateType

MODEL = Path(__file__).resolve().parents[1] / 'test_model' / 'branched_fixed.urdf'


def mixed_states(order):
    # Preserve ordering, repeated outputs, both owners, and explicit local frames.
    spatial = ('vel', 'acc', 'jerk', 'snap', 'crackle', 'pop')
    states = [StateType('link', 'a_tip', spatial[order - 2], 'local'),
              StateType('joint', 'b_payload_fixed', 'acc'),
              StateType('joint', 'a_shoulder', 'vel', 'local'),
              StateType('link', 'b_payload', 'vel')]
    states += [StateType('link', 'a_tip', f'force_diff{order-3}' if order > 3 else 'force', 'local'),
               StateType('joint', 'b_payload_fixed', f'momentum_diff{order-2}'),
               StateType('total_joint', 'total_joint', 'torque'),
               states[0], states[2]]
    return states


@pytest.mark.parametrize('order,shape,zero_pose,gravity', [
    (3, (), False, [0., 0., 0.]),
    (4, (2, 1), False, [.2, -.3, -9.81]),
    (5, (), True, [.2, -.3, -9.81]),
    (5, (2, 1), False, [.2, -.3, -9.81]),
    (7, (), False, [.2, -.3, -9.81]),
])
def test_mixed_rust_dense_and_products(monkeypatch, order, shape, zero_pose, gravity):
    pytest.importorskip('robokots._rust')
    rust = Kots.from_urdf_file(str(MODEL), order=order)
    reference = Kots.from_urdf_file(str(MODEL), order=order)
    rng = np.random.default_rng(728)
    x = rng.normal(scale=.2, size=shape + (rust.dof()*order,))
    if zero_pose:
        x[..., 0::order] = 0
    states = mixed_states(order)
    for k, backend in ((rust, 'rust'), (reference, 'numpy')):
        k.import_motions(x)
        k.dynamics(backend=backend, gravity=gravity)
    def value(motion):
        reference.import_motions(motion)
        reference.dynamics(backend='numpy', gravity=gravity)
        return np.concatenate([np.asarray(reference.state_info(s)).reshape(shape + (-1,))
                               for s in reference._state_type_list(states)], axis=-1)
    h = 1e-6
    basis = np.eye(x.shape[-1])
    # Joint spatial selections are relative CMTM motion. The existing mixed
    # NumPy analytic assembly selects link rows for these, so use state values.
    expected = np.stack([(value(x+h*e)-value(x-h*e))/(2*h) for e in basis], axis=-1)
    reference.import_motions(x)
    reference.dynamics(backend='numpy', gravity=gravity)
    def forbidden(*args, **kwargs):
        raise AssertionError('mixed request left the unified Rust path')
    for name in ('outward_jacobian', 'outward_jacobian_matvec', 'outward_jacobian_matmul_rhs', 'outward_jacobian_transpose_matvec'):
        monkeypatch.setattr(outward, name, forbidden)
    monkeypatch.setattr(rust, '_rust_cmtm_jacobian_transpose_apply', forbidden)
    actual = rust.jacobian(states)
    np.testing.assert_allclose(actual, expected, atol=5e-7, rtol=5e-7)
    assert np.isfinite(actual).all()
    parts = rust.jacobian(states, list_output=True)
    np.testing.assert_allclose(np.concatenate(parts, axis=-2), expected, atol=5e-7, rtol=5e-7)
    monkeypatch.setattr(rust, '_jacobian_from_state', forbidden)
    v = rng.normal(size=shape + (expected.shape[-1], 2))
    w = rng.normal(size=shape + (expected.shape[-2], 2))
    jvp = rust.jacobian_mul(states, v)
    vjp = rust.jacobian_transpose_mul(states, w)
    np.testing.assert_allclose(jvp, expected @ v, atol=5e-7, rtol=5e-7)
    np.testing.assert_allclose(vjp, np.swapaxes(expected, -1, -2) @ w, atol=5e-7, rtol=5e-7)
    np.testing.assert_allclose(rust.jacobian_mul(states, v[..., 0]), jvp[..., 0], atol=5e-7, rtol=5e-7)
    np.testing.assert_allclose(rust.jacobian_transpose_mul(states, w[..., 0]), vjp[..., 0], atol=5e-7, rtol=5e-7)
    broadcast_v = rng.normal(size=expected.shape[-1])
    np.testing.assert_allclose(rust.jacobian_mul(states, broadcast_v), (expected @ broadcast_v[..., None])[..., 0], atol=5e-7, rtol=5e-7)
    pieces = rust.jacobian_mul(states, v, list_output=True)
    np.testing.assert_allclose(np.concatenate(pieces, axis=-2), jvp, atol=5e-7, rtol=5e-7)
    finite_difference = (value(x + h*v[..., 0]) - value(x - h*v[..., 0])) / (2*h)
    np.testing.assert_allclose(jvp[..., 0], finite_difference, atol=5e-7, rtol=5e-7)
    np.testing.assert_allclose(np.sum(v[..., 0]*vjp[..., 0], axis=-1),
                               np.sum(finite_difference*w[..., 0], axis=-1), atol=2e-6, rtol=5e-7)


@pytest.mark.parametrize('method', ['dynamics_selected_tangent_batch', 'dynamics_selected_transpose_batch'])
def test_spatial_descriptors_reject_world_and_excess_order(method):
    pytest.importorskip('robokots._rust')
    k = Kots.from_urdf_file(str(MODEL), order=3)
    fn = getattr(k._rust_compiled_robot(), method)
    for descriptor in [(0, 0, 3, 0, True), (1, 0, 3, 2, False)]:
        with pytest.raises(ValueError, match='invalid dynamics output'):
            fn(np.zeros((1, k.dof()*3)), np.zeros((1, k.dof()*3, 1)), [descriptor], 1)


def test_selected_dispatch_keeps_existing_boundaries():
    pytest.importorskip('robokots._rust')
    k = Kots.from_urdf_file(str(MODEL), order=4)
    k.dynamics(backend='rust')
    force = StateType('link', 'a_tip', 'force')
    for state in (StateType('link', 'a_tip', 'vel', 'world'),
                  StateType('link', 'a_tip', 'frame'),
                  StateType('joint', 'a_shoulder', 'jerk')):
        assert k._rust_selected_dynamics_specs([state, force], 4) is None
    for state in (StateType('link', 'a_tip', 'acc'),
                  StateType('joint', 'a_shoulder', 'torque')):
        assert k._rust_selected_dynamics_specs([state], 4) is None
