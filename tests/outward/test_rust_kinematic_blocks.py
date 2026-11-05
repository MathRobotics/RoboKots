"""Ancestor-route derivative blocks: high orders, frames, batches and products."""
from pathlib import Path

import numpy as np
import pytest

from robokots.kots import Kots, StateType

MODEL = Path(__file__).resolve().parents[1] / "test_model/branched_fixed.urdf"


def selections(order):
    keys = ["pos", "rot", "frame"] + ["vel", "acc", "jerk", "snap", "crackle", "pop"][:order-1]
    return [StateType(owner, name, key, frame)
            for owner, name in [("link", "world"), ("link", "a_tip"), ("link", "b_payload"),
                                ("joint", "a_elbow"), ("joint", "b_payload_fixed")]
            for frame in ("local", "world") for key in keys]


@pytest.mark.parametrize("order", [1, 2, 3, 6])
@pytest.mark.parametrize("shape", [(), (2, 3)])
def test_route_blocks_match_numpy_and_direct_products(order, shape, monkeypatch):
    rng = np.random.default_rng(555)
    numpy = Kots.from_urdf_file(str(MODEL), order=order, backend="numpy")
    rust = Kots.from_urdf_file(str(MODEL), order=order, backend="rust")
    motion = rng.normal(scale=.2, size=shape + (rust.dof()*order,))
    states = selections(order)
    states += [states[3], states[-1]]  # Repeated selections preserve output order.
    for k in (numpy, rust):
        k.import_motions(motion)
        k.kinematics()
    expected = numpy.jacobian(states)
    direction = rng.normal(size=shape + (expected.shape[-1], 2))
    cotangent = rng.normal(size=shape + (expected.shape[-2], 2))
    from robokots import outward
    def forbidden(*a, **kw):
        raise AssertionError("No Python fallback or identity seed is allowed")
    monkeypatch.setattr(outward, "outward_jacobian", forbidden)
    monkeypatch.setattr(np, "eye", forbidden)
    actual = rust.jacobian(states)
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)
    pieces = rust.jacobian(states, list_output=True)
    np.testing.assert_allclose(np.concatenate(pieces, axis=-2), actual, atol=2e-11)
    for v, w in [(direction, cotangent), (direction[..., 0], cotangent[..., 0])]:
        matrix = v.ndim == len(shape)+2
        jvp = rust.jacobian_mul(states, v)
        vjp = rust.jacobian_transpose_mul(states, w)
        expected_jvp = actual @ v if matrix else (actual @ v[..., None])[..., 0]
        expected_vjp = actual.swapaxes(-1, -2) @ w if matrix else (actual.swapaxes(-1, -2) @ w[..., None])[..., 0]
        np.testing.assert_allclose(jvp, expected_jvp, atol=2e-11)
        np.testing.assert_allclose(vjp, expected_vjp, atol=2e-11)
    # No dynamics evaluation is required for purely kinematic selections.
    assert rust._rust_selected_workspace_[2].cache_info()[1] == 0


def test_world_route_blocks_match_finite_differences():
    order = 6
    k = Kots.from_urdf_file(str(MODEL), order=order, backend="rust")
    rng = np.random.default_rng(334)
    x = rng.normal(scale=.25, size=k.dof()*order)
    x[::order] = 0.
    states = [StateType("link", "a_tip", "pos", "world"),
              StateType("link", "a_tip", "crackle", "world"),
              StateType("joint", "a_elbow", "jerk", "world"),
              StateType("joint", "b_payload_fixed", "acc", "world")]
    def values(motion):
        k.import_motions(motion)
        k.kinematics()
        return k.state_info_list(states)
    values(x)
    jac = k.jacobian(states)
    h = 1e-6
    numerical = np.column_stack([(values(x+h*d)-values(x-h*d))/(2*h) for d in np.eye(x.size)])
    np.testing.assert_allclose(jac, numerical, atol=2e-8, rtol=2e-7)


def test_route_product_cache_tracks_motion_outputs_and_batch():
    rng = np.random.default_rng(519)
    rust = Kots.from_urdf_file(str(MODEL), order=3, backend="rust")
    numpy = Kots.from_urdf_file(str(MODEL), order=3, backend="numpy")
    states = [StateType("link", "a_tip", "acc", "world"),
              StateType("joint", "a_elbow", "vel", "local")]
    for shape in [(), (), (2, 3), (2, 3), ()]:
        motion = rng.normal(scale=.2, size=shape + (rust.dof()*3,))
        for k in (rust, numpy):
            k.import_motions(motion)
            k.kinematics()
        # Both reordered outputs and new motion must invalidate route coefficients.
        for selected in [states, states[::-1], states]:
            jac = numpy.jacobian(selected)
            v = rng.normal(size=shape + (jac.shape[-1], 2))
            w = rng.normal(size=shape + (jac.shape[-2], 2))
            np.testing.assert_allclose(rust.jacobian_mul(selected, v), jac @ v, atol=2e-11)
            np.testing.assert_allclose(rust.jacobian_transpose_mul(selected, w),
                                       jac.swapaxes(-1, -2) @ w, atol=2e-11)
