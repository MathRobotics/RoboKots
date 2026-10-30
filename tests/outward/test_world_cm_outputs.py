"""World outputs and derivatives use CM coefficients until the output boundary."""
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from mathrobo import CMVector, CMTM, Factorial, SE3, SE3wrench

from robokots.core.kernels.cmtm_apply import last_ordinary_derivative, world_spatial_value
from robokots.kots import Kots, StateType
from robokots.state_io.dictionary import (
    cmtm_to_state_list, vecs_to_state_dict,
    extract_dict_link_info, extract_dict_joint_info,
)


MODEL = Path(__file__).resolve().parents[1] / "test_model" / "branched_fixed.urdf"


def forbidden(*args, **kwargs):
    raise AssertionError("ordinary series must not be reconstructed")


@pytest.mark.parametrize("dtype", [np.float32, np.int32, np.complex64])
def test_last_derivative_dtype_values_and_ownership(dtype):
    coefficients = np.arange(60).reshape(2, 5, 6).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        coefficients += 1j * coefficients
    expected = CMVector.set_cmvecs(coefficients).vecs()[..., -1, :]
    actual = last_ordinary_derivative(coefficients)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    assert not np.shares_memory(actual, coefficients)


@pytest.mark.parametrize("batch", [(), (2, 1)])
def test_world_motion_uses_owner_coefficients(batch):
    order = 4
    rng = np.random.default_rng(61)
    ordinary = rng.normal(size=batch + (order, 6))
    series = CMVector(ordinary)
    owner = SimpleNamespace(cmvecs=lambda: series, vecs=forbidden)
    frame = CMTM(SE3.eye(), rng.normal(scale=.2, size=(order - 1, 6)))
    factors = np.asarray([math.factorial(i) for i in range(order)])[:, None]
    flat = (ordinary / factors).reshape(batch + (-1,))
    expected = (frame.mat_adj() @ flat[..., None])[..., -6:, 0] * math.factorial(order - 1)
    np.testing.assert_allclose(world_spatial_value(owner, frame, order), expected, atol=1e-13)


@pytest.mark.parametrize("owner_type", ["link", "joint"])
@pytest.mark.parametrize("family", ["momentum", "force"])
@pytest.mark.parametrize("derivative", [0, 2])
@pytest.mark.parametrize("batch", [(), (2, 1)])
def test_dictionary_world_outputs_without_round_trip(owner_type, family, derivative, batch, monkeypatch):
    order = derivative + 1
    rng = np.random.default_rng(62)
    frame = CMTM(SE3.set_mat(SE3.exp(np.full(6, .2))),
                 rng.normal(scale=.2, size=batch + (order - 1, 6)))
    values = rng.normal(size=batch + (order, 6))
    series = CMVector(values)
    payload = dict(cmtm_to_state_list(frame, owner_type, "target"))
    payload.update(vecs_to_state_dict(values, owner_type, "target", family, order))
    wrench = CMTM.change_elemclass(frame, SE3wrench)
    transformed = (wrench.mat_adj() @ series.cm_vec()[..., None])[..., 0]
    expected = transformed[..., -6:] * math.factorial(derivative)
    key = family if derivative == 0 else f"{family}_diff{derivative}"
    extractor = extract_dict_link_info if owner_type == "link" else extract_dict_joint_info
    monkeypatch.setattr(CMVector, "vecs", forbidden)
    monkeypatch.setattr(Factorial, "mat", forbidden)
    actual = extractor(payload, key, "target", frame="world")
    assert actual.shape == batch + (6,)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def selections(derivative):
    force = "force" if derivative == 0 else f"force_diff{derivative}"
    spatial = "vel" if derivative == 0 else "jerk"
    return [
        StateType("link", "a_tip", spatial, "world"),
        StateType("joint", "a_elbow", "acc", "world"),
        StateType("link", "a_tip", force, "world"),
        StateType("joint", "a_shoulder", force, "world"),
        StateType("joint", "a_shoulder", f"momentum_diff{derivative + 1}", "world"),
    ]


@pytest.mark.parametrize("derivative", [0, 2])
@pytest.mark.parametrize("batch", [(), (2, 1)])
@pytest.mark.parametrize("zero_pose", [False, True])
def test_world_jacobians_and_direct_products(derivative, batch, zero_pose, monkeypatch):
    order = derivative + 4
    kots = Kots.from_urdf_file(str(MODEL), order=order, backend="numpy")
    rng = np.random.default_rng(63)
    motion = rng.normal(scale=.2, size=batch + (kots.dof(), order))
    if zero_pose:
        motion[..., 0] = 0
    states = selections(derivative)

    def values(current):
        kots.import_motion_array(current)
        kots.dynamics(backend="numpy", gravity=(.2, -.3, -9.81))
        return kots.state_info_list(states)

    baseline = values(motion)
    max_order = StateType.max_time_order(states)
    v = rng.normal(size=batch + (kots.dof() * max_order, 2))
    # Build the state before forbidding full ordinary-series reconstruction.
    with monkeypatch.context() as patch:
        patch.setattr(CMVector, "vecs", forbidden)
        np.testing.assert_allclose(kots.state_info_list(states), baseline)
        analytic = kots.jacobian(states)
        direct = kots.jacobian_mul(states, v)
    np.testing.assert_allclose(direct, analytic @ v, rtol=1e-10, atol=1e-10)
    w = rng.normal(size=batch + (analytic.shape[-2], 2))
    reverse = kots.jacobian_transpose_mul(states, w)
    np.testing.assert_allclose(reverse, np.swapaxes(analytic, -1, -2) @ w, rtol=1e-10, atol=1e-10)

    columns = []
    for owner in range(kots.dof()):
        for k in range(max_order):
            step = np.zeros_like(motion)
            step[..., owner, k] = 1e-6
            columns.append((values(motion + step) - values(motion - step)) / 2e-6)
    numerical = np.stack(columns, axis=-1)
    np.testing.assert_allclose(analytic, numerical, rtol=3e-7, atol=3e-7)
