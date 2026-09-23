"""World spatial values share the native convention with selected derivatives."""
from math import factorial
from pathlib import Path

import numpy as np
import pytest

from robokots.kots import Kots, StateType

MODEL = Path(__file__).resolve().parents[1] / "test_model/branched_fixed.urdf"
KEYS = ("vel", "acc", "jerk", "snap", "crackle", "pop")
OWNERS = (("link", "a_tip"), ("joint", "a_elbow"), ("joint", "b_payload_fixed"))


def selections(order):
    return [StateType(owner, name, key, "world") for owner, name in OWNERS for key in KEYS[:order-1]]


@pytest.mark.parametrize("order,mode", [(2, "kinematics")] + [(o, m) for o in (3, 6) for m in ("kinematics", "dynamics", "minimal")])
@pytest.mark.parametrize("shape", [(), (2, 1)])
def test_world_values_match_numpy_without_python_conversion(order, shape, mode, monkeypatch):
    reference = Kots.from_urdf_file(str(MODEL), order=order)
    k = Kots.from_urdf_file(str(MODEL), order=order, backend="rust")
    rng = np.random.default_rng(102)
    x = rng.normal(scale=.25, size=shape + (k.dof()*order,))
    x[..., ::order] = 0  # Include the zero-pose case with nonzero motion.
    reference.import_motions(x)
    reference.kinematics()
    expected = reference.state_info_list(selections(order))
    k.import_motions(x)
    if mode == "kinematics":
        k.kinematics()
    elif mode == "dynamics":
        k.dynamics(gravity=[.2, -.3, -9.81])
    else:
        k.kinematics()
        k.outward_state_.compute_dynamics_minimal(x, gravity=[.2, -.3, -9.81])
    from robokots.core.kernels import cmtm_apply
    def forbidden(*a, **kw):
        raise AssertionError("Python world conversion or model expansion")
    monkeypatch.setattr(cmtm_apply, "world_spatial_value", forbidden)
    monkeypatch.setattr(k.outward_state_, "cmtm", forbidden)
    actual = k.state_info_list(selections(order))
    assert actual.shape == shape + (len(OWNERS)*(order-1)*6,)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
    assert k._python_robot_ is None
    # Detached outputs and cache invalidation after changing the motion.
    actual[...] = 99
    np.testing.assert_allclose(k.state_info_list(selections(order)), expected, atol=2e-12)
    k.import_motions(x*.4)
    k.kinematics()
    assert not np.allclose(k.state_info_list(selections(order)), expected)


def test_world_series_are_time_derivatives_and_selected_jacobians(monkeypatch):
    order = 6
    k = Kots.from_urdf_file(str(MODEL), order=order, backend="rust")
    rng = np.random.default_rng(442)
    x = rng.normal(scale=.2, size=(k.dof(), order))
    def values(y, states):
        k.import_motion_array(y)
        k.kinematics()
        return k.state_info_list(states)
    h = 1e-6
    def advance(dt):
        return np.stack([sum(x[:, j+m]*dt**m/factorial(m) for m in range(order-j)) for j in range(order)], axis=-1)
    for owner, name in OWNERS:
        lower = [StateType(owner, name, key, "world") for key in KEYS[:order-2]]
        higher = [StateType(owner, name, key, "world") for key in KEYS[1:order-1]]
        numerical = (values(advance(h), lower) - values(advance(-h), lower))/(2*h)
        np.testing.assert_allclose(values(x, higher), numerical, atol=2e-8, rtol=2e-7)
    states = selections(order)
    base = values(x, states)
    columns = []
    for e in np.eye(x.size).reshape(-1, *x.shape):
        columns.append((values(x+h*e, states)-values(x-h*e, states))/(2*h))
    expected = np.stack(columns, axis=-1)
    values(x, states)
    jac = k.jacobian(states)
    np.testing.assert_allclose(jac, expected, atol=2e-8, rtol=2e-7)
    def forbidden(*a, **kw):
        raise AssertionError("dense Jacobian materialized for product")
    monkeypatch.setattr(k, "_jacobian_from_state", forbidden)
    v, w = rng.normal(size=x.size), rng.normal(size=base.size)
    np.testing.assert_allclose(k.jacobian_mul(states, v), jac@v, atol=2e-11)
    np.testing.assert_allclose(k.jacobian_transpose_mul(states, w), jac.T@w, atol=2e-11)


def test_world_native_getter_validation():
    k = Kots.from_urdf_file(str(MODEL), order=3, backend="rust")
    robot = k._rust_compiled_robot()
    for state in [robot.create_outward_data(3), robot.create_batch_outward_data(3, 2)]:
        for getter in [state.world_link_vec, state.world_joint_vec]:
            with pytest.raises(ValueError, match="compute_"):
                getter(0, 2)
            with pytest.raises(ValueError):
                getter(999, 2)
            with pytest.raises(ValueError):
                getter(0, 1)
            with pytest.raises(ValueError):
                getter(0, 4)


@pytest.mark.parametrize("backend", ["numpy", "rust"])
def test_joint_jerk_frame_disambiguates_coordinate_and_spatial(backend):
    k = Kots.from_urdf_file(str(MODEL), order=4, backend=backend)
    x = np.arange(k.dof()*4, dtype=float)*.03
    k.import_motions(x)
    k.kinematics(backend=backend)
    coordinate = StateType("joint", "a_elbow", "jerk")
    local = StateType("joint", "a_elbow", "jerk", "local")
    world = StateType("joint", "a_elbow", "jerk", "world")
    assert k.state_info_list([coordinate]).shape == (1,)
    assert k.state_info_list([local, world]).shape == (12,)
    assert k.jacobian([coordinate]).shape == (1, k.dof()*4)
    jac = k.jacobian([local, world])
    assert jac.shape == (12, k.dof()*4)
    direction = np.arange(k.dof()*4)*.02
    np.testing.assert_allclose(k.jacobian_mul([local, world], direction), jac@direction, atol=1e-12)


def test_empty_batch_world_getters_require_computation():
    k = Kots.from_urdf_file(str(MODEL), order=3, backend="rust")
    raw = k._rust_compiled_robot().create_batch_outward_data(3, 0)
    for get in [raw.world_link_vec, raw.world_joint_vec]:
        with pytest.raises(ValueError, match="compute_"):
            get(0, 2)
    raw.compute_kinematics(np.empty((0, k.dof()*3)))
    assert raw.world_link_vec(0, 2).shape == (0, 6)
    assert raw.world_joint_vec(0, 2).shape == (0, 6)
