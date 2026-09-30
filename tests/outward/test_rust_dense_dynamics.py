"""Dedicated dense dynamics assembly and transitions to direct products."""
from pathlib import Path

import numpy as np
import pytest

from robokots import outward
from robokots.kots import Kots, StateType

MODEL = Path(__file__).resolve().parents[1] / "test_model/branched_fixed.urdf"


@pytest.mark.parametrize("derivative", range(5))
@pytest.mark.parametrize("shape", [(), (2, 1)])
@pytest.mark.parametrize("coordinates", [False, True])
def test_dense_torque_without_identity_or_python_fallback(monkeypatch, derivative, shape, coordinates):
    order = derivative + 3
    reference = Kots.from_urdf_file(str(MODEL), order=order, backend="numpy")
    rust = Kots.from_urdf_file(str(MODEL), order=order, backend="rust")
    x = np.random.default_rng(632).normal(scale=.2, size=shape + (rust.dof() * order,))
    key = "torque" if derivative == 0 else f"torque_diff{derivative}"
    states = [StateType("total_joint", "total_joint", key)]
    if coordinates:
        states = [StateType("joint", "a_elbow", "coord"), *states,
                  StateType("joint", "b_shoulder", "accel")]
    for k in (reference, rust):
        k.import_motions(x)
        k.dynamics(gravity=[.2, -.3, -9.81])
    expected = reference.jacobian(states)

    def forbidden(*args, **kwargs):
        raise AssertionError("dense Rust dynamics must not use Python assembly or an identity RHS")

    monkeypatch.setattr(outward, "outward_jacobian", forbidden)
    monkeypatch.setattr(rust, "_rust_selected_dynamics_apply", forbidden)
    monkeypatch.setattr(rust, "_rust_cmtm_torque_jacobian_apply", forbidden)
    monkeypatch.setattr(np, "eye", forbidden)
    np.testing.assert_allclose(rust.jacobian(states), expected, atol=2e-8, rtol=2e-9)
    parts = rust.jacobian(states, list_output=True)
    np.testing.assert_allclose(np.concatenate(parts, axis=-2), expected, atol=2e-8, rtol=2e-9)


def test_dense_workspace_interleaves_products_and_invalidates_state():
    order = 5
    rust = Kots.from_urdf_file(str(MODEL), order=order, backend="rust")
    reference = Kots.from_urdf_file(str(MODEL), order=order, backend="numpy")
    rng = np.random.default_rng(517)
    x = rng.normal(scale=.2, size=(2, rust.dof() * order))
    gravity = np.array([.2, -.3, -9.81])
    robot = rust._rust_compiled_robot()
    ws = robot.create_selected_workspace(order)
    force = StateType("joint", "a_shoulder", "force_diff2", "world")
    states = [force, StateType("link", "b_payload", "momentum_diff2", "local"),
              StateType("joint", "a_elbow", "vel", "world"), force]
    for step in range(5):
        if step == 1:
            x[1, 0] += .17
        elif step == 2:
            gravity = np.zeros(3)
        elif step == 3:
            x = x[:1].copy()
        elif step == 4:
            x = np.zeros((2, rust.dof() * order))
            gravity = np.array([.1, -.5, -9.81])
        for k in (reference, rust):
            k.import_motions(x)
            k.dynamics(gravity=gravity)
        for selected in (states, states[::-1], [StateType("link", "a_tip", "snap", "world")], states):
            expected = reference.jacobian(selected)
            expanded = rust._state_type_list(selected)
            specs, _ = rust._rust_selected_dynamics_specs(expanded, order, dense=True)
            actual = np.asarray(ws.jacobian(x, specs, gravity))
            np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-9)
            saved = actual.copy()
            counts = ws.cache_info()
            for width in (1, 3):
                v = rng.normal(size=(len(x), x.shape[-1], width))
                w = rng.normal(size=(len(x), actual.shape[-2], width))
                np.testing.assert_allclose(ws.apply(x, v, specs, gravity), expected @ v, atol=2e-8, rtol=2e-9)
                np.testing.assert_allclose(ws.apply(x, w, specs, gravity, transpose=True),
                                           expected.swapaxes(-1, -2) @ w, atol=2e-8, rtol=2e-9)
                np.testing.assert_allclose(ws.jacobian(x, specs, gravity), expected, atol=2e-9, rtol=2e-9)
                assert ws.cache_info() == counts
            np.testing.assert_array_equal(actual, saved)
