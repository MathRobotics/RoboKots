"""NumPy mode must not invoke native kernels, including on-demand quantities."""
import builtins
from pathlib import Path

import numpy as np
import pytest

from robokots.kots import Kots, StateType
from robokots.urdf_io import load_urdf_file

MODEL = Path(__file__).parent / "test_model/branched_fixed.urdf"


def forbid_rust(monkeypatch):
    original = builtins.__import__
    def checked(name, globals=None, locals=None, fromlist=(), level=0):
        if "rust" in name:
            raise AssertionError(f"unexpected native import: {name}")
        return original(name, globals, locals, fromlist, level)
    def forbidden(*a, **kw):
        raise AssertionError("unexpected native computation")
    monkeypatch.setattr(builtins, "__import__", checked)
    for method in ("_rust_compiled_robot", "_rust_inverse_dynamics_robot", "_rust_model_info", "_cached_rust_data"):
        monkeypatch.setattr(Kots, method, forbidden)


@pytest.mark.parametrize("shape", [(), (2, 1)])
def test_numpy_mode_state_energy_and_direct_products(monkeypatch, shape):
    data = load_urdf_file(str(MODEL))
    rng = np.random.default_rng(14)
    rust = Kots.from_json_data(data, order=4, backend="rust")
    motion = rng.normal(scale=.2, size=shape + (rust.dof()*4,))
    rust.import_motions(motion)
    rust.dynamics(gravity=[.2, -.3, -9.81])
    selections = [StateType("link", "a_tip", "vel", "world"),
                  StateType("joint", "b_payload_fixed", "force", "world"),
                  StateType("link", "a_tip", "pos")]
    expected = rust.state_info_list(selections)
    jac = rust.jacobian(selections)
    energy = rust.kinetic_energy_state()
    gradient = rust.kinetic_energy_jacobian_transpose_mul(np.ones(1))
    forbid_rust(monkeypatch)
    k = Kots.from_json_data(data, order=4, backend="numpy")
    k.import_motions(motion)
    k.dynamics(gravity=[.2, -.3, -9.81])
    np.testing.assert_allclose(k.state_info_list(selections), expected, atol=1e-11)
    np.testing.assert_allclose(k.jacobian(selections), jac, atol=1e-10)
    direction = rng.normal(size=shape + (jac.shape[-1], 2))
    cotangent = rng.normal(size=shape + (jac.shape[-2], 2))
    np.testing.assert_allclose(k.jacobian_mul(selections, direction), jac @ direction, atol=1e-10)
    np.testing.assert_allclose(k.jacobian_transpose_mul(selections, cotangent), jac.swapaxes(-1, -2) @ cotangent, atol=1e-10)
    np.testing.assert_allclose(k.kinetic_energy_state(), energy, atol=1e-12)
    np.testing.assert_allclose(k.kinetic_energy_jacobian_transpose_mul(np.ones(1)), gradient, atol=1e-11)
    from robokots import outward as outward_api
    from robokots.outward.diff import outward_total_gradient
    def dense_forbidden(*args, **kwargs):
        raise AssertionError("energy products must not construct a dense Jacobian")
    monkeypatch.setattr(outward_api, "outward_jacobian", dense_forbidden)
    monkeypatch.setattr(outward_total_gradient, "outward_jacobian", dense_forbidden)
    energy_direction = rng.normal(size=shape + (k.dof()*2, 2))
    np.testing.assert_allclose(k.kinetic_energy_jacobian_mul(energy_direction), gradient[..., None, :] @ energy_direction, atol=1e-11)
    weights = rng.normal(size=shape + (1, 2))
    np.testing.assert_allclose(k.kinetic_energy_jacobian_transpose_mul(weights), gradient[..., :, None] * weights, atol=1e-11)
    energy_state = StateType("total_body", "total_body", "kinetic_energy")
    np.testing.assert_allclose(k.jacobian(energy_state), gradient[..., None, :], atol=1e-11)
    k.import_motions(motion*.7)
    k.update_state(is_dynamics=True)
    k.kinetic_energy_state()


@pytest.mark.parametrize("prismatic", [False, True])
@pytest.mark.parametrize("batch_size", [None, 2, 6, 12])
def test_numpy_inverse_forward_and_cache(monkeypatch, prismatic, batch_size):
    data = load_urdf_file(str(MODEL))
    if prismatic:
        next(j for j in data["joints"] if j["type"] == "revolute")["type"] = "prismatic"
    rust = Kots.from_json_data(data, backend="rust")
    rng = np.random.default_rng(41)
    shape = (rust.dof(),) if batch_size is None else (batch_size, rust.dof())
    q, v, a = (rng.normal(scale=.2, size=shape) for _ in range(3))
    gravity = [.3, -.1, -9.81]
    torque = rust.inverse_dynamics(q, v, a, gravity=gravity)
    forbid_rust(monkeypatch)
    k = Kots.from_json_data(data, backend="numpy")
    np.testing.assert_allclose(k.inverse_dynamics(q, v, a, gravity=gravity), torque, atol=1e-11)
    np.testing.assert_allclose(k.forward_dynamics(q, v, torque, gravity=gravity), a, atol=1e-10)
    np.testing.assert_allclose(k.forward_dynamics(q, v, torque, gravity=gravity, backend="reference"), a, atol=1e-10)
    if batch_size is None:
        cache = k.create_inward_cache().prepare(q, v, gravity)
        np.testing.assert_allclose(cache.forward_dynamics(torque), a, atol=1e-10)
        np.testing.assert_allclose(cache.forward_dynamics_many(np.stack([torque, torque])), np.stack([a, a]), atol=1e-10)
        cache.invalidate()
        assert not cache.is_prepared
    assert k.outward_state_ is None


def test_numpy_prismatic_energy_matches_closed_form_and_finite_difference(monkeypatch):
    forbid_rust(monkeypatch)
    model = {"schema_version": "0.0.2", "links": [
        {"id": 0, "name": "world"}, {"id": 1, "name": "body", "mass": 2.}],
        "joints": [{"id": 0, "name": "slide", "type": "prismatic", "axis": [0., 0., 1.],
                    "parent_link_id": 0, "child_link_id": 1}]}
    k = Kots.from_json_data(model, backend="numpy")
    motion = np.array([.2, .5, 0.])
    k.import_motions(motion)
    assert k.kinetic_energy_state() == pytest.approx(.25)
    gradient = k.kinetic_energy_jacobian_transpose_mul(np.ones(1))
    np.testing.assert_allclose(gradient, [0., 1.], atol=1e-12)
    finite_difference = []
    for i in range(2):
        step = np.eye(3)[i]*1e-6
        k.import_motions(motion + step)
        plus = k.kinetic_energy_state()
        k.import_motions(motion - step)
        minus = k.kinetic_energy_state()
        finite_difference.append((plus-minus)/2e-6)
    np.testing.assert_allclose(gradient, finite_difference, atol=1e-10)


def test_explicit_numpy_state_switch_controls_on_demand_calculations(monkeypatch):
    k = Kots.from_urdf_file(str(MODEL), backend="rust")
    k.kinematics()
    k.dynamics(backend="numpy")
    forbid_rust(monkeypatch)
    k.kinetic_energy_state()
    q = np.zeros(k.dof())
    torque = k.inverse_dynamics(q, q, q)
    np.testing.assert_allclose(k.forward_dynamics(q, q, torque), q, atol=1e-10)
    k.create_inward_cache().prepare(q, q)


@pytest.mark.parametrize("shape", [(0,), (2, 0)])
def test_numpy_motion_preserves_empty_batch_rejection(monkeypatch, shape):
    forbid_rust(monkeypatch)
    k = Kots.from_urdf_file(str(MODEL), backend="numpy")
    with pytest.raises(ValueError, match="empty batches are unsupported"):
        k.import_motions(np.empty(shape + (k.dof()*3,)))


def test_numpy_empty_dynamics_and_cache_reuse(monkeypatch):
    forbid_rust(monkeypatch)
    k = Kots.from_urdf_file(str(MODEL), backend="numpy")
    empty = np.empty((0, k.dof()))
    assert k.inverse_dynamics(empty, empty, empty).shape == empty.shape
    assert k.forward_dynamics(empty, empty, empty).shape == empty.shape
    q = np.zeros(k.dof())
    cache = k.create_inward_cache().prepare(q, q)
    torque = k.inverse_dynamics(q, q, q)
    def forbidden(*args, **kwargs):
        raise AssertionError("prepared cache must reuse mass and bias")
    monkeypatch.setattr(k, "inverse_dynamics", forbidden)
    cache.prepare(q, q)
    np.testing.assert_allclose(cache.forward_dynamics(torque), q, atol=1e-12)
    np.testing.assert_allclose(cache.forward_dynamics_many(np.stack([torque, torque])), np.zeros((2, k.dof())), atol=1e-12)
    assert cache.forward_dynamics_many(empty).shape == empty.shape
