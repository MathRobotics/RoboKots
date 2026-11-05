"""Native-first input, lazy Python models, and canonical input validation."""
import json
from pathlib import Path

import numpy as np
import pytest

from robokots.kots import Kots, StateType
from robokots.core.robot import RobotStruct
from robokots.urdf_io import load_urdf_file

MODEL = Path(__file__).parent / "test_model/branched_fixed.urdf"


def states():
    return [StateType("joint", "a_elbow", "acc", "world"),
            StateType("link", "a_tip", "force_diff1", "world"),
            StateType("link", "b_payload", "pos")]


@pytest.mark.parametrize("source", ["dict", "json", "urdf"])
@pytest.mark.parametrize("batch_shape", [(), (2, 1)])
def test_direct_input_avoids_python_model_and_matches_existing_path(source, batch_shape, tmp_path, monkeypatch):
    pytest.importorskip("robokots._rust_core")
    baseline = Kots.from_urdf_file(str(MODEL), order=4)
    data = load_urdf_file(str(MODEL))
    # Array order is unrelated to IDs and must not alter motion order.
    data["links"].reverse()
    data["joints"].reverse()
    path = tmp_path / "model.json"
    path.write_text(json.dumps(data))
    rng = np.random.default_rng(346)
    motion = rng.normal(scale=.2, size=batch_shape + (baseline.dof()*4,))
    gravity = [.2, -.3, -9.81]
    baseline.import_motions(motion)
    baseline.dynamics(backend="rust", gravity=gravity)
    expected = baseline.state_info_list(states())
    jac = baseline.jacobian(states())
    direction = rng.normal(size=batch_shape + (jac.shape[-1],))
    weight = rng.normal(size=batch_shape + (jac.shape[-2],))
    q, v, a = (np.ascontiguousarray(motion[..., i::4]) for i in range(3))
    if batch_shape:
        q, v, a = (x.reshape(-1, baseline.dof()) for x in (q, v, a))
    torque = baseline.inverse_dynamics(q, v, a, gravity=gravity)

    def forbidden(*args, **kwargs):
        raise AssertionError("Python model expansion or reserialization")

    monkeypatch.setattr(RobotStruct, "from_dict", forbidden)
    monkeypatch.setattr(RobotStruct, "to_dict", forbidden)
    if source == "dict":
        k = Kots.from_json_data(data, order=4, backend="rust")
        data["links"].clear()  # Owned input snapshot must survive caller mutation.
    elif source == "json":
        k = Kots.from_json_file(path, order=4, backend="rust")
        path.unlink()  # No later file reads.
    else:
        k = Kots.from_urdf_file(str(MODEL), order=4, backend="rust")
    assert k._python_robot_ is None
    assert k.link_name_list() == baseline.link_name_list()
    assert k.joint_name_list() == baseline.joint_name_list()
    k.import_motions(motion)
    k.dynamics(gravity=gravity)
    np.testing.assert_allclose(k.state_info_list(states()), expected, atol=1e-11)
    np.testing.assert_allclose(k.jacobian(states()), jac, atol=1e-11)
    np.testing.assert_allclose(k.jacobian_mul(states(), direction), (jac @ direction[..., None])[..., 0], atol=1e-10)
    np.testing.assert_allclose(k.jacobian_transpose_mul(states(), weight), (jac.swapaxes(-1, -2) @ weight[..., None])[..., 0], atol=1e-10)
    assert k.to_state_dict().keys() == baseline.to_state_dict().keys()
    np.testing.assert_allclose(k.inverse_dynamics(q, v, a, gravity=gravity), torque, atol=1e-11)
    np.testing.assert_allclose(k.forward_dynamics(q, v, torque, gravity=gravity), a, atol=1e-10)
    k.set_order(3)
    k.import_motions(motion.reshape(batch_shape + (k.dof(), 4))[..., :3].reshape(batch_shape + (-1,)))
    k.kinematics(materialize_dict=True)
    k.update_state(is_dynamics=True)
    assert k._python_robot_ is None


def test_python_model_is_materialized_once_only_when_needed(monkeypatch):
    data = load_urdf_file(str(MODEL))
    k = Kots.from_json_data(data, backend="rust")
    data["links"].clear()
    original = RobotStruct.from_dict
    calls = []

    def counted(data, *args, **kwargs):
        calls.append(1)
        return original(data, *args, **kwargs)

    monkeypatch.setattr(RobotStruct, "from_dict", counted)
    k.kinematics()
    assert calls == []
    k.kinematics(backend="numpy")
    assert len(calls) == 1
    assert isinstance(k.robot_, RobotStruct)
    k.kinematics(backend="numpy")
    assert len(calls) == 1
    k.kinematics()
    assert hasattr(k.outward_state_, "raw_data")


@pytest.mark.parametrize("change", [
    lambda d: d.pop("schema_version"),
    lambda d: d.update(schema_version="0.0.1"),
    lambda d: d["links"][0].update(id=1),
    lambda d: d["links"][0].update(id=True),
    lambda d: d["links"][0].update(name=d["links"][1]["name"]),
    lambda d: d["links"][0].update(mass=float("nan")),
    lambda d: d["links"][0].update(mass=-1),
    lambda d: d["links"][1].update(cog=[0, 1]),
    lambda d: d["links"][1].update(type="soft"),
    lambda d: d["links"][1].update(inertia={"ixx": 1}),
    lambda d: d["joints"][1].update(axis=[0, 0, 0]),
    lambda d: d["joints"][1].pop("axis"),
    lambda d: d["joints"][1].update(dof=3),
    lambda d: d["joints"][1].update(parent_link_id=999),
    lambda d: d["joints"][1].update(origin={"orientation": [0, 0, 0, 0]}),
    lambda d: d["joints"][1].update(origin={"position": [0, float("inf"), 0]}),
])
def test_rust_input_validates_without_python_model(change, monkeypatch):
    data = load_urdf_file(str(MODEL))
    change(data)
    monkeypatch.setattr(RobotStruct, "from_dict", lambda *a, **k: pytest.fail("Python validation called"))
    with pytest.raises(ValueError):
        Kots.from_json_data(data, backend="rust")


def test_prismatic_direct_input_keeps_rnea_aba_support():
    data = load_urdf_file(str(MODEL))
    data["joints"][1]["type"] = "prismatic"
    k = Kots.from_json_data(data, backend="rust")
    q = np.full(k.dof(), .1)
    v = np.full(k.dof(), .2)
    a = np.full(k.dof(), -.3)
    torque = k.inverse_dynamics(q, v, a)
    np.testing.assert_allclose(k.forward_dynamics(q, v, torque), a, atol=1e-10)
    with pytest.raises(ValueError, match="CMTM supports"):
        k.dynamics()
    assert k._python_robot_ is None


def test_invalid_input_mode():
    with pytest.raises(ValueError, match="backend"):
        Kots.from_json_data({}, backend="invalid")
    with pytest.raises(ValueError, match="dim=3"):
        Kots.from_json_data({}, backend="rust", dim=2)


def test_cyclic_decoded_input_is_rejected_at_binding_boundary():
    rust = pytest.importorskip("robokots._rust_core")
    data = {}
    data["cycle"] = data
    with pytest.raises(ValueError, match="cyclic"):
        rust.RustCompiledRobot.from_input_data(data)


def single_joint_model(joint_type="revolute", quaternion=None):
    return {
        "schema_version": "0.0.2",
        "links": [{"id": 0, "name": "world"},
                  {"id": 1, "name": "arm", "mass": 2.}],
        "joints": [{"id": 0, "name": "j", "type": joint_type,
                    "axis": [0., 0., 1.], "parent_link_id": 0, "child_link_id": 1,
                    "origin": {"orientation": quaternion or [1., 0., 0., 0.]}}],
    }


@pytest.mark.parametrize("batch_size", [None, 0, 2])
@pytest.mark.parametrize("operation", ["value", "jvp", "vjp"])
def test_native_prismatic_energy_is_explicitly_unsupported(batch_size, operation):
    rust = pytest.importorskip("robokots._rust_core")
    model = rust.RustCompiledRobot.from_input_data(single_joint_model("prismatic"))
    motion = np.array([.2, .5])
    suffix = "" if batch_size is None else "_batch"
    if batch_size is not None:
        motion = np.broadcast_to(motion, (batch_size, 2)).copy()
    prefix = () if batch_size is None else (batch_size,)
    args = {
        "value": ("kinetic_energy", (motion,)),
        "jvp": ("kinetic_energy_jacobian_mul_rhs", (motion, np.ones(prefix + (2, 1)))),
        "vjp": ("kinetic_energy_jacobian_transpose_mul_rhs", (motion, np.ones(prefix + (1, 1)))),
    }
    method, inputs = args[operation]
    with pytest.raises(ValueError, match="CMTM supports fixed/revolute"):
        getattr(model, method + suffix)(*inputs)


@pytest.mark.parametrize("batched", [False, True])
def test_public_native_prismatic_energy_rejects_unsupported_model(batched):
    k = Kots.from_json_data(single_joint_model("prismatic"), backend="rust")
    motion = np.array([.2, .5, 0.])
    k.import_motions(np.tile(motion, (2, 1)) if batched else motion)
    for compute in (lambda: k.kinetic_energy_state(),
                    lambda: k.kinetic_energy_jacobian_mul(np.ones(2)),
                    lambda: k.kinetic_energy_jacobian_transpose_mul(np.ones(1))):
        with pytest.raises(ValueError, match="CMTM supports fixed/revolute"):
            compute()
    assert k._python_robot_ is None


@pytest.mark.parametrize("scale", [2., -3., .1])
@pytest.mark.parametrize("source", ["dict", "json"])
def test_nonunit_quaternion_matches_on_lazy_backend_switch(scale, source, tmp_path):
    data = single_joint_model(quaternion=[scale, 0., 0., scale])
    original = json.dumps(data)
    if source == "json":
        path = tmp_path / "nonunit.json"
        path.write_text(original)
        k = Kots.from_json_file(path, backend="rust")
    else:
        k = Kots.from_json_data(data, backend="rust")
    rotation = StateType("link", "arm", "rot")
    expected = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    k.kinematics()
    native = k.state_info(rotation)
    assert k._python_robot_ is None
    np.testing.assert_allclose(np.asarray(native).reshape(3, 3), expected, atol=1e-14)
    k.kinematics(backend="numpy")
    np.testing.assert_allclose(k.state_info(rotation), native, atol=1e-14)
    # The regular Python constructor must use the same convention, without mutating input.
    python = Kots.from_json_data(data)
    python.kinematics(backend="numpy")
    np.testing.assert_allclose(python.state_info(rotation), native, atol=1e-14)
    assert json.dumps(data) == original
