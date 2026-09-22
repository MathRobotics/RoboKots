"""Compatibility and validation of the Python-to-native model boundary."""
import numpy as np
import pytest


def model(kind="revolute"):
    return {
        "links": [{}, {"mass": 2.0, "inertia": {"ixx": 1.0, "iyy": 2.0, "izz": 3.0}}],
        "joints": [{"parent_link_id": 0, "child_link_id": 1, "type": kind}],
    }


def compiled(data, allow_prismatic=False):
    rust = pytest.importorskip("robokots._rust_core")
    return rust.RustCompiledRobot.from_model_data(data, allow_prismatic)


@pytest.mark.parametrize("kind,expected", [("revolute", 2.1), ("prismatic", 1.4)])
def test_native_model_preserves_defaults_and_dynamics(kind, expected):
    robot = compiled(model(kind), True)
    assert (robot.link_num, robot.joint_num, robot.dof) == (2, 1, 1)
    np.testing.assert_allclose(
        robot.rnea(np.array([.3]), np.zeros(1), np.array([.7])), [expected], atol=1e-12)
    for obj, name in [
        (robot, "RustCompiledRobot"),
        (robot.create_fast_data(), "RustFastData"),
        (robot.create_aba_data(), "RustAbaData"),
        (robot.create_outward_data(3), "RustOutwardData"),
        (robot.create_batch_outward_data(3, 2), "RustBatchOutwardData"),
        (robot.create_selected_workspace(3), "RustSelectedWorkspace"),
    ]:
        assert type(obj).__name__ == name


@pytest.mark.parametrize("problem", ["empty", "index", "cycle", "disconnected", "reverse_order"])
def test_invalid_native_topology_raises_value_error(problem):
    data = model()
    if problem == "empty":
        data = {"links": [], "joints": []}
    elif problem == "index":
        data["joints"][0]["child_link_id"] = 99
    elif problem == "cycle":
        data["joints"][0]["child_link_id"] = 0
    elif problem == "disconnected":
        data["links"].append({})
    else:
        data["links"].append({})
        data["joints"].insert(0, {"parent_link_id": 1, "child_link_id": 2, "type": "fixed"})
    with pytest.raises(ValueError):
        compiled(data)


@pytest.mark.parametrize("kind,representation,message", [
    ("prismatic", None, "fixed/revolute joints only"),
    ("spherical", None, "q_representation='rotation_vector'"),
    ("spherical", "rotation_vector", "spherical/floating joints"),
    ("floating", None, "q_representation='expmap'"),
    ("floating", "expmap", "spherical/floating joints"),
    ("unknown", None, "fixed/revolute/prismatic joints only"),
])
def test_unsupported_joint_contract(kind, representation, message):
    data = model(kind)
    if representation is not None:
        data["joints"][0]["q_representation"] = representation
    with pytest.raises(ValueError, match=message):
        compiled(data)
