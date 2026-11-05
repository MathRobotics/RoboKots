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
    objects = [
        (robot, "RustCompiledRobot"),
        (robot.create_fast_data(), "RustFastData"),
        (robot.create_aba_data(), "RustAbaData"),
    ]
    if kind == "revolute":
        objects.extend([
            (robot.create_outward_data(3), "RustOutwardData"),
            (robot.create_batch_outward_data(3, 2), "RustBatchOutwardData"),
            (robot.create_selected_workspace(3), "RustSelectedWorkspace"),
        ])
    else:
        for factory, args in [(robot.create_outward_data, (3,)),
                              (robot.create_batch_outward_data, (3, 2)),
                              (robot.create_selected_workspace, (3,))]:
            with pytest.raises(ValueError, match="CMTM supports fixed/revolute"):
                factory(*args)
    for obj, name in objects:
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


@pytest.mark.parametrize("batch_shape", [(), (2, 1)])
def test_rust_state_and_selection_use_metadata_without_robotstruct(batch_shape):
    from pathlib import Path
    from dataclasses import FrozenInstanceError
    from robokots.kots import Kots, StateType
    from robokots.core.robot import RobotModelInfo

    path = Path(__file__).resolve().parents[1] / "test_model/branched_fixed.urdf"
    k = Kots.from_urdf_file(str(path), order=4)
    rng = np.random.default_rng(453)
    motion = rng.normal(scale=.2, size=batch_shape + (k.dof()*4,))
    k.import_motions(motion)
    k.dynamics(backend="rust", gravity=[.2, -.3, -9.81])
    info = k._rust_model_info()
    assert isinstance(info, RobotModelInfo)
    assert info.link_names == tuple(k.robot_.link_names)
    assert info.joint_names == tuple(k.robot_.joint_names)
    assert info.motion_owners() == k.robot_.motion_owners()
    for joint, expected in zip(info.joints, k.robot_.joints):
        assert (joint.id, joint.dof, joint.dof_index, joint.parent_link_id, joint.child_link_id) == (
            expected.id, expected.dof, expected.dof_index, expected.parent_link_id, expected.child_link_id)
    assert k.outward_state_.robot is info
    with pytest.raises(FrozenInstanceError):
        info.dof = 99

    states = [
        StateType("joint", "a_elbow", "acc", "world"),
        StateType("link", "a_tip", "force_diff1", "world"),
        StateType("link", "b_payload", "pos"),
    ]
    expected_values = k.state_info_list(states)
    expected_export = k.to_state_dict()
    expected_total = k.state_info(StateType("total_joint", "total_joint", "torque"))
    expected_coords = k.state_info_list([StateType("joint", "a_elbow", "coord")])
    jac = k.jacobian(states)
    direction = rng.normal(size=batch_shape + (motion.shape[-1],))
    weight = rng.normal(size=batch_shape + (jac.shape[-2],))

    class ForbiddenRobot:
        def __getattribute__(self, name):
            raise AssertionError(f"Rust state/selection accessed RobotStruct.{name}")

    k.robot_ = ForbiddenRobot()
    np.testing.assert_array_equal(k.state_info_list(states), expected_values)
    np.testing.assert_array_equal(k.state_info(StateType("total_joint", "total_joint", "torque")), expected_total)
    np.testing.assert_array_equal(k.state_info_list([StateType("joint", "a_elbow", "coord")]), expected_coords)
    actual_export = k.to_state_dict()
    assert actual_export.keys() == expected_export.keys()
    for key in actual_export:
        np.testing.assert_array_equal(actual_export[key], expected_export[key])
    jvp = k._rust_selected_dynamics_apply(states, 4, direction, batch_shape, rhs_is_matrix=False)
    vjp = k._rust_selected_dynamics_apply(states, 4, weight, batch_shape, rhs_is_matrix=False, transpose=True)
    np.testing.assert_allclose(jvp, (jac @ direction[..., None])[..., 0], atol=1e-11)
    np.testing.assert_allclose(vjp, (np.swapaxes(jac, -1, -2) @ weight[..., None])[..., 0], atol=1e-11)
    # Rebuild the outward container using cached native model metadata.
    k._rust_outward_data_cache_.clear()
    k.update_rust_data(order=4, is_dynamics=True, gravity=[.2, -.3, -9.81])
    np.testing.assert_allclose(k.state_info_list(states), expected_values, atol=1e-12)


def test_metadata_cache_follows_compiled_model_identity():
    from pathlib import Path
    from robokots.kots import Kots
    path = Path(__file__).resolve().parents[1] / "test_model/branched_fixed.urdf"
    k = Kots.from_urdf_file(str(path), order=3)
    original = k._rust_model_info()
    assert k._rust_model_info() is original
    k._rust_compiled_robot_ = compiled(model())
    replacement = k._rust_model_info()
    assert replacement is not original
    assert replacement.link_names == ("link_0", "link_1")
    assert replacement.joint_names == ("joint_0",)
    assert replacement.dof == 1


def test_rust_state_constructs_from_raw_metadata_without_model_compile(monkeypatch):
    from robokots.core.robot import RobotModelInfo
    from robokots.outward.rust.data import RustOutwardState
    robot = compiled(model())
    raw = robot.create_outward_data(3)

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected model recompilation")

    monkeypatch.setattr("robokots.outward.rust.data._rust_compiled_robot", forbidden)
    state = RustOutwardState(None, raw, 3)
    assert isinstance(state.robot, RobotModelInfo)
    state.compute_kinematics(np.zeros(3))
    np.testing.assert_array_equal(state.link_mat("link_1"), np.eye(4))


def test_native_owner_names_are_validated():
    data = model()
    data["links"][0]["name"] = data["links"][1]["name"] = "duplicate"
    with pytest.raises(ValueError, match="names must be nonempty and unique"):
        compiled(data)
