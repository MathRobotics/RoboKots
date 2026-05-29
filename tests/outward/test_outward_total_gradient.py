from collections import Counter
from pathlib import Path

import numpy as np

from robokots.kots import Kots, StateType
import robokots.outward.diff.outward_total_gradient as outward_total_gradient


TEST_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = TEST_DIR / "test_model" / "sample_robot.json"


def _make_counter_wrapper(counts: Counter, key: str, factory):
    def wrapped(*args, **kwargs):
        counts[key] += 1
        return factory(*args, **kwargs)

    return wrapped


def _unexpected_builder(name: str):
    def wrapped(*args, **kwargs):
        raise AssertionError(f"unexpected builder call: {name}")

    return wrapped


def test_outward_jacobian_link_momentum_uses_only_needed_builder(monkeypatch):
    kots = Kots.from_json_file(str(MODEL_PATH), order=5)
    robot = kots.robot_
    counts = Counter()

    monkeypatch.setattr(
        outward_total_gradient,
        "_selected_coord_to_link_momentum_grad_mat",
        _make_counter_wrapper(
            counts,
            "link_mom",
            lambda robot, state, links, order, dim=3: np.zeros((len(links) * 6 * (order - 1), robot.dof * order)),
        ),
    )

    for name in (
        "total_coord_to_link_vel_grad_mat",
        "_selected_coord_to_link_vel_grad_mat",
        "_selected_coord_to_link_tan_vel_grad_mat",
        "_selected_coord_to_world_link_momentum_grad_mat",
        "_selected_coord_to_link_force_grad_mat",
        "total_coord_to_link_momentum_grad_mat",
        "total_coord_to_link_tan_vel_grad_mat",
        "total_partial_link_momentum_to_world_link_momentum_grad_mat",
        "total_partial_link_tan_vel_to_world_link_momentum_grad_mat",
        "total_world_link_wrench_to_world_joint_wrench_mat",
        "total_partial_world_joint_momentum_to_joint_momentum_grad_mat",
        "total_partial_link_tan_vel_to_joint_momentum_grad_mat",
        "total_partial_momentum_to_force_grad_mat",
        "total_partial_link_sp_vel_to_link_force_grad_mat",
        "total_partial_link_sp_vel_to_joint_force_grad_mat",
        "total_joint_wrench_to_joint_torque_mat",
    ):
        monkeypatch.setattr(outward_total_gradient, name, _unexpected_builder(name))

    jacob = outward_total_gradient.outward_jacobian(
        robot,
        {},
        [StateType("link", "arm3", "momentum")],
    )

    assert jacob.shape == (6, robot.dof * 2)
    assert counts == Counter({"link_mom": 1})


def test_outward_jacobian_link_force_uses_selected_rows(monkeypatch):
    kots = Kots.from_json_file(str(MODEL_PATH), order=5)
    robot = kots.robot_
    counts = Counter()

    monkeypatch.setattr(
        outward_total_gradient,
        "_selected_coord_to_link_force_grad_mat",
        _make_counter_wrapper(
            counts,
            "link_force",
            lambda robot, state, links, force_order, dim=3: np.zeros((len(links) * 6 * force_order, robot.dof * (force_order + 2))),
        ),
    )

    for name in (
        "total_coord_to_link_force_grad_mat",
        "total_partial_momentum_to_force_grad_mat",
        "total_partial_link_sp_vel_to_link_force_grad_mat",
        "total_coord_to_link_vel_grad_mat",
        "total_coord_to_link_momentum_grad_mat",
    ):
        monkeypatch.setattr(outward_total_gradient, name, _unexpected_builder(name))

    jacob = outward_total_gradient.outward_jacobian(
        robot,
        {},
        [StateType("link", "arm3", "force")],
    )

    assert jacob.shape == (6, robot.dof * 3)
    assert counts == Counter({"link_force": 1})


def test_outward_jacobian_joint_world_momentum_uses_selected_rows(monkeypatch):
    kots = Kots.from_json_file(str(MODEL_PATH), order=5)
    robot = kots.robot_
    counts = Counter()

    monkeypatch.setattr(
        outward_total_gradient,
        "_selected_coord_to_world_joint_momentum_grad_mat",
        _make_counter_wrapper(
            counts,
            "joint_wmom",
            lambda robot, state, joints, order, dim=3: np.zeros((len(joints) * 6 * (order - 1), robot.dof * order)),
        ),
    )

    for name in (
        "total_coord_to_world_joint_momentum_grad_mat",
        "total_coord_to_joint_momentum_grad_mat",
        "total_coord_to_joint_force_grad_mat",
        "total_coord_to_joint_torque_grad_mat",
    ):
        monkeypatch.setattr(outward_total_gradient, name, _unexpected_builder(name))

    jacob = outward_total_gradient.outward_jacobian(
        robot,
        {},
        [StateType("joint", "joint3", "momentum", "world")],
    )

    assert jacob.shape == (6, robot.dof * 2)
    assert counts == Counter({"joint_wmom": 1})


def test_outward_jacobian_joint_torque_uses_selected_rows(monkeypatch):
    kots = Kots.from_json_file(str(MODEL_PATH), order=5)
    robot = kots.robot_
    counts = Counter()

    monkeypatch.setattr(
        outward_total_gradient,
        "_selected_coord_to_joint_torque_grad_mat",
        _make_counter_wrapper(
            counts,
            "joint_torque",
            lambda robot, state, joints, torque_order, dim=3: np.zeros(
                (sum(joint.dof * torque_order for joint in joints), robot.dof * (torque_order + 2))
            ),
        ),
    )

    for name in (
        "total_coord_to_joint_torque_grad_mat",
        "total_coord_to_joint_force_grad_mat",
        "total_coord_to_joint_momentum_grad_mat",
    ):
        monkeypatch.setattr(outward_total_gradient, name, _unexpected_builder(name))

    jacob = outward_total_gradient.outward_jacobian(
        robot,
        {},
        [StateType("joint", "joint3", "torque")],
    )

    assert jacob.shape == (1, robot.dof * 3)
    assert counts == Counter({"joint_torque": 1})


def test_outward_kinematics_jacobian_single_link_uses_selected_rows(monkeypatch):
    kots = Kots.from_json_file(str(MODEL_PATH), order=3)
    robot = kots.robot_
    counts = Counter()

    monkeypatch.setattr(
        outward_total_gradient,
        "total_coord_to_link_vel_grad_mat",
        _unexpected_builder("total_coord_to_link_vel_grad_mat"),
    )
    monkeypatch.setattr(
        outward_total_gradient,
        "_selected_coord_to_link_vel_grad_mat",
        _make_counter_wrapper(
            counts,
            "selected_link_vel",
            lambda robot, state, links, order, dim=3: np.zeros((len(links) * 6 * order, robot.dof * order)),
        ),
    )

    jacob = outward_total_gradient.outward_kinematics_jacobian(
        robot,
        {},
        [StateType("link", "arm3", "frame")],
    )

    assert jacob.shape == (6, robot.dof)
    assert counts == Counter({"selected_link_vel": 1})


def test_outward_jacobian_link_momentum_matches_full_builder():
    kots = Kots.from_json_file(str(MODEL_PATH), order=5)
    motion = np.random.default_rng(0).standard_normal(kots.order() * kots.dof())
    kots.import_motions(motion)
    kots.dynamics()

    state = StateType("link", "arm3", "momentum")
    actual = outward_total_gradient.outward_jacobian(kots.robot_, kots.state_dict_, [state])
    full = outward_total_gradient.total_coord_to_link_momentum_grad_mat(
        kots.robot_, kots.state_dict_, order=state.time_order
    )
    link = kots.robot_.link(state.owner_name)
    expected = full[link.id * 6 : (link.id + 1) * 6, :]

    np.testing.assert_allclose(actual, expected)


def test_outward_jacobian_world_link_momentum_matches_full_builder():
    kots = Kots.from_json_file(str(MODEL_PATH), order=5)
    motion = np.random.default_rng(1).standard_normal(kots.order() * kots.dof())
    kots.import_motions(motion)
    kots.dynamics()

    state = StateType("link", "arm3", "momentum", "world")
    actual = outward_total_gradient.outward_jacobian(kots.robot_, kots.state_dict_, [state])
    full = outward_total_gradient.total_coord_to_world_link_momentum_grad_mat(
        kots.robot_, kots.state_dict_, order=state.time_order
    )
    link = kots.robot_.link(state.owner_name)
    expected = full[link.id * 6 : (link.id + 1) * 6, :]

    np.testing.assert_allclose(actual, expected)


def test_outward_jacobian_link_force_matches_full_builder():
    kots = Kots.from_json_file(str(MODEL_PATH), order=5)
    motion = np.random.default_rng(2).standard_normal(kots.order() * kots.dof())
    kots.import_motions(motion)
    kots.dynamics()

    state = StateType("link", "arm3", "force")
    actual = outward_total_gradient.outward_jacobian(kots.robot_, kots.state_dict_, [state])
    full = outward_total_gradient.total_coord_to_link_force_grad_mat(
        kots.robot_, kots.state_dict_, force_order=state.time_order - 2
    )
    link = kots.robot_.link(state.owner_name)
    expected = full[link.id * 6 : (link.id + 1) * 6, :]

    np.testing.assert_allclose(actual, expected)


def test_outward_jacobian_joint_momentum_matches_full_builder():
    kots = Kots.from_json_file(str(MODEL_PATH), order=5)
    motion = np.random.default_rng(3).standard_normal(kots.order() * kots.dof())
    kots.import_motions(motion)
    kots.dynamics()

    state = StateType("joint", "joint3", "momentum")
    actual = outward_total_gradient.outward_jacobian(kots.robot_, kots.state_dict_, [state])
    full = outward_total_gradient.total_coord_to_joint_momentum_grad_mat(
        kots.robot_, kots.state_dict_, order=state.time_order
    )
    joint = kots.robot_.joint(state.owner_name)
    expected = full[joint.id * 6 : (joint.id + 1) * 6, :]

    np.testing.assert_allclose(actual, expected)


def test_outward_jacobian_world_joint_momentum_matches_full_builder():
    kots = Kots.from_json_file(str(MODEL_PATH), order=5)
    motion = np.random.default_rng(4).standard_normal(kots.order() * kots.dof())
    kots.import_motions(motion)
    kots.dynamics()

    state = StateType("joint", "joint3", "momentum", "world")
    actual = outward_total_gradient.outward_jacobian(kots.robot_, kots.state_dict_, [state])
    full = outward_total_gradient.total_coord_to_world_joint_momentum_grad_mat(
        kots.robot_, kots.state_dict_, order=state.time_order
    )
    joint = kots.robot_.joint(state.owner_name)
    expected = full[joint.id * 6 : (joint.id + 1) * 6, :]

    np.testing.assert_allclose(actual, expected)


def test_outward_jacobian_joint_force_matches_full_builder():
    kots = Kots.from_json_file(str(MODEL_PATH), order=5)
    motion = np.random.default_rng(5).standard_normal(kots.order() * kots.dof())
    kots.import_motions(motion)
    kots.dynamics()

    state = StateType("joint", "joint3", "force")
    actual = outward_total_gradient.outward_jacobian(kots.robot_, kots.state_dict_, [state])
    full = outward_total_gradient.total_coord_to_joint_force_grad_mat(
        kots.robot_, kots.state_dict_, force_order=state.time_order - 2
    )
    joint = kots.robot_.joint(state.owner_name)
    expected = full[joint.id * 6 : (joint.id + 1) * 6, :]

    np.testing.assert_allclose(actual, expected)


def test_outward_jacobian_joint_torque_matches_full_builder():
    kots = Kots.from_json_file(str(MODEL_PATH), order=5)
    motion = np.random.default_rng(6).standard_normal(kots.order() * kots.dof())
    kots.import_motions(motion)
    kots.dynamics()

    state = StateType("joint", "joint3", "torque")
    actual = outward_total_gradient.outward_jacobian(kots.robot_, kots.state_dict_, [state])
    full = outward_total_gradient.total_coord_to_joint_torque_grad_mat(
        kots.robot_, kots.state_dict_, torque_order=state.time_order - 2
    )
    joint = kots.robot_.joint(state.owner_name)
    start = joint.dof_index * (state.time_order - 2)
    stop = (joint.dof_index + joint.dof) * (state.time_order - 2)
    expected = full[start:stop, :]

    np.testing.assert_allclose(actual, expected)
