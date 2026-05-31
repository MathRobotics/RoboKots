from pathlib import Path

import numpy as np

from robokots.kots import Kots
from robokots.kots import StateType
from robokots.outward import (
    build_dynamics_cmtm_state,
    build_dynamics_outward_state,
    build_kinematics_outward_state,
    build_kinematics_state,
)
from robokots.outward.diff.outward_total_gradient import outward_jacobian


TEST_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = TEST_DIR / "test_model" / "sample_robot.json"


def _assert_state_dict_allclose(actual: dict, expected: dict):
    assert actual.keys() == expected.keys()
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key])


def test_kinematics_outward_state_exports_existing_state_dict_format():
    kots = Kots.from_json_file(str(MODEL_PATH), order=4)
    motion = np.random.default_rng(0).standard_normal(kots.order() * kots.dof())

    outward_state = build_kinematics_outward_state(kots.robot_, motion, kots.order())
    expected = build_kinematics_state(kots.robot_, motion, kots.order())

    _assert_state_dict_allclose(outward_state.to_state_dict(kots.robot_), expected)


def test_dynamics_outward_state_exports_existing_state_dict_format():
    kots = Kots.from_json_file(str(MODEL_PATH), order=5)
    motion = np.random.default_rng(1).standard_normal(kots.order() * kots.dof())

    outward_state = build_dynamics_outward_state(kots.robot_, motion, kots.order() - 2)
    expected = build_dynamics_cmtm_state(kots.robot_, motion, kots.order() - 2)

    _assert_state_dict_allclose(outward_state.to_state_dict(kots.robot_), expected)


def test_outward_state_reuses_relative_cmtm_cache():
    kots = Kots.from_json_file(str(MODEL_PATH), order=3)
    motion = np.random.default_rng(2).standard_normal(kots.order() * kots.dof())
    outward_state = build_kinematics_outward_state(kots.robot_, motion, kots.order())

    rel0 = outward_state.rel_cmtm("world", "arm3")
    rel1 = outward_state.rel_cmtm("world", "arm3")

    assert rel0 is rel1


def test_outward_jacobian_accepts_outward_state():
    kots = Kots.from_json_file(str(MODEL_PATH), order=5)
    motion = np.random.default_rng(3).standard_normal(kots.order() * kots.dof())
    outward_state = build_dynamics_outward_state(kots.robot_, motion, kots.order() - 2)
    state_dict = outward_state.to_state_dict(kots.robot_)
    states = [
        StateType("link", "arm3", "momentum", "world"),
        StateType("link", "arm3", "force"),
    ]

    actual = outward_jacobian(kots.robot_, outward_state, states)
    expected = outward_jacobian(kots.robot_, state_dict, states)

    np.testing.assert_allclose(actual, expected)


def test_kots_jacobian_uses_outward_state_after_dynamics():
    kots = Kots.from_json_file(str(MODEL_PATH), order=5)
    motion = np.random.default_rng(4).standard_normal(kots.order() * kots.dof())
    kots.import_motions(motion)
    kots.dynamics()

    assert kots.outward_state_ is not None

    state = StateType("joint", "joint3", "torque")
    actual = kots.jacobian(state)
    expected = outward_jacobian(kots.robot_, kots.state_dict_, [state])

    np.testing.assert_allclose(actual, expected)


def test_update_state_dict_caches_kinematics_outward_state():
    kots = Kots.from_json_file(str(MODEL_PATH), order=4)
    motion = np.random.default_rng(5).standard_normal(kots.order() * kots.dof())
    kots.import_motions(motion)

    state_dict = kots.update_state_dict()

    assert kots.outward_state_ is not None
    _assert_state_dict_allclose(state_dict, kots.outward_state_.to_state_dict(kots.robot_))

    cached_outward_state = kots.outward_state_
    cached_state_dict = kots.update_state_dict()

    assert kots.outward_state_ is cached_outward_state
    _assert_state_dict_allclose(cached_state_dict, state_dict)


def test_update_state_defers_state_dict_materialization():
    kots = Kots.from_json_file(str(MODEL_PATH), order=4)
    motion = np.random.default_rng(8).standard_normal(kots.order() * kots.dof())
    kots.import_motions(motion)

    outward_state = kots.update_state()

    assert kots.outward_state_ is outward_state
    assert kots.state_dict_ == {}

    state = StateType("link", "arm3", "jerk")
    expected = outward_state.cmtm("link", "arm3", state.time_order).elem_vecs(state.key_order - 2)
    np.testing.assert_allclose(kots.state_info(state), expected)

    state_dict = kots.to_state_dict()
    _assert_state_dict_allclose(state_dict, outward_state.to_state_dict(kots.robot_))


def test_state_info_reads_outward_state_directly_when_available():
    kots = Kots.from_json_file(str(MODEL_PATH), order=4)
    motion = np.random.default_rng(7).standard_normal(kots.order() * kots.dof())
    kots.import_motions(motion)
    kots.update_state_dict()

    state = StateType("link", "arm3", "jerk")
    expected = kots.outward_state_.cmtm("link", "arm3", state.time_order).elem_vecs(state.key_order - 2)
    kots.state_dict_[state.alliance] = np.full_like(expected, np.nan)

    np.testing.assert_allclose(kots.state_info(state), expected)


def test_update_state_dict_caches_dynamics_outward_state():
    kots = Kots.from_json_file(str(MODEL_PATH), order=5)
    motion = np.random.default_rng(6).standard_normal(kots.order() * kots.dof())
    kots.import_motions(motion)

    state_dict = kots.update_state_dict(is_dynamics=True)

    assert kots.outward_state_ is not None
    _assert_state_dict_allclose(state_dict, kots.outward_state_.to_state_dict(kots.robot_))

    state = StateType("link", "arm3", "momentum", "world")
    actual = kots.jacobian(state)
    expected = outward_jacobian(kots.robot_, kots.state_dict_, [state])

    np.testing.assert_allclose(actual, expected)
