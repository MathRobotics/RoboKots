from pathlib import Path

from mathrobo import numerical_difference
import numpy as np
import pytest

from robokots.kots import Kots
from robokots.core.state import StateType, data_type_to_sub_func
from robokots.outward.diff.numerical_diff import (
    _make_lifted_update_func,
    diff_outward_numerical,
    link_diff_kinematics_numerical,
)
from robokots.outward.values import compute_outward_value


TEST_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = TEST_DIR / "test_model" / "sample_robot.json"
SOFT_MODEL_PATH = TEST_DIR / "test_model" / "soft_rod.json"


def test_diff_outward_numerical_invalid_order_raises():
    kots = Kots.from_json_file(str(MODEL_PATH), order=3)
    kots.import_motions(np.random.rand(kots.order() * kots.dof()))

    state_type = StateType(owner_type="link", owner_name="arm3", data_type="acc")
    with pytest.raises(ValueError, match="Invalid order"):
        diff_outward_numerical(kots.robot_, kots.motion(3), state_type, order=2)


def test_diff_outward_numerical_soft_link_matches_manual_fd():
    rng = np.random.default_rng(0)
    kots = Kots.from_json_file(str(SOFT_MODEL_PATH), order=3)
    kots.import_motions(rng.standard_normal(kots.order() * kots.dof()))

    robot = kots.robot_
    motion = kots.motion(3)
    state_type = StateType(owner_type="link", owner_name="rod1", data_type="frame")
    direction = np.zeros(robot.dof)
    direction[robot.link("rod1").dof_index] = 1.0
    eps = 1e-8

    update_func = _make_lifted_update_func(robot.dof, 3, update_method="poly")

    def func(x):
        return compute_outward_value(robot, x, state_type, input_order=3)

    expected = numerical_difference(
        motion,
        func,
        sub_func=data_type_to_sub_func(state_type.data_type),
        update_func=update_func,
        direction=direction,
        eps=eps,
    )
    actual = diff_outward_numerical(
        robot,
        motion,
        state_type,
        order=3,
        eps=eps,
        update_method="poly",
        update_direction=direction,
    )

    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert np.linalg.norm(actual) > 1e-8


@pytest.mark.parametrize("data_type", ["pos", "frame"])
def test_link_diff_kinematics_numerical_multi_link_matches_single(data_type: str):
    rng = np.random.default_rng(1)
    kots = Kots.from_json_file(str(MODEL_PATH), order=3)
    kots.import_motions(rng.standard_normal(kots.order() * kots.dof()))

    robot = kots.robot_
    motion = kots.motion(3)
    links = ["arm1", "arm2", "arm3"]
    direction = rng.standard_normal(robot.dof)

    actual = link_diff_kinematics_numerical(
        robot,
        motion,
        links,
        data_type=data_type,
        order=3,
        eps=1e-8,
        update_method="poly",
        update_direction=direction,
    )

    expected = np.vstack(
        [
            link_diff_kinematics_numerical(
                robot,
                motion,
                [name],
                data_type=data_type,
                order=3,
                eps=1e-8,
                update_method="poly",
                update_direction=direction,
            )[0]
            for name in links
        ]
    )

    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)
