from typing import Optional

import numpy as np
import pytest
from pathlib import Path

from robokots.kots import Kots, StateType


TEST_DIR = Path(__file__).resolve().parent
MODEL_PATH = TEST_DIR / "test_model" / "sample_robot.json"
TARGET_PATH = TEST_DIR / "target_list.json"


@pytest.mark.slow
def test_dynamics_jacobians():
    kots = Kots.from_json_file(str(MODEL_PATH))
    kots.set_target_from_file(str(TARGET_PATH))

    motion = np.random.rand(kots.order() * kots.dof())
    kots.import_motions(motion)

    kots.dynamics()

    link_name = kots.target_._targets[0].owner_name

    def link_state(data_type: str, frame_name: Optional[str] = None) -> StateType:
        return StateType("link", link_name, data_type, frame_name)

    def joint_state(
        joint_name: str,
        data_type: str,
        frame_name: Optional[str] = None,
    ) -> StateType:
        return StateType("joint", joint_name, data_type, frame_name)

    # velocity, acceleration, jerk, snap
    np.testing.assert_array_almost_equal(
        kots.state_info(link_state("vel")),
        kots.jacobian(link_state("frame")) @ kots.motion_diff(1),
    )
    np.testing.assert_array_almost_equal(
        kots.state_info(link_state("acc")),
        kots.jacobian(link_state("vel")) @ kots.motion_diff(2),
    )
    np.testing.assert_array_almost_equal(
        kots.state_info(link_state("jerk")),
        kots.jacobian(link_state("acc")) @ kots.motion_diff(3),
    )
    np.testing.assert_array_almost_equal(
        kots.state_info(link_state("snap")),
        kots.jacobian(link_state("jerk")) @ kots.motion_diff(4),
    )

    np.testing.assert_array_almost_equal(
        kots.jacobian(link_state("frame")),
        kots.jacobian(link_state("frame"), numerical=True),
        decimal=6,
    )
    np.testing.assert_array_almost_equal(
        kots.jacobian(link_state("vel")),
        kots.jacobian(link_state("vel"), numerical=True),
        decimal=5,
    )
    np.testing.assert_array_almost_equal(
        kots.jacobian(link_state("acc")),
        kots.jacobian(link_state("acc"), numerical=True),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        kots.jacobian(link_state("jerk")),
        kots.jacobian(link_state("jerk"), numerical=True),
        decimal=3,
    )

    # local link momentum
    np.testing.assert_array_almost_equal(
        kots.state_info(link_state("momentum_diff1")),
        kots.jacobian(link_state("momentum")) @ kots.motion_diff(2),
    )
    np.testing.assert_array_almost_equal(
        kots.state_info(link_state("momentum_diff2")),
        kots.jacobian(link_state("momentum_diff1")) @ kots.motion_diff(3),
    )
    np.testing.assert_array_almost_equal(
        kots.state_info(link_state("momentum_diff3")),
        kots.jacobian(link_state("momentum_diff2")) @ kots.motion_diff(4),
    )
    np.testing.assert_array_almost_equal(
        kots.jacobian(link_state("momentum")),
        kots.jacobian(link_state("momentum"), numerical=True),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        kots.jacobian(link_state("momentum_diff1")),
        kots.jacobian(link_state("momentum_diff1"), numerical=True),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        kots.jacobian(link_state("momentum_diff2")),
        kots.jacobian(link_state("momentum_diff2"), numerical=True),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        kots.jacobian(link_state("momentum_diff3")),
        kots.jacobian(link_state("momentum_diff3"), numerical=True),
        decimal=3,
    )

    # world link momentum
    np.testing.assert_array_almost_equal(
        kots.state_info(link_state("momentum_diff1", "world")),
        kots.jacobian(link_state("momentum", "world")) @ kots.motion_diff(2),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        kots.state_info(link_state("momentum_diff2", "world")),
        kots.jacobian(link_state("momentum_diff1", "world")) @ kots.motion_diff(3),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        kots.state_info(link_state("momentum_diff3", "world")),
        kots.jacobian(link_state("momentum_diff2", "world")) @ kots.motion_diff(4),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        kots.jacobian(link_state("momentum", "world")),
        kots.jacobian(link_state("momentum", "world"), numerical=True),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        kots.jacobian(link_state("momentum_diff1", "world")),
        kots.jacobian(link_state("momentum_diff1", "world"), numerical=True),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        kots.jacobian(link_state("momentum_diff2", "world")),
        kots.jacobian(link_state("momentum_diff2", "world"), numerical=True),
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        kots.jacobian(link_state("momentum_diff3", "world")),
        kots.jacobian(link_state("momentum_diff3", "world"), numerical=True),
        decimal=3,
    )

    # link force
    np.testing.assert_array_almost_equal(
        kots.state_info(link_state("force_diff1")),
        kots.jacobian(link_state("force")) @ kots.motion_diff(3),
    )
    np.testing.assert_array_almost_equal(
        kots.state_info(link_state("force_diff2")),
        kots.jacobian(link_state("force_diff1")) @ kots.motion_diff(4),
    )
    np.testing.assert_array_almost_equal(
        kots.state_info(link_state("force_diff3")),
        kots.jacobian(link_state("force_diff2")) @ kots.motion_diff(5),
    )

    np.testing.assert_array_almost_equal(
        kots.jacobian(link_state("force")),
        kots.jacobian(link_state("force"), numerical=True),
        decimal=3,
    )
    np.testing.assert_array_almost_equal(
        kots.jacobian(link_state("force_diff1")),
        kots.jacobian(link_state("force_diff1"), numerical=True),
        decimal=3,
    )
    np.testing.assert_array_almost_equal(
        kots.jacobian(link_state("force_diff2")),
        kots.jacobian(link_state("force_diff2"), numerical=True),
        decimal=3,
    )

    # joint world momentum
    for joint_name in reversed(kots.joint_name_list()):
        np.testing.assert_array_almost_equal(
            kots.state_info(joint_state(joint_name, "momentum_diff1", "world")),
            kots.jacobian(joint_state(joint_name, "momentum", "world")) @ kots.motion_diff(2),
            decimal=3,
        )
        np.testing.assert_array_almost_equal(
            kots.state_info(joint_state(joint_name, "momentum_diff2", "world")),
            kots.jacobian(joint_state(joint_name, "momentum_diff1", "world")) @ kots.motion_diff(3),
            decimal=3,
        )
        np.testing.assert_array_almost_equal(
            kots.state_info(joint_state(joint_name, "momentum_diff3", "world")),
            kots.jacobian(joint_state(joint_name, "momentum_diff2", "world")) @ kots.motion_diff(4),
            decimal=3,
        )
        np.testing.assert_array_almost_equal(
            kots.jacobian(joint_state(joint_name, "momentum", "world")),
            kots.jacobian(joint_state(joint_name, "momentum", "world"), numerical=True),
            decimal=3,
        )
        np.testing.assert_array_almost_equal(
            kots.jacobian(joint_state(joint_name, "momentum_diff1", "world")),
            kots.jacobian(joint_state(joint_name, "momentum_diff1", "world"), numerical=True),
            decimal=3,
        )
        np.testing.assert_array_almost_equal(
            kots.jacobian(joint_state(joint_name, "momentum_diff2", "world")),
            kots.jacobian(joint_state(joint_name, "momentum_diff2", "world"), numerical=True),
            decimal=3,
        )
        np.testing.assert_array_almost_equal(
            kots.jacobian(joint_state(joint_name, "momentum_diff3", "world")),
            kots.jacobian(joint_state(joint_name, "momentum_diff3", "world"), numerical=True),
            decimal=3,
        )

    # joint momentum
    for joint_name in kots.joint_name_list():
        np.testing.assert_array_almost_equal(
            kots.state_info(joint_state(joint_name, "momentum_diff1")),
            kots.jacobian(joint_state(joint_name, "momentum")) @ kots.motion_diff(2),
            decimal=3,
        )
        np.testing.assert_array_almost_equal(
            kots.state_info(joint_state(joint_name, "momentum_diff2")),
            kots.jacobian(joint_state(joint_name, "momentum_diff1")) @ kots.motion_diff(3),
            decimal=3,
        )
        np.testing.assert_array_almost_equal(
            kots.state_info(joint_state(joint_name, "momentum_diff3")),
            kots.jacobian(joint_state(joint_name, "momentum_diff2")) @ kots.motion_diff(4),
            decimal=3,
        )
        np.testing.assert_array_almost_equal(
            kots.jacobian(joint_state(joint_name, "momentum")),
            kots.jacobian(joint_state(joint_name, "momentum"), numerical=True),
            decimal=3,
        )
        np.testing.assert_array_almost_equal(
            kots.jacobian(joint_state(joint_name, "momentum_diff1")),
            kots.jacobian(joint_state(joint_name, "momentum_diff1"), numerical=True),
            decimal=3,
        )
        np.testing.assert_array_almost_equal(
            kots.jacobian(joint_state(joint_name, "momentum_diff2")),
            kots.jacobian(joint_state(joint_name, "momentum_diff2"), numerical=True),
            decimal=3,
        )
        np.testing.assert_array_almost_equal(
            kots.jacobian(joint_state(joint_name, "momentum_diff3")),
            kots.jacobian(joint_state(joint_name, "momentum_diff3"), numerical=True),
            decimal=3,
        )

    # joint force
    for joint_name in reversed(kots.joint_name_list()):
        np.testing.assert_array_almost_equal(
            kots.state_info(joint_state(joint_name, "force_diff1")),
            kots.jacobian(joint_state(joint_name, "force")) @ kots.motion_diff(3),
            decimal=3,
        )
        np.testing.assert_array_almost_equal(
            kots.state_info(joint_state(joint_name, "force_diff2")),
            kots.jacobian(joint_state(joint_name, "force_diff1")) @ kots.motion_diff(4),
            decimal=3,
        )
        np.testing.assert_array_almost_equal(
            kots.state_info(joint_state(joint_name, "force_diff3")),
            kots.jacobian(joint_state(joint_name, "force_diff2")) @ kots.motion_diff(5),
            decimal=3,
        )

        np.testing.assert_array_almost_equal(
            kots.jacobian(joint_state(joint_name, "force")),
            kots.jacobian(joint_state(joint_name, "force"), numerical=True),
            decimal=3,
        )
        np.testing.assert_array_almost_equal(
            kots.jacobian(joint_state(joint_name, "force_diff1")),
            kots.jacobian(joint_state(joint_name, "force_diff1"), numerical=True),
            decimal=3,
        )
        np.testing.assert_array_almost_equal(
            kots.jacobian(joint_state(joint_name, "force_diff2")),
            kots.jacobian(joint_state(joint_name, "force_diff2"), numerical=True),
            decimal=3,
        )
        np.testing.assert_array_almost_equal(
            kots.jacobian(joint_state(joint_name, "force_diff3")),
            kots.jacobian(joint_state(joint_name, "force_diff3"), numerical=True),
            decimal=2,
        )

    # joint torque
    for joint_name in kots.joint_name_list():
        if kots.robot_.joint_list([joint_name])[0].dof == 0:
            continue

        np.testing.assert_array_almost_equal(
            kots.state_info(joint_state(joint_name, "torque_diff1")),
            kots.jacobian(joint_state(joint_name, "torque")) @ kots.motion_diff(3),
            decimal=3,
        )
        np.testing.assert_array_almost_equal(
            kots.state_info(joint_state(joint_name, "torque_diff2")),
            kots.jacobian(joint_state(joint_name, "torque_diff1")) @ kots.motion_diff(4),
            decimal=3,
        )
        np.testing.assert_array_almost_equal(
            kots.state_info(joint_state(joint_name, "torque_diff3")),
            kots.jacobian(joint_state(joint_name, "torque_diff2")) @ kots.motion_diff(5),
            decimal=3,
        )

        np.testing.assert_array_almost_equal(
            kots.jacobian(joint_state(joint_name, "torque")),
            kots.jacobian(joint_state(joint_name, "torque"), numerical=True),
            decimal=3,
        )
        np.testing.assert_array_almost_equal(
            kots.jacobian(joint_state(joint_name, "torque_diff1")),
            kots.jacobian(joint_state(joint_name, "torque_diff1"), numerical=True),
            decimal=3,
        )
        np.testing.assert_array_almost_equal(
            kots.jacobian(joint_state(joint_name, "torque_diff2")),
            kots.jacobian(joint_state(joint_name, "torque_diff2"), numerical=True),
            decimal=3,
        )
        np.testing.assert_array_almost_equal(
            kots.jacobian(joint_state(joint_name, "torque_diff3")),
            kots.jacobian(joint_state(joint_name, "torque_diff3"), numerical=True),
            decimal=2,
        )
