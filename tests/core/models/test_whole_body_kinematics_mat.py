from pathlib import Path

import numpy as np

from robokots.kots import Kots
from robokots.outward.state import build_kinematics_state
from robokots.core.models.whole_body.total_kinematics_mat import total_world_joint_cmtm


TEST_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = TEST_DIR / "test_model" / "sample_robot.json"


def test_total_world_joint_cmtm_shape():
    kots = Kots.from_json_file(str(MODEL_PATH), order=2)
    kots.import_motions(np.random.rand(kots.order() * kots.dof()))
    state = build_kinematics_state(kots.robot_, kots.motion(kots.order()), kots.order())

    mat = total_world_joint_cmtm(kots.robot_, state, order=1, dim=3)

    expected_size = kots.robot_.joint_num * 6
    assert mat.shape == (expected_size, expected_size)
    assert np.all(np.isfinite(mat))
