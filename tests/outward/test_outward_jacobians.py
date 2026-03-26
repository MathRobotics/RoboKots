from pathlib import Path

import numpy as np

from robokots.kots import Kots
from robokots.outward.state import build_kinematics_state
from robokots.outward.diff.outward_jacobians import link_cmtm_jacobian


TEST_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = TEST_DIR / "test_model" / "soft_rod.json"


def test_link_cmtm_jacobian_soft_link_smoke():
    kots = Kots.from_json_file(str(MODEL_PATH), order=3)
    kots.import_motions(np.random.rand(kots.order() * kots.dof()))
    state = build_kinematics_state(kots.robot_, kots.motion(kots.order()), kots.order())

    target_link = kots.link_name_list()[-1]
    jac = link_cmtm_jacobian(
        kots.robot_,
        kots.motions_,
        state,
        [target_link],
        order=kots.order(),
    )

    assert jac.shape == (6 * kots.order(), kots.dof() * kots.order())
    assert np.all(np.isfinite(jac))
