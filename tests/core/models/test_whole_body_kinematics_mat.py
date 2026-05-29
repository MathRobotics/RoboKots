from pathlib import Path

import numpy as np

from robokots.kots import Kots
from robokots.outward.state import build_kinematics_state
from robokots.core.models.whole_body.total_kinematics_mat import total_world_joint_cmtm
from robokots.core.models.whole_body.total_kinematics_grad_mat import (
    total_coord_to_joint_tan_vel_grad_mat,
    total_coord_to_joint_tan_vel_grad_matvec,
    total_coord_to_link_tan_vel_grad_mat,
    total_coord_to_link_tan_vel_grad_matvec,
    total_coord_to_link_vel_grad_mat,
    total_coord_to_link_vel_grad_matvec,
    total_joint_tan_vel_to_link_sp_vel_grad_mat,
    total_joint_tan_vel_to_link_sp_vel_grad_matvec,
)
from robokots.core.models.whole_body.total_dynamics_grad_mat import (
    total_coord_to_link_momentum_grad_mat,
    total_coord_to_link_momentum_grad_matvec,
)


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


def test_total_kinematics_grad_matvec_matches_matrix_product():
    order = 4
    kots = Kots.from_json_file(str(MODEL_PATH), order=order)
    rng = np.random.default_rng(3)
    kots.import_motions(rng.standard_normal(kots.order() * kots.dof()))
    state = build_kinematics_state(kots.robot_, kots.motion(kots.order()), kots.order())

    coord_vec = rng.standard_normal(kots.dof() * order)
    joint_tan_vec = rng.standard_normal(kots.robot_.joint_num * 6 * order)

    np.testing.assert_allclose(
        total_coord_to_joint_tan_vel_grad_matvec(kots.robot_, state, coord_vec, order=order),
        total_coord_to_joint_tan_vel_grad_mat(kots.robot_, state, order=order) @ coord_vec,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        total_coord_to_link_vel_grad_matvec(kots.robot_, state, coord_vec, order=order),
        total_coord_to_link_vel_grad_mat(kots.robot_, state, order=order) @ coord_vec,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        total_coord_to_link_tan_vel_grad_matvec(kots.robot_, state, coord_vec, out_order=order-1, in_order=order),
        total_coord_to_link_tan_vel_grad_mat(kots.robot_, state, out_order=order-1, in_order=order) @ coord_vec,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        total_joint_tan_vel_to_link_sp_vel_grad_matvec(kots.robot_, state, joint_tan_vec, order=order),
        total_joint_tan_vel_to_link_sp_vel_grad_mat(kots.robot_, state, order=order) @ joint_tan_vec,
        atol=1e-10,
        rtol=1e-10,
    )


def test_total_link_momentum_grad_matvec_matches_matrix_product():
    order = 4
    kots = Kots.from_json_file(str(MODEL_PATH), order=order)
    rng = np.random.default_rng(4)
    kots.import_motions(rng.standard_normal(kots.order() * kots.dof()))
    state = build_kinematics_state(kots.robot_, kots.motion(kots.order()), kots.order())

    vec = rng.standard_normal(kots.dof() * order)
    np.testing.assert_allclose(
        total_coord_to_link_momentum_grad_matvec(kots.robot_, state, vec, order=order),
        total_coord_to_link_momentum_grad_mat(kots.robot_, state, order=order) @ vec,
        atol=1e-10,
        rtol=1e-10,
    )
