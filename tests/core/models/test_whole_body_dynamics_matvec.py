from pathlib import Path

import numpy as np

from robokots.kots import Kots
from robokots.core.models.whole_body.total_dynamics_mat import (
    total_joint_wrench_to_joint_torque_mat,
    total_joint_wrench_to_joint_torque_matvec,
    total_world_link_wrench_to_world_joint_wrench_mat,
    total_world_link_wrench_to_world_joint_wrench_matvec,
)
from robokots.core.models.whole_body.total_partial_grad_mat import (
    total_partial_link_momentum_to_world_link_momentum_grad_mat,
    total_partial_link_momentum_to_world_link_momentum_grad_matvec,
    total_partial_link_sp_vel_to_joint_force_grad_mat,
    total_partial_link_sp_vel_to_joint_force_grad_matvec,
    total_partial_link_sp_vel_to_link_force_grad_mat,
    total_partial_link_sp_vel_to_link_force_grad_matvec,
    total_partial_link_tan_vel_to_joint_momentum_grad_mat,
    total_partial_link_tan_vel_to_joint_momentum_grad_matvec,
    total_partial_link_tan_vel_to_world_link_momentum_grad_mat,
    total_partial_link_tan_vel_to_world_link_momentum_grad_matvec,
    total_partial_momentum_to_force_grad_mat,
    total_partial_momentum_to_force_grad_matvec,
    total_partial_world_joint_momentum_to_joint_momentum_grad_mat,
    total_partial_world_joint_momentum_to_joint_momentum_grad_matvec,
)
from robokots.outward.diff.outward_transpose_matvec import (
    _transpose_total_joint_wrench_to_joint_torque_matvec,
    _transpose_total_partial_link_momentum_to_world_link_momentum_grad_matvec,
    _transpose_total_partial_link_sp_vel_to_joint_force_grad_matvec,
    _transpose_total_partial_link_sp_vel_to_link_force_grad_matvec,
    _transpose_total_partial_link_tan_vel_to_joint_momentum_grad_matvec,
    _transpose_total_partial_link_tan_vel_to_world_link_momentum_grad_matvec,
    _transpose_total_partial_momentum_to_force_grad_matvec,
    _transpose_total_partial_world_joint_momentum_to_joint_momentum_grad_matvec,
    _transpose_total_world_link_wrench_to_world_joint_wrench_matvec,
)


TEST_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = TEST_DIR / "test_model" / "sample_robot.json"


def test_total_dynamics_matvec_matches_matrix_product():
    order = 4
    force_order = order - 2
    momentum_order = order - 1
    dim_dof = 6

    kots = Kots.from_json_file(str(MODEL_PATH), order=order)
    rng = np.random.default_rng(5)
    kots.import_motions(rng.standard_normal(kots.order() * kots.dof()))
    kots.dynamics()

    robot = kots.robot_
    state = kots.state_dict_

    link_momentum_vec = rng.standard_normal(robot.link_num * dim_dof * momentum_order)
    joint_momentum_vec = rng.standard_normal(robot.joint_num * dim_dof * momentum_order)
    joint_force_vec = rng.standard_normal(robot.joint_num * dim_dof * force_order)

    np.testing.assert_allclose(
        total_world_link_wrench_to_world_joint_wrench_matvec(robot, link_momentum_vec, momentum_order),
        total_world_link_wrench_to_world_joint_wrench_mat(robot, momentum_order) @ link_momentum_vec,
        atol=1e-10,
        rtol=1e-10,
    )


def test_total_dynamics_transpose_matvec_matches_matrix_product():
    order = 4
    force_order = order - 2
    momentum_order = order - 1
    dim_dof = 6

    kots = Kots.from_json_file(str(MODEL_PATH), order=order)
    rng = np.random.default_rng(18)
    kots.import_motions(rng.standard_normal(kots.order() * kots.dof()))
    kots.dynamics()

    robot = kots.robot_
    state = kots.state_dict_

    link_to_joint_mat = total_world_link_wrench_to_world_joint_wrench_mat(robot, momentum_order)
    joint_torque_mat = total_joint_wrench_to_joint_torque_mat(robot, force_order)
    link_mom_to_world_mat = total_partial_link_momentum_to_world_link_momentum_grad_mat(robot, state, order)
    link_tan_to_world_mat = total_partial_link_tan_vel_to_world_link_momentum_grad_mat(robot, state, order)
    joint_world_to_mom_mat = total_partial_world_joint_momentum_to_joint_momentum_grad_mat(robot, state, order)
    link_tan_to_joint_mat = total_partial_link_tan_vel_to_joint_momentum_grad_mat(robot, state, order)
    mom_to_force_mat = total_partial_momentum_to_force_grad_mat(robot, state, force_order)
    link_sp_to_force_mat = total_partial_link_sp_vel_to_link_force_grad_mat(robot, state, force_order)
    joint_sp_to_force_mat = total_partial_link_sp_vel_to_joint_force_grad_mat(robot, state, force_order)

    joint_momentum_vec = rng.standard_normal(robot.joint_num * dim_dof * momentum_order)
    joint_torque_vec = rng.standard_normal(robot.dof * force_order)
    link_momentum_vec = rng.standard_normal(robot.link_num * dim_dof * momentum_order)
    link_force_vec = rng.standard_normal(robot.link_num * dim_dof * force_order)
    joint_force_vec = rng.standard_normal(robot.joint_num * dim_dof * force_order)

    np.testing.assert_allclose(
        _transpose_total_world_link_wrench_to_world_joint_wrench_matvec(robot, joint_momentum_vec, momentum_order),
        link_to_joint_mat.T @ joint_momentum_vec,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        _transpose_total_joint_wrench_to_joint_torque_matvec(robot, joint_torque_vec, force_order),
        joint_torque_mat.T @ joint_torque_vec,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        _transpose_total_partial_link_momentum_to_world_link_momentum_grad_matvec(robot, state, link_momentum_vec, order),
        link_mom_to_world_mat.T @ link_momentum_vec,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        _transpose_total_partial_link_tan_vel_to_world_link_momentum_grad_matvec(robot, state, link_momentum_vec, order),
        link_tan_to_world_mat.T @ link_momentum_vec,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        _transpose_total_partial_world_joint_momentum_to_joint_momentum_grad_matvec(robot, state, joint_momentum_vec, order),
        joint_world_to_mom_mat.T @ joint_momentum_vec,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        _transpose_total_partial_link_tan_vel_to_joint_momentum_grad_matvec(robot, state, joint_momentum_vec, order),
        link_tan_to_joint_mat.T @ joint_momentum_vec,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        _transpose_total_partial_momentum_to_force_grad_matvec(robot, state, link_force_vec, force_order),
        mom_to_force_mat.T @ link_force_vec,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        _transpose_total_partial_link_sp_vel_to_link_force_grad_matvec(robot, state, link_force_vec, force_order),
        link_sp_to_force_mat.T @ link_force_vec,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        _transpose_total_partial_link_sp_vel_to_joint_force_grad_matvec(robot, state, joint_force_vec, force_order),
        joint_sp_to_force_mat.T @ joint_force_vec,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        total_joint_wrench_to_joint_torque_matvec(robot, joint_force_vec, force_order),
        total_joint_wrench_to_joint_torque_mat(robot, force_order) @ joint_force_vec,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        total_partial_link_momentum_to_world_link_momentum_grad_matvec(robot, state, link_momentum_vec, order),
        total_partial_link_momentum_to_world_link_momentum_grad_mat(robot, state, order) @ link_momentum_vec,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        total_partial_link_tan_vel_to_world_link_momentum_grad_matvec(robot, state, link_momentum_vec, order),
        total_partial_link_tan_vel_to_world_link_momentum_grad_mat(robot, state, order) @ link_momentum_vec,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        total_partial_world_joint_momentum_to_joint_momentum_grad_matvec(robot, state, joint_momentum_vec, order),
        total_partial_world_joint_momentum_to_joint_momentum_grad_mat(robot, state, order) @ joint_momentum_vec,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        total_partial_link_tan_vel_to_joint_momentum_grad_matvec(robot, state, joint_momentum_vec, order),
        total_partial_link_tan_vel_to_joint_momentum_grad_mat(robot, state, order) @ joint_momentum_vec,
        atol=1e-10,
        rtol=1e-10,
    )

    link_force_input = rng.standard_normal(robot.link_num * dim_dof * (force_order + 1))
    link_sp_input = rng.standard_normal(robot.link_num * dim_dof * (force_order + 2))
    joint_sp_input = rng.standard_normal(robot.joint_num * dim_dof * (force_order + 2))

    np.testing.assert_allclose(
        total_partial_momentum_to_force_grad_matvec(robot, state, link_force_input, force_order),
        total_partial_momentum_to_force_grad_mat(robot, state, force_order) @ link_force_input,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        total_partial_link_sp_vel_to_link_force_grad_matvec(robot, state, link_sp_input, force_order),
        total_partial_link_sp_vel_to_link_force_grad_mat(robot, state, force_order) @ link_sp_input,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        total_partial_link_sp_vel_to_joint_force_grad_matvec(robot, state, joint_sp_input, force_order),
        total_partial_link_sp_vel_to_joint_force_grad_mat(robot, state, force_order) @ joint_sp_input,
        atol=1e-10,
        rtol=1e-10,
    )
