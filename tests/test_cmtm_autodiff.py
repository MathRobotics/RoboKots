from math import factorial
from pathlib import Path

import jax
import numpy as np
import pytest
from mathrobo import CMTM, SE3

from developer.benchmarks.cmtm_autodiff import make_cmtm_ad_value, relative_cmtms
from developer.benchmarks.jacobian_accuracy_table import make_model
from robokots.kots import Kots, StateType


def test_cmtm_blocks_match_mathrobo():
    jax.config.update("jax_enable_x64", True)
    robot = Kots.from_json_data(make_model(2), order=7).robot_
    joint = robot.joint("joint1")
    q = np.array([.4])
    derivatives = np.random.default_rng(9).normal(size=(6, 1)) @ joint.select_mat.T
    coefficients = derivatives / np.array([factorial(i) for i in range(6)])[:, None]
    x, y = relative_cmtms(joint, q, coefficients)
    pose = joint.origin @ SE3.set_mat(SE3.exp(joint.select_mat[:, 0] * q[0]))
    reference = CMTM(pose, derivatives)
    np.testing.assert_allclose(x, reference.mat_inv_adj(output_order=6), atol=1e-12)
    adjoint = reference.mat_adj(output_order=6)
    for i in range(6):
        # Wrench transform is the transpose of each inverse-adjoint block.
        np.testing.assert_allclose(y[6*i:6*(i+1), :6],
                                   x[6*i:6*(i+1), :6].T, atol=1e-12)
    np.testing.assert_allclose(np.asarray(x) @ adjoint, np.eye(36), atol=1e-12)


@pytest.mark.parametrize("k", [0, 1, 4])
def test_cmtm_values_and_outer_ad(k):
    jax.config.update("jax_enable_x64", True)
    order = k + 3
    robot = Kots.from_json_data(make_model(2), order=order)
    state = StateType("joint", "joint1", "torque" + (f"_diff{k}" if k else ""))
    gravity = [.3, -.4, -9.81]
    value = make_cmtm_ad_value(robot.robot_, state, order, gravity)
    forward, reverse = jax.jit(jax.jacfwd(value)), jax.jit(jax.jacrev(value))
    for motion in (np.zeros(2 * order), np.random.default_rng(39).normal(scale=.3, size=2 * order)):
        robot.import_motions(motion)
        robot.dynamics(gravity=gravity)
        np.testing.assert_allclose(value(motion), robot.state_info(state), atol=1e-9, rtol=1e-9)
        reference = robot.jacobian(state)
        np.testing.assert_allclose(forward(motion), reference, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(reverse(motion), reference, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(forward(motion), robot.jacobian(state, numerical=True), atol=2e-5, rtol=2e-5)
    jax.clear_caches()


def test_cmtm_branched_fixed_prismatic(tmp_path):
    jax.config.update("jax_enable_x64", True)
    source = Path(__file__).parent / "test_model/branched_fixed.urdf"
    path = tmp_path / "prismatic.urdf"
    path.write_text(source.read_text().replace('name="a_elbow" type="revolute"', 'name="a_elbow" type="prismatic"'))
    robot = Kots.from_urdf_file(str(path), order=4)
    state = StateType("joint", "a_shoulder", "torque_diff1")
    motion = np.random.default_rng(5).normal(scale=.3, size=robot.dof() * 4)
    gravity = [.3, -.4, -9.81]
    robot.import_motions(motion)
    robot.dynamics(gravity=gravity)
    value = make_cmtm_ad_value(robot.robot_, state, 4, gravity)
    np.testing.assert_allclose(value(motion), robot.state_info(state), atol=1e-10)
    np.testing.assert_allclose(jax.jacfwd(value)(motion), robot.jacobian(state), atol=1e-10)
    with pytest.raises(ValueError, match="Insufficient"):
        make_cmtm_ad_value(robot.robot_, state, 3)
