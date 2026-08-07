import numpy as np
from types import SimpleNamespace

from mathrobo import CMVector, SO3, SE3
from robokots.core.models.dynamics import *
from robokots.core.models.kinematics.kinematics import local_tangent_mat

'''
Test dynamics function
'''

def test_inertia():
    i_vec = np.array([1, 2, 3, 4, 5, 6])
    i = inertia(i_vec)
    # Check the shape of the output
    assert i.shape == (3, 3)
    # Check the values of the output
    assert np.allclose(i[0,0], 1)
    assert np.allclose(i[1,1], 2)
    assert np.allclose(i[2,2], 3)
    assert np.allclose(i[0,1], 4)
    assert np.allclose(i[0,2], 5)
    assert np.allclose(i[1,2], 6)

def test_spatial_inertia():
    m = 2
    i_vec = np.array([1, 2, 3, 4, 5, 6])
    c = np.array([1, 2, 3])
    inertia_matrix = spatial_inertia(m, i_vec, c)
    # Check the shape of the output
    assert inertia_matrix.shape == (6, 6)
    # Check the values of the output
    assert np.allclose(inertia_matrix[0:3,0:3], inertia(i_vec) - SO3.hat(c) @ SO3.hat(c))
    assert np.allclose(inertia_matrix[3:6,3:6], m * np.eye(3))
    assert np.allclose(inertia_matrix[0:3,3:6], -m * SO3.hat(c))
    assert np.allclose(inertia_matrix[3:6,0:3], m * SO3.hat(c))

def test_link_dynamics():
    inertia_matrix = np.eye(6)
    veloc = np.array([1, 2, 3, 4, 5, 6])
    accel = np.array([7, 8, 9, 10, 11, 12])
    force = link_dynamics(inertia_matrix, veloc, accel)
    expected_force = inertia_matrix @ accel - SE3.hat_adj(veloc).T @ inertia_matrix @ veloc
    # Check the shapes of the outputs
    assert force.shape == (6,)
    # Check the values of the outputs
    assert np.allclose(force, expected_force)

def test_joint_dynamics():
    select_mat = np.array([[1, 0, 0, 0, 0, 0]]).T
    rel_frame = SE3()
    p_joint_force = np.array([1, 2, 3, 4, 5, 6])
    link_force = np.array([7, 8, 9, 10, 11, 12])
    joint_torque, joint_force = joint_dynamics(select_mat, rel_frame, p_joint_force, link_force)
    expected_force = rel_frame.mat_inv_adj() @ p_joint_force - link_force
    expected_torque = select_mat.T @ expected_force
    # Check the shapes of the outputss
    assert np.allclose(joint_force, expected_force)
    assert np.allclose(joint_torque, expected_torque)


def test_joint_project_wrench_specializes_one_dof_joints():
    wrench = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    revolute = SimpleNamespace(
        type="revolute",
        dof=1,
        axis=np.array([0.0, 0.0, 1.0]),
        select_mat=np.array([[0.0], [0.0], [1.0], [0.0], [0.0], [0.0]]),
    )
    prismatic = SimpleNamespace(
        type="prismatic",
        dof=1,
        axis=np.array([0.0, 1.0, 0.0]),
        select_mat=np.array([[0.0], [0.0], [0.0], [0.0], [1.0], [0.0]]),
    )

    assert np.allclose(joint_project_wrench(revolute, wrench), np.array([3.0]))
    assert np.allclose(joint_project_wrench(prismatic, wrench), np.array([5.0]))


def test_joint_project_wrench_falls_back_to_motion_subspace():
    wrench = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    joint = SimpleNamespace(
        type="custom",
        dof=2,
        axis=np.array([1.0, 0.0, 0.0]),
        select_mat=np.array(
            [
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
    )

    assert np.allclose(joint_project_wrench(joint, wrench), np.array([[1.0, 4.0]]))


def test_joint_project_wrench_supports_spherical_and_floating_motion_subspaces():
    wrench = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    spherical = SimpleNamespace(
        type="spherical",
        dof=3,
        axis=np.array([1.0, 0.0, 0.0]),
        select_mat=np.vstack([np.eye(3), np.zeros((3, 3))]),
    )
    floating = SimpleNamespace(
        type="floating",
        dof=6,
        axis=np.array([1.0, 0.0, 0.0]),
        select_mat=np.eye(6),
    )

    assert np.allclose(joint_project_wrench(spherical, wrench), np.array([1.0, 2.0, 3.0]))
    assert np.allclose(joint_project_wrench(floating, wrench), wrench)


def test_joint_project_wrench_uses_coordinate_dependent_tangent_map():
    wrench = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    joint_coord = np.array([0.4, -0.2, 0.3])
    spherical = SimpleNamespace(
        type="spherical",
        dof=3,
        axis=np.eye(3),
        select_mat=np.vstack([np.eye(3), np.zeros((3, 3))]),
    )

    expected = wrench @ local_tangent_mat(spherical.select_mat, joint_coord)
    result = joint_project_wrench(spherical, wrench, joint_coord)

    assert not np.allclose(joint_project_wrench(spherical, wrench), result)
    assert np.allclose(result, expected)


def test_link_force_cmvec_preserves_row_dimension():
    vel = CMVector(np.arange(30, dtype=float).reshape(5, 6) / 10.0)
    momentum = CMVector(np.arange(30, 60, dtype=float).reshape(5, 6) / 10.0)

    force = link_force_cmvec(vel, momentum)

    assert isinstance(force, CMVector)
    assert force.vecs().shape == (4, 6)
