import numpy as np

from mathrobo import SO3, SE3
from robokots.dynamics import *

'''
Test dynamics function
'''

class MockJoint():
    def __init__(self):
       self.joint_select_mat = np.array([[1, 0, 0, 0, 0, 0]]).T

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
    joint = MockJoint()
    rel_frame = SE3()
    p_joint_force = np.array([1, 2, 3, 4, 5, 6])
    link_force = np.array([7, 8, 9, 10, 11, 12])
    joint_torque, joint_force = joint_dynamics(joint, rel_frame, p_joint_force, link_force)
    expected_force = rel_frame.mat_inv_adj() @ p_joint_force - link_force
    expected_torque = joint.joint_select_mat.T @ expected_force
    # Check the shapes of the outputss
    assert np.allclose(joint_force, expected_force)
    assert np.allclose(joint_torque, expected_torque)

