import numpy as np

import mathrobo as mr
from robokots.kinematics import *

# Define a mock joint object with a select_mat attribute
class MockJoint:
    def __init__(self, select_mat):
        self.select_mat = select_mat
        self.origin = SE3.eye() # Identity matrix for simplicity
        self.dof = select_mat.shape[1]

delta = 1e-8 # for numercal difference

def test_joint_local_frame():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)

    # Test with non-zero joint_coord
    joint_coord = np.random.rand(1)
    expected_frame = SE3.set_mat(SE3.exp(joint.select_mat @ joint_coord))
    result_frame = joint_local_frame(joint, joint_coord)
    assert np.allclose(result_frame.mat(), expected_frame.mat())
    
def test_joint_local_vel():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)

    # Test with non-zero joint_vel
    joint_vel = np.random.rand(1)
    expected_vel = joint.select_mat @ joint_vel
    result_vel = joint_local_vel(joint, joint_vel)
    assert np.allclose(result_vel, expected_vel)

def test_joint_local_vel_numerical_1dof():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)  

    # Test with non-zero joint_vel
    joint_vel = np.random.rand(1)
    result_vel = joint_local_vel(joint, joint_vel)

    # Calculate numerical velocity
    joint_coord = np.random.rand(1)
    h0 = SE3.set_mat(SE3.exp(joint.select_mat @ joint_coord))
    joint_coord += joint_vel * delta
    h1 = SE3.set_mat(SE3.exp(joint.select_mat @ joint_coord))
    expected_vel = mr.SE3.sub_tan_vec(h0, h1, "bframe") / delta

    assert np.allclose(expected_vel, result_vel)
    
def test_joint_local_acc():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)

    # Test with non-zero joint_acc
    joint_acc = np.random.rand(1)
    expected_acc = joint.select_mat @ joint_acc
    result_acc = joint_local_acc(joint, joint_acc)
    assert np.allclose(result_acc, expected_acc)

def test_joint_local_acc_numerical():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)  

    # Test with non-zero joint_vel
    joint_acc = np.random.rand(1)
    result_acc = joint_local_vel(joint, joint_acc)

    # Calculate numerical velocity
    joint_vel = np.random.rand(1)
    v0 = joint_local_vel(joint, joint_vel)
    joint_vel += joint_acc * delta
    v1 = joint_local_vel(joint, joint_vel)
    expected_acc = (v1 - v0) / delta

    assert np.allclose(expected_acc, result_acc)
    
def test_link_rel_frame():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)
    # Test with non-zero joint_coord  
    joint_coord = np.random.rand(1)
    expected_frame = joint.origin @ SE3.set_mat(SE3.exp(joint.select_mat @ joint_coord))
    result_frame = link_rel_frame(joint, joint_coord)

    assert np.allclose(result_frame.mat(), expected_frame.mat())
    
def test_link_rel_vel():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)

    # Test with non-zero joint_vel
    joint_vel = np.random.rand(1)
    expected_vel = joint.origin.mat_inv_adj() @ (joint.select_mat @ joint_vel)
    result_vel = link_rel_vel(joint, joint_vel)
    assert np.allclose(result_vel, expected_vel)

def test_link_rel_vel_numerical():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)

    # Test with non-zero joint_vel
    joint_vel = np.random.rand(1)
    result_vel = link_rel_vel(joint, joint_vel)

    # Calculate numerical relative velocity
    joint_coord = np.random.rand(1)
    h0 = link_rel_frame(joint, joint_coord)
    joint_coord += joint_vel * delta
    h1 = link_rel_frame(joint, joint_coord)
    expected_vel = mr.SE3.sub_tan_vec(h0, h1, "bframe") / delta

    assert np.allclose(result_vel, expected_vel)
    
def test_link_rel_acc():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)

    # Test with non-zero joint_acc
    joint_acc = np.random.rand(1)
    expected_acc = joint.origin.mat_inv_adj() @ (joint.select_mat @ joint_acc)
    result_acc = link_rel_acc(joint, joint_acc)
    assert np.allclose(result_acc, expected_acc)

def test_joint_local_acc_numerical():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)  

    # Test with non-zero joint_vel
    joint_acc = np.random.rand(1)
    result_acc = link_rel_acc(joint, joint_acc)

    # Calculate numerical acceleration
    joint_vel = np.random.rand(1)
    v0 = link_rel_vel(joint, joint_vel)
    joint_vel += joint_acc * delta
    v1 = link_rel_vel(joint, joint_vel)
    expected_acc = (v1 - v0) / delta

    assert np.allclose(expected_acc, result_acc)
    
def test_kinematics():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)
    
    # Test with non-zero joint_coord
    joint_coord = np.random.rand(1)
    p_link_frame = SE3.rand()
    expected_frame = p_link_frame @ (joint.origin @ SE3.set_mat(SE3.exp(joint.select_mat @ joint_coord)))
    result_frame = kinematics(joint, p_link_frame, joint_coord)
    assert np.allclose(result_frame.mat(), expected_frame.mat())
    
def test_vel_kinematics():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)

    # Test with non-zero joint_coord and joint_veloc
    joint_coord = np.random.rand(1)
    joint_veloc = np.random.rand(1)
    p_link_vel = np.random.rand(6)
    rel_frame = (joint.origin @ SE3.set_mat(SE3.exp(joint.select_mat @ joint_coord)))
    rel_vel = joint.origin.mat_inv_adj() @ (joint.select_mat @ joint_veloc)
    expected_vel = rel_frame.mat_inv_adj() @ p_link_vel + rel_vel
    result_vel = vel_kinematics(joint, p_link_vel, joint_coord, joint_veloc)
    assert np.allclose(result_vel, expected_vel)

def test_vel_kinematics_numerical():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)

    # Test with non-zero joint_coord and joint_veloc
    joint_coord = np.random.rand(1)
    joint_veloc = np.random.rand(1)
    p_link_vel = np.zeros(6)
    result_vel = vel_kinematics(joint, p_link_vel, joint_coord, joint_veloc)

    # Calculate numerical relative velocity
    p_link_frame = SE3.rand()
    rel_frame = link_rel_frame(joint, joint_coord)
    h0 = kinematics(joint, p_link_frame, joint_coord)
    joint_coord += joint_veloc * delta
    h1 = kinematics(joint, p_link_frame, joint_coord)
    expected_vel = mr.SE3.sub_tan_vec(h0, h1, "bframe") / delta
    assert np.allclose(result_vel, expected_vel)
    
def test_acc_kinematics():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)

    # Test with non-zero joint_coord, joint_veloc, and joint_accel
    joint_coord = np.random.rand(1)
    joint_veloc = np.random.rand(1)
    joint_accel = np.random.rand(1)
    p_link_vel = np.random.rand(6)
    p_link_acc = np.random.rand(6)
    rel_frame = joint.origin @ SE3.set_mat(SE3.exp(joint.select_mat @ joint_coord))
    rel_vel = joint.origin.mat_inv_adj() @ (joint.select_mat @ joint_veloc)
    rel_acc = joint.origin.mat_inv_adj() @ (joint.select_mat @ joint_accel)
    expected_acc = rel_frame.mat_inv_adj() @ p_link_acc + \
                   SE3.hat_adj(rel_frame @ rel_vel) @ p_link_vel + rel_acc
    result_acc = acc_kinematics(joint, p_link_vel, p_link_acc, joint_coord, joint_veloc, joint_accel)
    assert np.allclose(result_acc, expected_acc)

def test_acc_kinematics_numerical():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)

    # Test with non-zero joint_coord, joint_veloc, and joint_accel
    joint_coord = np.random.rand(1)
    joint_veloc = np.random.rand(1)
    joint_accel = np.random.rand(1)
    p_link_vel = np.zeros(6)
    p_link_acc = np.zeros(6)
    rel_frame = joint.origin @ SE3.set_mat(SE3.exp(joint.select_mat @ joint_coord))
    rel_vel = joint.origin.mat_inv_adj() @ (joint.select_mat @ joint_veloc)
    rel_acc = joint.origin.mat_inv_adj() @ (joint.select_mat @ joint_accel)
    result_acc = acc_kinematics(joint, p_link_vel, p_link_acc, joint_coord, joint_veloc, joint_accel)

    # Calculate numerical acceleration
    rel_frame = link_rel_frame(joint, joint_coord)
    rel_veloc = link_rel_vel(joint, joint_veloc)

    v0 = vel_kinematics(joint, p_link_vel, joint_coord, joint_veloc)
    joint_coord += joint_coord * delta
    joint_veloc += joint_accel * delta
    v1 =  vel_kinematics(joint, p_link_vel, joint_coord, joint_veloc)
    expected_acc = (v1 - v0) / delta

    assert np.allclose(result_acc, expected_acc)
    
def test_part_link_jacob():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)
    
    # Test with a specific rel_frame
    rel_frame = SE3.set_mat(np.eye(4)) # Identity matrix for simplicity
    expected_jacob = rel_frame.mat_inv_adj() @ joint.origin.mat_inv_adj() @ joint.select_mat
    result_jacob = part_link_jacob(joint, rel_frame)
    assert np.allclose(result_jacob, expected_jacob)

def test_rel_cmtm():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)

    # Test with non-zero joint_coord, joint_veloc, and joint_accel
    joint_coord = np.random.rand(1)
    joint_veloc = np.random.rand(1)
    joint_accel = np.random.rand(1)
    expected_frame = joint.origin @ SE3.set_mat(SE3.exp(joint.select_mat @ joint_coord))
    expected_vel = joint.origin.mat_inv_adj() @ (joint.select_mat @ joint_veloc)
    expected_acc = joint.origin.mat_inv_adj() @ (joint.select_mat @ joint_accel)

    joint_motions = np.array([joint_coord, joint_veloc, joint_accel])
    result_cmtm = link_rel_cmtm(joint, joint_motions)

    assert np.allclose(result_cmtm.elem_mat(), expected_frame.mat())
    assert np.allclose(result_cmtm.elem_vecs(0), expected_vel)
    assert np.allclose(result_cmtm.elem_vecs(1), expected_acc)

def test_kinematics_cmtm():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)

    # Test with non-zero joint_coord, joint_veloc, and joint_accel
    joint_coord = np.random.rand(1)
    joint_veloc = np.random.rand(1)
    joint_accel = np.random.rand(1)
    p_link_frame = SE3.rand()
    p_link_vel = np.zeros(6)
    p_link_acc = np.zeros(6)
    p_link_cmtm = CMTM[SE3](p_link_frame, np.array((p_link_vel, p_link_acc)))

    rel_frame = joint.origin @ SE3.set_mat(SE3.exp(joint.select_mat @ joint_coord))
    rel_veloc = joint.origin.mat_inv_adj() @ (joint.select_mat @ joint_veloc)
    rel_accel = joint.origin.mat_inv_adj() @ (joint.select_mat @ joint_accel)

    expected_frame = p_link_frame @ rel_frame
    expected_vel = rel_frame.mat_inv_adj() @ p_link_vel + rel_veloc
    expected_acc = rel_frame.mat_inv_adj() @ p_link_acc + \
                   SE3.hat_adj(rel_frame @ rel_veloc) @ p_link_vel + rel_accel

    joint_motions = np.array([joint_coord, joint_veloc, joint_accel])
    result_cmtm = kinematics_cmtm(joint, p_link_cmtm, joint_motions)

    assert np.allclose(result_cmtm.elem_mat(), expected_frame.mat())
    assert np.allclose(result_cmtm.elem_vecs(0), expected_vel)
    assert np.allclose(result_cmtm.elem_vecs(1), expected_acc)
    
def test_kinematics_():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)

    # Test with non-zero joint_coord, joint_veloc, and joint_accel
    joint_coord = np.random.rand(1)
    joint_veloc = np.random.rand(1)
    joint_accel = np.random.rand(1)
    p_link_frame = SE3.rand()
    p_link_vel = np.random.rand(6)
    p_link_acc = np.random.rand(6)

    p_link_cmtm = CMTM[SE3](p_link_frame, np.array((p_link_vel, p_link_acc)))

    expected_frame = kinematics(joint, p_link_frame, joint_coord)
    expected_vel = vel_kinematics(joint, p_link_vel, joint_coord, joint_veloc)
    expected_acc = acc_kinematics(joint, p_link_vel, p_link_acc, joint_coord, joint_veloc, joint_accel)

    joint_motions = np.array([joint_coord, joint_veloc, joint_accel])
    result_cmtm = kinematics_cmtm(joint, p_link_cmtm, joint_motions)

    assert np.allclose(result_cmtm.elem_mat(), expected_frame.mat())
    assert np.allclose(result_cmtm.elem_vecs(0), expected_vel)
    assert np.allclose(result_cmtm.elem_vecs(1), expected_acc)