import numpy as np

from robokots.kinematics import *

# Define a mock joint object with a joint_select_mat attribute
class MockJoint:
    def __init__(self, joint_select_mat):
        self.joint_select_mat = joint_select_mat
        self.origin = SE3.set_mat(np.eye(4)) # Identity matrix for simplicity

def test_joint_local_frame():
    # Create a mock joint with a specific joint_select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)

    # Test with non-zero joint_coord
    joint_coord = np.array([np.pi/4])
    expected_frame = SE3.set_mat(SE3.exp(joint.joint_select_mat @ joint_coord))
    result_frame = joint_local_frame(joint, joint_coord)
    assert np.allclose(result_frame.mat(), expected_frame.mat())
    
def test_joint_local_vel():
    # Create a mock joint with a specific joint_select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)

    # Test with non-zero joint_vel
    joint_vel = np.array([1.0])
    expected_vel = joint.joint_select_mat @ joint_vel
    result_vel = joint_local_vel(joint, joint_vel)
    assert np.allclose(result_vel, expected_vel)
    
def test_joint_local_acc():
    # Create a mock joint with a specific joint_select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)

    # Test with non-zero joint_acc
    joint_acc = np.array([1.0])
    expected_acc = joint.joint_select_mat @ joint_acc
    result_acc = joint_local_acc(joint, joint_acc)
    assert np.allclose(result_acc, expected_acc)
    
def test_link_rel_frame():
    # Create a mock joint with a specific joint_select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)
    # Test with non-zero joint_coord  
    joint_angle = np.array([np.pi/4])
    expected_frame = joint.origin @ SE3.set_mat(SE3.exp(joint.joint_select_mat @ joint_angle))
    print("expected_frame", expected_frame.mat())
    result_frame = link_rel_frame(joint, joint_angle)
    print("result_frame", result_frame.mat())
    assert np.allclose(result_frame.mat(), expected_frame.mat())
    
def test_link_rel_vel():
    # Create a mock joint with a specific joint_select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)

    # Test with non-zero joint_vel
    joint_vel = np.array([1.0])
    expected_vel = joint.origin.mat_inv_adj() @ (joint.joint_select_mat @ joint_vel)
    result_vel = link_rel_vel(joint, joint_vel)
    assert np.allclose(result_vel, expected_vel)
    
def test_link_rel_acc():
    # Create a mock joint with a specific joint_select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)

    # Test with non-zero joint_acc
    joint_acc = np.array([1.0])
    expected_acc = joint.origin.mat_inv_adj() @ (joint.joint_select_mat @ joint_acc)
    result_acc = link_rel_acc(joint, joint_acc)
    assert np.allclose(result_acc, expected_acc)
    
def test_kinematics():
    # Create a mock joint with a specific joint_select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)
    
    # Test with non-zero joint_coord
    joint_angle = np.array([np.pi/4])
    p_link_frame = SE3.set_mat(np.eye(4)) # Identity matrix for simplicity
    expected_frame = p_link_frame @ (joint.origin @ SE3.set_mat(SE3.exp(joint.joint_select_mat @ joint_angle)))
    result_frame = kinematics(joint, p_link_frame, joint_angle)
    assert np.allclose(result_frame.mat(), expected_frame.mat())
    
def test_vel_kinematics():
    # Create a mock joint with a specific joint_select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)

    # Test with non-zero joint_coord and joint_veloc
    joint_angle = np.array([np.pi/4])
    joint_veloc = np.array([1.0])
    p_link_vel = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0]) # Zero velocity for simplicity
    expected_vel = (joint.origin.mat_inv_adj() @ (joint.joint_select_mat @ joint_veloc)) + p_link_vel
    result_vel = vel_kinematics(joint, p_link_vel, joint_angle, joint_veloc)
    assert np.allclose(result_vel, expected_vel)
    
def test_acc_kinematics():
    # Create a mock joint with a specific joint_select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)

    # Test with non-zero joint_coord, joint_veloc, and joint_accel
    joint_angle = np.array([np.pi/4])
    joint_veloc = np.array([1.5])
    joint_accel = np.array([2.0])
    p_link_vel = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0]) # Zero velocity for simplicity
    p_link_acc = np.array([0.0, 3.0, 0.0, 0.0, 0.0, 0.0]) # Zero acceleration for simplicity
    expected_acc = (joint.origin.mat_inv_adj() @ p_link_acc) + \
                   SE3.hat_adj(joint.origin @ SE3.set_mat(SE3.exp(joint.joint_select_mat @ joint_angle)) @ \
                   (joint.joint_select_mat @ joint_veloc)) @ p_link_vel + \
                   (joint.origin.mat_inv_adj() @ (joint.joint_select_mat @ joint_accel))
    result_acc = acc_kinematics(joint, p_link_vel, p_link_acc, joint_angle, joint_veloc, joint_accel)
    assert np.allclose(result_acc, expected_acc)
    
def test_part_link_jacob():
    # Create a mock joint with a specific joint_select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)
    
    # Test with a specific rel_frame
    rel_frame = SE3.set_mat(np.eye(4)) # Identity matrix for simplicity
    expected_jacob = rel_frame.mat_inv_adj() @ joint.origin.mat_inv_adj() @ joint.joint_select_mat
    result_jacob = part_link_jacob(joint, rel_frame)
    assert np.allclose(result_jacob, expected_jacob)

def test_rel_cmtm():
    # Create a mock joint with a specific joint_select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)

    # Test with non-zero joint_coord, joint_veloc, and joint_accel
    joint_angle = np.array([np.pi/4])
    joint_veloc = np.array([1.5])
    joint_accel = np.array([2.0])
    expected_frame = joint.origin @ SE3.set_mat(SE3.exp(joint.joint_select_mat @ joint_angle))
    expected_vel = joint.origin.mat_inv_adj() @ (joint.joint_select_mat @ joint_veloc)
    expected_acc = joint.origin.mat_inv_adj() @ (joint.joint_select_mat @ joint_accel)
    result_cmtm = link_rel_cmtm(joint, joint_angle, joint_veloc, joint_accel)
    assert np.allclose(result_cmtm.elem_mat(), expected_frame.mat())
    assert np.allclose(result_cmtm.elem_vecs(0), expected_vel)
    assert np.allclose(result_cmtm.elem_vecs(1), expected_acc)

def test_kinematics_cmtm():
    # Create a mock joint with a specific joint_select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)

    # Test with non-zero joint_coord, joint_veloc, and joint_accel
    joint_angle = np.array([np.pi/4])
    joint_veloc = np.array([1.5])
    joint_accel = np.array([2.0])
    p_link_frame = SE3()
    p_link_vel = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0]) 
    p_link_acc = np.array([0.0, 3.0, 0.0, 0.0, 0.0, 0.0]) 
    p_link_cmtm = CMTM[SE3](p_link_frame, np.array((p_link_vel, p_link_acc)))

    expected_frame = p_link_frame @ (joint.origin @ SE3.set_mat(SE3.exp(joint.joint_select_mat @ joint_angle)))
    expected_vel = (joint.origin.mat_inv_adj() @ (joint.joint_select_mat @ joint_veloc)) + p_link_vel
    expected_acc = (joint.origin.mat_inv_adj() @ p_link_acc) + \
                   SE3.hat_adj(joint.origin @ SE3.set_mat(SE3.exp(joint.joint_select_mat @ joint_angle)) @ \
                   (joint.joint_select_mat @ joint_veloc)) @ p_link_vel + \
                   (joint.origin.mat_inv_adj() @ (joint.joint_select_mat @ joint_accel))

    result_cmtm = kinematics_cmtm(joint, p_link_cmtm, joint_angle, joint_veloc, joint_accel)

    assert np.allclose(result_cmtm.elem_mat(), expected_frame.mat())
    assert np.allclose(result_cmtm.elem_vecs(0), expected_vel)
    assert np.allclose(result_cmtm.elem_vecs(1), expected_acc)