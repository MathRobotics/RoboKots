import numpy as np

import mathrobo as mr
from robokots.basic.kinematics import *

# Define a mock joint object with a select_mat attribute
class MockJoint:
    def __init__(self, select_mat):
        self.select_mat = select_mat
        self.origin = mr.SE3.rand()
        self.dof = select_mat.shape[1]
        self.select_indeces = np.argmax(self.select_mat, axis=0)

    def selector(self, mat: np.ndarray) -> np.ndarray:
        return mat[:, self.select_indeces]

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
    h0 = joint_local_frame(joint, joint_coord)
    joint_coord += joint_vel * delta
    h1 = joint_local_frame(joint, joint_coord)
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

def test_joint_local_cmtm():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)
    order = 5

    # Test with non-zero joint_coord, joint_veloc, and joint_accel
    joint_motions = np.random.rand(order)
    result_cmtm = joint_local_cmtm(joint, joint_motions, order)

    expected_frame = joint_local_frame(joint, joint_motions[:joint.dof].reshape(joint.dof))
    if order > 1:
        expected_vecs = np.zeros((order-1, 6))
        for i in range(order-1):
            expected_vecs[i] = joint_local_tan_vec(joint, joint_motions[(i+1)*joint.dof:(i+2)*joint.dof].reshape(joint.dof))

    expected_cmtm = CMTM[SE3](expected_frame, expected_vecs)

    assert np.allclose(result_cmtm.elem_mat(), expected_cmtm.elem_mat())

def test_joint_local_cmtm_numerical():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)
    order = 5

    joint_motions = np.random.rand(order)
    
    # Test with non-zero joint_coord, joint_veloc, and joint_accel
    result_cmtm = joint_local_cmtm(joint, joint_motions, order)

    # Calculate numerical Jacobian
    p0 = joint_local_frame(joint, joint_motions[:joint.dof].reshape(joint.dof))

    def func(x):
        return joint_local_cmtm(joint, x, order)
    
    def update_func(x_init, direct, eps):
        D, d = mr.build_integrator(joint.dof, order, eps, method="poly")

        x_ = D @ x_init + d @ direct
        return x_

    expected_cmtm = mr.numerical_difference(joint_motions, func, delta, sub_func = mr.CMTM.sub_vec, update_func=update_func, direction=joint_motions[order-1])

    for i in range(order-1):
        assert np.allclose(result_cmtm.elem_vecs(i), expected_cmtm[i*6:(i+1)*6])

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
    expected_vel = joint.select_mat @ joint_vel
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
    expected_acc = joint.select_mat @ joint_acc
    result_acc = link_rel_acc(joint, joint_acc)
    assert np.allclose(result_acc, expected_acc)

def test_link_rel_acc_numerical():
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

def test_link_rel_cmtm():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)
    order = 5

    # Test with non-zero joint_coord, joint_veloc, and joint_accel
    joint_motions = np.random.rand(order)
    result_cmtm = link_rel_cmtm(joint, joint_motions, order)

    expected_frame = link_rel_frame(joint, joint_motions[:joint.dof].reshape(joint.dof))
    if order > 1:
        expected_vecs = np.zeros((order-1, 6))
        for i in range(order-1):
            expected_vecs[i] = link_rel_tan_vec(joint, joint_motions[(i+1)*joint.dof:(i+2)*joint.dof].reshape(joint.dof))

    expected_cmtm = CMTM[SE3](expected_frame, expected_vecs)

    assert np.allclose(result_cmtm.elem_mat(), expected_cmtm.elem_mat())

def test_link_rel_cmtm_numerical():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)
    order = 5

    joint_motions = np.random.rand(order)
    
    # Test with non-zero joint_coord, joint_veloc, and joint_accel
    result_cmtm = link_rel_cmtm(joint, joint_motions, order)

    def func(x):
        return link_rel_cmtm(joint, x, order)

    def update_func(x_init, direct, eps):
        D, d = mr.build_integrator(joint.dof, order, eps, method="poly")

        x_ = D @ x_init + d @ direct
        return x_

    expected_cmtm = mr.numerical_difference(joint_motions, func, delta, sub_func = mr.CMTM.sub_vec, update_func=update_func, direction=joint_motions[order-1])

    for i in range(order-1):
        assert np.allclose(result_cmtm.elem_vecs(i), expected_cmtm[i*6:(i+1)*6])

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
    rel_frame = link_rel_frame(joint, joint_coord)
    rel_vel = link_rel_vel(joint, joint_veloc)
    expected_vel = rel_frame.mat_inv_adj() @ p_link_vel + rel_vel
    result_vel = vel_kinematics(joint, p_link_vel, joint_coord, joint_veloc)
    assert np.allclose(result_vel, expected_vel)

def test_vel_kinematics_numerical():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)
    
    # Test with non-zero joint_coord and joint_veloc
    joint_coord = np.random.rand(1)
    joint_veloc = np.random.rand(1)
    p_link_vel = np.random.rand(6)
    result_vel = vel_kinematics(joint, p_link_vel, joint_coord, joint_veloc)

    # Calculate numerical relative velocity
    p_link_frame = SE3.rand()
    rel_frame = link_rel_frame(joint, joint_coord)
    h0 = kinematics(joint, p_link_frame, joint_coord)
    joint_coord += joint_veloc * delta
    h1 = kinematics(joint, p_link_frame, joint_coord)
    expected_vel = rel_frame.mat_inv_adj() @ p_link_vel + mr.SE3.sub_tan_vec(h0, h1, "bframe") / delta
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
    rel_frame = link_rel_frame(joint, joint_coord)
    rel_vel = link_rel_vel(joint, joint_veloc)
    rel_acc = link_rel_acc(joint, joint_accel)
    expected_acc = rel_frame.mat_inv_adj() @ p_link_acc + \
                   SE3.hat_adj(rel_frame.mat_inv_adj() @ p_link_vel ) @ rel_vel + rel_acc
    result_acc = acc_kinematics(joint, p_link_vel, p_link_acc, joint_coord, joint_veloc, joint_accel)
    assert np.allclose(result_acc, expected_acc)

def test_acc_kinematics_numerical():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)
    
    # Test with non-zero joint_coord, joint_veloc, and joint_accel
    joint_coord = np.random.rand(1)
    joint_veloc = np.random.rand(1)
    joint_accel = np.random.rand(1)
    p_link_vel = np.random.rand(6)
    p_link_acc = np.random.rand(6)
    result_acc = acc_kinematics(joint, p_link_vel, p_link_acc, joint_coord, joint_veloc, joint_accel)

    # Calculate numerical acceleration
    rel_frame = link_rel_frame(joint, joint_coord)
    rel_veloc = link_rel_vel(joint, joint_veloc)

    v0 = vel_kinematics(joint, p_link_vel, joint_coord, joint_veloc)
    joint_veloc += joint_accel * delta
    v1 =  vel_kinematics(joint, p_link_vel, joint_coord, joint_veloc)
    expected_acc = rel_frame.mat_inv_adj() @ p_link_acc + \
                  SE3.hat_adj(rel_frame.mat_inv_adj() @ p_link_vel ) @ rel_veloc + \
                  (v1 - v0) / delta

    assert np.allclose(result_acc, expected_acc)  

def test_kinematics_cmtm():
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

def test_kinematics_cmtm_numerical():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)
    order = 6

    joint_motions = np.random.rand(order)
    p_link_cmtm = CMTM.rand(SE3, order)

    result_cmtm = kinematics_cmtm(joint, p_link_cmtm, joint_motions, order)

    rel_cmtm = link_rel_cmtm(joint, joint_motions, order)

    # Calculate numerical kinematics
    def func(x):
        return kinematics_cmtm(joint, p_link_cmtm, x, order)

    def update_func(x_init, direct, eps):
        D, d = mr.build_integrator(joint.dof, order, eps, method="poly")

        x_ = D @ x_init + d @ direct
        return x_

    diff = mr.numerical_difference(joint_motions, func, delta, sub_func = mr.CMTM.sub_tan_vec_var, update_func=update_func, direction=np.ones(1))

    link_cmtm = p_link_cmtm @ rel_cmtm
    base_vec = link_cmtm.tan_map_inv(order-1) @ rel_cmtm.mat_inv_adj(order-1) @ p_link_cmtm.tan_vecs_flatten()

    diff = link_cmtm.tan_map_inv() @ diff
    diff[:(order-1)*6] +=  base_vec
    
    result_vecs = result_cmtm.vecs()

    for i in range(order-1):
        assert np.allclose(result_cmtm.elem_vecs(i), diff[i*6:(i+1)*6])
    
def test_part_link_jacob():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)
    
    # Test with a specific rel_frame
    rel_frame = SE3.rand() # Identity matrix for simplicity
    expected_jacob = rel_frame.mat_inv_adj() @ joint.select_mat
    result_jacob = part_link_jacob(joint, rel_frame)
    assert np.allclose(result_jacob, expected_jacob)

def test_part_link_jacob_vec():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)

    joint_coord = np.random.rand(1)
    joint_veloc = np.random.rand(1)
    
    # Test with a specific rel_frame
    rel_frame = link_rel_frame(joint, joint_coord)
    expected_vel = rel_frame.mat_inv_adj() @ vel_kinematics(joint, np.zeros(6), np.zeros(1), joint_veloc)
    result_vel = part_link_jacob(joint, rel_frame) @ joint_veloc
    assert np.allclose(result_vel, expected_vel)  

def test_part_link_jacob_numerical():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)
    
    # Test with non-zero joint_coord and joint_veloc
    joint_coord = np.random.rand(1)
    rel_frame = SE3.rand()
    result_jacob = part_link_jacob(joint, rel_frame)

    # Calculate numerical Jacobian
    h0 = joint_local_frame(joint, joint_coord) @ rel_frame
    joint_coord += delta
    h1 = joint_local_frame(joint, joint_coord) @ rel_frame
    expected_jacob = mr.SE3.sub_tan_vec(h0, h1, "bframe") / delta
    
    assert np.allclose(result_jacob[:,0], expected_jacob)

def test_part_link_cmtm_jacob():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)
    order = 5

    # Test with a specific rel_frame
    rel_cmtm = CMTM.rand(SE3, order) # Identity matrix for simplicity
    joint_cmtm = joint_local_cmtm(joint, np.random.rand(order,1), order)
    expected_jacob = np.zeros((6 * order, joint.dof * order))
    tmp = rel_cmtm.mat_inv_adj() @ joint_cmtm.tan_map()

    for i in range(order):
        for j in range(i+1):
            expected_jacob[i*6:(i+1)*6, j*joint.dof:(j+1)*joint.dof] \
            = joint.selector(tmp[i*6:(i+1)*6, j*6:(j+1)*6])
    result_jacob = part_link_cmtm_tan_jacob(joint, rel_cmtm, joint_cmtm)

    assert np.allclose(result_jacob, expected_jacob)

def test_part_link_cmtm_jacob_vec():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)
    order = 5

    joint_motions = np.random.rand(order)
    joint_motions[0] = 0
    rel_motions = np.zeros((order))
    rel_motions[0] = np.random.rand()
    joint_dmotions = np.zeros((order)) 
    joint_dmotions[0:order-1] = joint_motions[1:]
    
    # Test with a specific rel_frame
    rel_cmtm = link_rel_cmtm(joint, rel_motions, order)
    joint_cmtm = joint_local_cmtm(joint, joint_motions, order)

    expected_cmtm = link_rel_cmtm(joint, joint_motions, order) @ rel_cmtm
    result_vecs = part_link_cmtm_jacob(joint, rel_cmtm, joint_cmtm, expected_cmtm) @ joint_dmotions

    for i in range(order-1):
        assert np.allclose(result_vecs[i*6:(i+1)*6].T, expected_cmtm.elem_vecs(i))

def test_part_link_cmtm_tan_jacob_numerical():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)

    order = 5

    joint_motions = np.random.rand(order)
    
    rel_frame = CMTM.rand(SE3, order)
    joint_frame = joint_local_cmtm(joint, joint_motions, order)
    result_jacob = part_link_cmtm_tan_jacob(joint, rel_frame, joint_frame)

    # Calculate numerical Jacobian
    def func(x):
        return joint_local_cmtm(joint, x, order) @ rel_frame
    numerical_jacob = mr.numerical_grad(joint_motions, func, delta, sub_func = mr.CMTM.sub_tan_vec_var)
    
    assert np.allclose(result_jacob, numerical_jacob)

def test_part_link_cmtm_jacob_numerical():
    # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)

    order = 4

    joint_motions = np.random.rand(order)
    
    rel_frame = CMTM.rand(SE3, order)
    joint_frame = joint_local_cmtm(joint, joint_motions, order)
    link_frame = joint_frame @ rel_frame
    result_jacob = part_link_cmtm_jacob(joint, rel_frame, joint_frame, link_frame)

    # Calculate numerical Jacobian
    def func(x):
        return joint_local_cmtm(joint, x, order) @ rel_frame

    numerical_jacob =  mr.numerical_grad(joint_motions, func, delta, sub_func = mr.CMTM.sub_tan_vec_var)
    numerical_jacob = link_frame.tan_map_inv() @ numerical_jacob
    
    assert np.allclose(result_jacob, numerical_jacob, atol=1e-6, rtol=1e-6)