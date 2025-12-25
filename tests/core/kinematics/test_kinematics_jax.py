import numpy as np

from mathrobo import SE3, CMTM, numerical_grad
from robokots.core.kinematics.kinematics import joint_rel_cmtm, part_link_cmtm_jacob
from robokots.core.kinematics.kinematics_jax import kinematics, kinematics_vel, kinematics_acc, kinematics_jerk
from jax import jacrev
import jax.numpy as jnp
import jax

import pytest

# Define a mock joint object with a select_mat attribute
class MockJoint:
    def __init__(self, select_mat, lib="jax"):
        if lib == "jax":
            self.select_mat = jnp.array(select_mat)
            self.origin = SE3.rand(LIB="jax")
            self.dof = select_mat.shape[1]
            self.select_indeces = jnp.argmax(self.select_mat, axis=0)
        else:
            self.select_mat = select_mat
            self.origin = SE3.rand()
            self.dof = select_mat.shape[1]
            self.select_indeces = np.argmax(self.select_mat, axis=0)

    def selector(self, mat: np.ndarray) -> np.ndarray:
        return mat[:, self.select_indeces]
    
@pytest.mark.slow 
def test_joint_vel_local_jacobian():
     # Create a mock joint with a specific select_mat
    joint = MockJoint(np.array([[0, 1, 0, 0, 0, 0]]).T)

    p = 1
    def func(x):
        f0 = SE3.eye(LIB="jax")
        frame = kinematics(joint, f0, x)
        return frame.mat()

    def func2(x):
        f0 = SE3.eye()
        frame = f0 @ SE3.set_mat(SE3.exp(joint.select_mat @ x[:joint.dof]))
        return frame

    motion = jax.random.uniform(jax.random.PRNGKey(0), (joint.dof * p))
    motion_np = np.array(motion)
    joint_cmtm = joint_rel_cmtm(joint, motion_np, order=p)
    rel_cmtm = CMTM.eye(SE3,output_order=p)
    jacob_ana = part_link_cmtm_jacob(joint, rel_cmtm, joint_cmtm, joint_cmtm)
    jac = jacrev(func)
    h = func(motion)
    j = (jac(motion).T @ jnp.linalg.inv(h).T).T
    jacob_auto = jnp.zeros((6,joint.dof))
    for i in range(joint.dof):
        tmp = jnp.zeros((4,4))
        tmp = tmp.at[0, :].set(j[0,:,i])
        tmp = tmp.at[1, :].set(j[1,:,i])
        tmp = tmp.at[2, :].set(j[2,:,i])
        tmp = tmp.at[3, :].set(j[3,:,i])
        jacob_auto = jacob_auto.at[:, i].set(SE3.vee(tmp))

    jacob_num = numerical_grad(motion_np, func2, eps=1e-3, sub_func=SE3.sub_tan_vec) # eps = 1e-3, bucause of numerical error

    assert np.allclose(jacob_ana[(p-1)*6:], jacob_auto, atol=1e-6, rtol=1e-6)
    assert np.allclose(jacob_ana[(p-1)*6:], jacob_num, atol=1e-3, rtol=1e-3) # eps = 1e-3, bucause of numerical error

@pytest.mark.slow 
def test_joint_acc_local_jacobian():
    # Create a mock joint with a specific select_mat
    select_mat = np.array([[1, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]])
    joint = MockJoint(select_mat)

    p = 2
    def func(x):
        v0 = jnp.zeros(6)
        vel = kinematics_vel(joint, v0, 
                               x[joint.dof*0:joint.dof*1],
                               x[joint.dof*1:joint.dof*2])
        return vel

    motion = jax.random.uniform(jax.random.PRNGKey(0), (joint.dof * p))
    motion_np = np.array(motion)
    joint_cmtm = joint_rel_cmtm(joint, motion_np, order=p)
    rel_cmtm = CMTM.eye(SE3,output_order=p)
    jacob_ana = part_link_cmtm_jacob(joint, rel_cmtm, joint_cmtm, joint_cmtm)
    jacob_auto = jacrev(func)(motion)
    jacob_num = numerical_grad(motion, func, eps=1e-3) # eps = 1e-3, bucause of numerical error

    assert np.allclose(jacob_ana[(p-1)*6:], jacob_auto, atol=1e-6, rtol=1e-6)
    assert np.allclose(jacob_ana[(p-1)*6:], jacob_num, atol=1e-3, rtol=1e-3) # eps = 1e-3, bucause of numerical error

@pytest.mark.slow 
def test_joint_jerk_local_jacobian():
    # Create a mock joint with a specific select_mat
    select_mat = np.array([[1, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]])
    joint = MockJoint(select_mat)

    def func(x):
        v0 = jnp.zeros(6)
        acc = kinematics_acc(joint, v0, v0, 
                               x[joint.dof*0:joint.dof*1],
                               x[joint.dof*1:joint.dof*2],
                               x[joint.dof*2:joint.dof*3])
        return acc

    motion = jax.random.uniform(jax.random.PRNGKey(0), (joint.dof * 3))
    motion_np = np.array(motion)
    joint_cmtm = joint_rel_cmtm(joint, motion_np, order=3)
    rel_cmtm = CMTM.eye(SE3,output_order=3)
    jacob_ana = part_link_cmtm_jacob(joint, rel_cmtm, joint_cmtm, joint_cmtm)
    jacob_auto = jacrev(func)(motion)
    jacob_num = numerical_grad(motion, func, eps=1e-3)

    assert np.allclose(jacob_ana[12:], jacob_auto)
    for i in range(6):
        print(f"Joint {i} Acceleration Jacobian: \n{jacob_ana[12+i:13+i]}")
        print(f"Joint {i} Acceleration Auto Jacobian:\n{jacob_auto[i:i+1]}")
        print(f"Joint {i} Acceleration Numerical Jacobian:\n{jacob_num[i:i+1]}")
    assert np.allclose(jacob_ana[12:], jacob_num, atol=1e-3, rtol=1e-3)

@pytest.mark.slow 
def test_joint_snap_local_jacobian():
    # Create a mock joint with a specific select_mat
    select_mat = np.array([[1, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]])
    joint = MockJoint(select_mat)

    def func(x):
        v0 = jnp.zeros(6)
        jerk = kinematics_jerk(joint, v0, v0, v0, 
                               x[joint.dof*0:joint.dof*1],
                               x[joint.dof*1:joint.dof*2],
                               x[joint.dof*2:joint.dof*3],
                               x[joint.dof*3:joint.dof*4])
        return jerk
    
    motion = jax.random.uniform(jax.random.PRNGKey(0), (joint.dof * 4))
    motion_np = np.array(motion)
    joint_cmtm = joint_rel_cmtm(joint, motion_np, order=4)
    rel_cmtm = CMTM.eye(SE3, output_order=4)
    jacob_ana = part_link_cmtm_jacob(joint, rel_cmtm, joint_cmtm, joint_cmtm)
    jacob_auto = jacrev(func)(motion)
    jacob_num = numerical_grad(motion, func, eps=1e-3)
    
    assert np.allclose(jacob_ana[18:], jacob_auto)
    assert np.allclose(jacob_ana[18:], jacob_num, atol=1e-3, rtol=1e-3)

@pytest.mark.slow 
def test_joint_jerk_part_jacobian():
    # Create a mock joint with a specific select_mat
    select_mat = np.array([[1, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]])
    joint = MockJoint(select_mat)

    motion0 = jax.random.uniform(jax.random.PRNGKey(0), (joint.dof * 3))

    def func(x):
        v0 = jnp.zeros(6)
        vel = kinematics_vel(joint, v0, 
                               x[joint.dof*0:joint.dof*1],
                               x[joint.dof*1:joint.dof*2])
        acc = kinematics_acc(joint, v0, v0, 
                               x[joint.dof*0:joint.dof*1],
                               x[joint.dof*1:joint.dof*2],
                               x[joint.dof*2:joint.dof*3])
        v = kinematics_acc(joint, vel, acc,
                               motion0[joint.dof*0:joint.dof*1],
                               motion0[joint.dof*1:joint.dof*2],
                               motion0[joint.dof*2:joint.dof*3])
        return v

    motion = jax.random.uniform(jax.random.PRNGKey(0), (joint.dof * 3))
    motion_np = np.array(motion)
    joint_cmtm = joint_rel_cmtm(joint, motion_np, order=3)
    rel_cmtm = joint_rel_cmtm(joint, motion0, order=3)
    link_cmtm = joint_cmtm @ rel_cmtm
    jacob_ana = part_link_cmtm_jacob(joint, rel_cmtm, joint_cmtm, link_cmtm)
    jacob_auto = jacrev(func)(motion)
    jacob_num = numerical_grad(motion, func, eps=1e-3)  # eps = 1e-3, because of numerical error

    for i in range(6):
        print(f"Joint {i} Jerk Jacobian: \n{jacob_ana[12+i:13+i]}")
        print(f"Joint {i} Jerk Auto Jacobian:\n{jacob_auto[i:i+1]}")
        print(f"Joint {i} Jerk Numerical Jacobian:\n{jacob_num[i:i+1]}")

    assert np.allclose(jacob_ana[12:], jacob_auto, atol=1e-6, rtol=1e-6)
    assert np.allclose(jacob_ana[12:], jacob_num, atol=1e-3, rtol=1e-3)

@pytest.mark.slow 
def test_joint_snap_part_jacobian():
    # Create a mock joint with a specific select_mat
    select_mat = np.array([[1, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]])
    joint = MockJoint(select_mat)

    p = 4

    motion0 = jax.random.uniform(jax.random.PRNGKey(0), (joint.dof * p))

    def func(x):
        v0 = jnp.zeros(6)
        vel = kinematics_vel(joint, v0, 
                               x[joint.dof*0:joint.dof*1],
                               x[joint.dof*1:joint.dof*2])
        acc = kinematics_acc(joint, v0, v0, 
                               x[joint.dof*0:joint.dof*1],
                               x[joint.dof*1:joint.dof*2],
                               x[joint.dof*2:joint.dof*3])
        jerk = kinematics_jerk(joint, v0, v0, v0,
                               x[joint.dof*0:joint.dof*1],
                               x[joint.dof*1:joint.dof*2],
                               x[joint.dof*2:joint.dof*3],
                               x[joint.dof*3:joint.dof*4])
        v = kinematics_jerk(joint, vel, acc, jerk,
                               motion0[joint.dof*0:joint.dof*1],
                               motion0[joint.dof*1:joint.dof*2],
                               motion0[joint.dof*2:joint.dof*3],
                               motion0[joint.dof*3:joint.dof*4])
        return v

    motion = jax.random.uniform(jax.random.PRNGKey(0), (joint.dof * p))
    motion_np = np.array(motion)
    joint_cmtm = joint_rel_cmtm(joint, motion_np, order=p)
    rel_cmtm = joint_rel_cmtm(joint, motion0, order=p)
    link_cmtm = joint_cmtm @ rel_cmtm
    jacob_ana = part_link_cmtm_jacob(joint, rel_cmtm, joint_cmtm, link_cmtm)
    jacob_auto = jacrev(func)(motion)
    jacob_num = numerical_grad(motion, func, eps=1e-3)  # eps = 1e-3, because of numerical error

    for i in range(6):
        print(f"Joint {i} Jerk Jacobian: \n{jacob_ana[18+i:19+i]}")
        print(f"Joint {i} Jerk Auto Jacobian:\n{jacob_auto[i:i+1]}")
        print(f"Joint {i} Jerk Numerical Jacobian:\n{jacob_num[i:i+1]}")

    assert np.allclose(jacob_ana[18:], jacob_auto, atol=1e-6, rtol=1e-6)
    assert np.allclose(jacob_ana[18:], jacob_num, atol=1e-3, rtol=1e-3)
