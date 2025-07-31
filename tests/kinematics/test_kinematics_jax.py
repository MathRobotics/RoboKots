import numpy as np

from mathrobo import SE3, CMTM
from robokots.kinematics.kinematics import joint_rel_cmtm, part_link_cmtm_jacob
from robokots.kinematics.kinematics_jax import kinematics_acc, kinematics_jerk
from jax import jacrev
import jax.numpy as jnp
import jax

# Define a mock joint object with a select_mat attribute
class MockJoint:
    def __init__(self, select_mat):
        self.select_mat = select_mat
        self.origin = SE3.rand()
        self.dof = select_mat.shape[1]
        self.select_indeces = np.argmax(self.select_mat, axis=0)

    def selector(self, mat: np.ndarray) -> np.ndarray:
        return mat[:, self.select_indeces]
    
def test_joint_jerk_local_jacobian():
    # Create a mock joint with a specific select_mat
    select_mat = np.array([[1, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]])
    joint = MockJoint(select_mat)

    def func(x):
        v0 = jnp.zeros(6)
        acc = kinematics_acc(joint, v0, v0, v0, 
                               x[joint.dof*0:joint.dof*1],
                               x[joint.dof*1:joint.dof*2],
                               x[joint.dof*2:joint.dof*3])
        return acc

    motion = jax.random.uniform(jax.random.PRNGKey(0), (joint.dof * 3))
    motion_np = np.array(motion)
    joint_cmtm = joint_rel_cmtm(joint, motion_np, order=4)
    rel_cmtm = CMTM.eye(SE3,output_order=4)
    jacob_ana = part_link_cmtm_jacob(joint, rel_cmtm, joint_cmtm, joint_cmtm)

    jacob_auto = jacrev(func)(motion)


    assert np.allclose(jacob_ana, jacob_auto)


def test_joint_snap_local_jacobian():
    # Create a mock joint with a specific select_mat
    select_mat = np.array([[1, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]])
    joint = MockJoint(select_mat)

    def func(x):
        v0 = np.zeros(6)
        jerk = kinematics_jerk(joint, v0, v0, v0, v0, 
                               x[joint.dof*0:joint.dof*1],
                               x[joint.dof*1:joint.dof*2],
                               x[joint.dof*2:joint.dof*3],
                               x[joint.dof*3:joint.dof*4])
        return jerk
    
