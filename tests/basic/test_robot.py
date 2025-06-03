import numpy as np
import pytest

from mathrobo import SE3
from robokots.basic.robot import RobotStruct, LinkStruct, JointStruct


def test_robot_struct_init():
    # Create mock links and joints
    link1 = LinkStruct(0, "link1", np.zeros(3), 1.0, np.eye(6), "rigid")
    link2 = LinkStruct(1, "link2", np.zeros(3), 1.0, np.eye(6), "rigid")
    joint1 = JointStruct(0, "joint1", "revolute", np.array((1,0,0)), 0, 1, SE3())
    
    # Initialize RobotStruct with mock links and joints
    robot = RobotStruct([link1,link2], [joint1])
    
    # Check if the robot is initialized correctly
    assert robot.link_num == 2
    assert robot.joint_num == 1
    assert robot.dof == 1
    assert robot.joint_dof == 1
    assert robot.link_dof == 0
    assert robot.link_names == ["link1","link2"]
    assert robot.joint_names == ["joint1"]

def test_joint_struct_init():
    origin = SE3.rand()
    # Create a mock joint
    joint = JointStruct(0, "joint1", "revolute", np.array((0,0,1)), 0, 1, origin)
    
    # Check if the joint is initialized correctly
    assert joint.name == "joint1"
    assert joint.type == "revolute"
    assert np.array_equal(joint.axis, np.array((0,0,1)))
    assert joint.parent_link_id == 0
    assert joint.child_link_id == 1
    assert joint.dof == 1
    assert joint.dof_index == 0
    assert np.allclose(joint.select_mat, np.array([[0], [0], [1], [0], [0], [0]]))
    assert joint.select_indeces == [2]
    assert isinstance(joint.origin, SE3)
    assert np.allclose(joint.origin.mat(), origin.mat())

def test_joint_set_dof_index():
    # Create a mock joint
    joint = JointStruct(0, "joint1", "revolute", np.array((1,0,0)), 0, 1, SE3())
    
    # Set a valid DOF index
    joint.set_dof_index(2)      
    assert joint.dof_index == 2
    
    # Test setting an invalid DOF index
    try:
        joint.set_dof_index(-1)
    except ValueError as e:
        assert str(e) == "Invalid DOF index: -1"

def test_selector():
    # Create a mock joint
    joint = JointStruct(0, "joint1", "revolute", np.array((0,1,0)), 0, 1, SE3())
    
    # Create a mock matrix
    mat = np.random.rand(6, 6)
    
    # Apply the selector method
    selected_mat = joint.selector(mat)
    
    # Check if the selected matrix is correct
    assert np.array_equal(selected_mat, mat[:,[1]])  # Only the first elements should be selected
 
def test_scatter():
    # Create a mock joint
    joint = JointStruct(0, "joint1", "revolute", np.array((0,1,0)), 0, 1, SE3())
    
    # Create a mock matrix
    mat = np.array([[2]])
    
    # Apply the scatter method
    scattered_mat = joint.scatter(mat)
    
    # Check if the scattered matrix is correct
    assert np.array_equal(scattered_mat, np.array([[0], [2], [0], [0], [0], [0]]))  # Only the first elements should be scattered