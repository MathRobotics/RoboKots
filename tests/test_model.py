import numpy as np
import pytest

from mathrobo import SE3
from robokots.robot_model import RobotStruct, LinkStruct, JointStruct


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