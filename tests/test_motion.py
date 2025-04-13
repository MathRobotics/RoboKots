import numpy as np

from robokots.motion import RobotMotions

'''
Test the RobotMotions class
'''

class MockRobot:
    def __init__(self, dof):
        self.dof = dof
        
class MockJoint:
    def __init__(self, dof_index, dof):
        self.dof_index = dof_index
        self.dof = dof

# Test the RobotMotions class initialization with default values
def test_robot_motions_init_default():
    robot = MockRobot(3)
    motions = RobotMotions(robot)
    assert motions.dof == 3
    assert motions.motion_num == 3
    assert np.array_equal(motions.motions, np.zeros(9))
    assert motions.aliases == ["coord", "veloc", "accel"]

# Test the set_aliases method  
def test_robot_motions_init_custom_aliases():
    robot = MockRobot(3)
    motions = RobotMotions(robot, ["coord", "veloc"])
    assert motions.dof == 3
    assert motions.motion_num == 2
    assert np.array_equal(motions.motions, np.zeros(6))
    assert motions.aliases == ["coord", "veloc"]

# Test invalid alias handling
def test_robot_motions_init_invalid_aliases():
    robot = MockRobot(3)
    try:
        _ = RobotMotions(robot, ["coord", "invalid"])
    except ValueError as e:
        assert str(e) == "Invalid alias: {'invalid'}"
    else:
        assert False, "Expected ValueError not raised"
        
# Test the set_aliases method
def test_set_aliases():
    robot = MockRobot(3)
    motions = RobotMotions(robot)
    motions.set_aliases(["coord", "veloc"])
    assert motions.aliases == ["coord", "veloc"]
    
    try:
        motions.set_aliases(["coord", "invalid"])
    except ValueError as e:
        assert str(e) == "Invalid alias: {'invalid'}"
    else:
        assert False, "Expected ValueError not raised"
        
# Test the set_motion method
def test_set_motion():
    robot = MockRobot(3)
    motions = RobotMotions(robot)
    motions.set_motion(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    assert np.array_equal(motions.motions, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    
# Test the motion_index method
def test_motion_index():
    robot = MockRobot(3)
    motions = RobotMotions(robot)
    assert motions.motion_index("coord") == 0
    assert motions.motion_index("veloc") == 1
    assert motions.motion_index("accel") == 2
    try:
        motions.motion_index("invalid")
    except ValueError as e:
        assert str(e) == "Invalid alias: invalid"
    
# Test the gen_values method
def test_gen_values():
    robot = MockRobot(3)
    motions = RobotMotions(robot)
    motions.set_motion(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    assert np.array_equal(motions.gen_values("coord"), np.array([1, 2, 3]))
    assert np.array_equal(motions.gen_values("veloc"), np.array([4, 5, 6]))
    assert np.array_equal(motions.gen_values("accel"), np.array([7, 8, 9]))
  
# Test the coord, veloc, and accel methods
def test_coord_veloc_accel():
    robot = MockRobot(3)
    motions = RobotMotions(robot)
    motions.set_motion(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    assert np.array_equal(motions.coord(), np.array([1, 2, 3]))
    assert np.array_equal(motions.veloc(), np.array([4, 5, 6]))
    assert np.array_equal(motions.accel(), np.array([7, 8, 9]))
    
# Test the gen_value method
def test_gen_value():
    robot = MockRobot(3)
    motions = RobotMotions(robot)
    motions.set_motion(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    joint = MockJoint(1, 1)
    assert np.array_equal(motions.gen_value(joint, "coord"), np.array([2]))
    assert np.array_equal(motions.gen_value(joint, "veloc"), np.array([5]))
    assert np.array_equal(motions.gen_value(joint, "accel"), np.array([8]))

# Test the joint_coord, joint_veloc, and joint_accel methods
def test_joint_coord_veloc_accel():
    robot = MockRobot(3)
    motions = RobotMotions(robot)
    motions.set_motion(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    joint = MockJoint(1, 1)
    assert np.array_equal(motions.joint_coord(joint), np.array([2]))
    assert np.array_equal(motions.joint_veloc(joint), np.array([5]))
    assert np.array_equal(motions.joint_accel(joint), np.array([8]))
    
# Test the link_coord, link_veloc, and link_accel methods
def test_link_coord_veloc_accel():
    robot = MockRobot(3)
    motions = RobotMotions(robot)
    motions.set_motion(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    link = MockJoint(1, 1)
    assert np.array_equal(motions.link_coord(link), np.array([2]))
    assert np.array_equal(motions.link_veloc(link), np.array([5]))
    assert np.array_equal(motions.link_accel(link), np.array([8]))
