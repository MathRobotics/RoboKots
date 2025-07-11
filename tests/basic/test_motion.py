import numpy as np

from robokots.basic.motion import RobotMotions
        
test_robot_dof = 3
test_dof = 1
test_dof_index = 1

# Test the RobotMotions class initialization with default values
def test_robot_motions_init_default():
    motions = RobotMotions(test_robot_dof)
    assert motions.dof == 3
    assert motions.motion_num == 3
    assert np.array_equal(motions.motions, np.zeros(9))
    assert motions.aliases == ["coord", "veloc", "accel"]

# Test the set_aliases method  
def test_robot_motions_init_custom_aliases():
    motions = RobotMotions(test_robot_dof, ["coord", "veloc"])
    assert motions.dof == 3
    assert motions.motion_num == 2
    assert np.array_equal(motions.motions, np.zeros(6))
    assert motions.aliases == ["coord", "veloc"]

def test_robot_motions_init_custom_aliases_with_accel_diff():
    motions = RobotMotions(test_robot_dof, ["coord", "veloc", "accel", "accel_diff1", "accel_diff2"])
    assert motions.dof == 3
    assert motions.motion_num == 5
    assert np.array_equal(motions.motions, np.zeros(15))
    assert motions.aliases == ["coord", "veloc", "accel", "accel_diff1", "accel_diff2"]

# Test invalid alias handling
def test_robot_motions_init_invalid_aliases():
    try:
        _ = RobotMotions(test_robot_dof, ["coord", "invalid"])
    except ValueError as e:
        assert str(e) == "Invalid alias: {'invalid'}"
    else:
        assert False, "Expected ValueError not raised"
        
# Test the set_aliases method
def test_set_aliases():
    motions = RobotMotions(test_robot_dof)
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
    motions = RobotMotions(test_robot_dof)
    motions.set_motion(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    assert np.array_equal(motions.motions, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    
# Test the motion_index method
def test_motion_index():
    motions = RobotMotions(test_robot_dof)
    assert motions.motion_index("coord") == 0
    assert motions.motion_index("veloc") == 1
    assert motions.motion_index("accel") == 2
    try:
        motions.motion_index("invalid")
    except ValueError as e:
        assert str(e) == "Invalid alias: invalid"
    
# Test the gen_values method
def test_gen_values():
    motions = RobotMotions(test_robot_dof)
    motions.set_motion(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    assert np.array_equal(motions.gen_values("coord"), np.array([1, 2, 3]))
    assert np.array_equal(motions.gen_values("veloc"), np.array([4, 5, 6]))
    assert np.array_equal(motions.gen_values("accel"), np.array([7, 8, 9]))
  
# Test the coord, veloc, and accel methods
def test_coord_veloc_accel():
    motions = RobotMotions(test_robot_dof)
    motions.set_motion(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    assert np.array_equal(motions.coord(), np.array([1, 2, 3]))
    assert np.array_equal(motions.veloc(), np.array([4, 5, 6]))
    assert np.array_equal(motions.accel(), np.array([7, 8, 9]))
    
# Test the gen_value method
def test_gen_value():
    motions = RobotMotions(test_robot_dof)
    motions.set_motion(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    assert np.array_equal(motions.gen_value(test_dof, test_dof_index, "coord"), np.array([2]))
    assert np.array_equal(motions.gen_value(test_dof, test_dof_index, "veloc"), np.array([5]))
    assert np.array_equal(motions.gen_value(test_dof, test_dof_index, "accel"), np.array([8]))

# Test the joint_coord, joint_veloc, and joint_accel methods
def test_joint_coord_veloc_accel():
    motions = RobotMotions(test_robot_dof)
    motions.set_motion(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    assert np.array_equal(motions.joint_coord(test_dof, test_dof_index), np.array([2]))
    assert np.array_equal(motions.joint_veloc(test_dof, test_dof_index), np.array([5]))
    assert np.array_equal(motions.joint_accel(test_dof, test_dof_index), np.array([8]))
    
# Test the link_coord, link_veloc, and link_accel methods
def test_link_coord_veloc_accel():
    motions = RobotMotions(test_robot_dof)
    motions.set_motion(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    assert np.array_equal(motions.link_coord(test_dof, test_dof_index), np.array([2]))
    assert np.array_equal(motions.link_veloc(test_dof, test_dof_index), np.array([5]))
    assert np.array_equal(motions.link_accel(test_dof, test_dof_index), np.array([8]))
