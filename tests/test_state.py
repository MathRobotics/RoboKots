
import numpy as np

from robokots.state import RobotDF, RobotState

'''
Test data for RobotDF
'''
# Test data names for RobotDF
test_data_names = ["data1", "data2", "data3"]
# Test data for RobotDF
test_data = {
    "data1": [1., 2., 3.],
    "data2": [4., 5., 6.],
    "data3": [7., 8., 9.]
}
# Test RobotDF initialization
def test_robot_df_init():
    robot_df = RobotDF(test_data_names)
    # Check if the DataFrame is initialized correctly
    assert robot_df.df is not None
    # Check if the DataFrame has Correct column names
    assert robot_df.df.columns == test_data_names
    # Check if the DataFrame has the correct column names
    assert robot_df.df.shape[0] == 0

# Test adding a row to the RobotDF
def test_robot_df_add_column():
    robot_df = RobotDF(test_data_names)
    robot_df.add_row(test_data)
    print(robot_df.df)
    # Check if the DataFrame has the correct column names
    assert robot_df.df.columns == test_data_names
    # Check if the DataFrame has the correct number of rows after adding a row
    assert robot_df.df.shape[0] == 1
    # Check if the DataFrame has the correct data after adding a row
    assert np.array_equal(robot_df.df.row(0)[0],test_data["data1"])
    assert np.array_equal(robot_df.df.row(0)[1],test_data["data2"])
    assert np.array_equal(robot_df.df.row(0)[2],test_data["data3"])

'''
Test data for RobotState
'''
# Mock robot object with link and joint names
class MockRobot:
    link_names = ["link1", "link2"]
    joint_names = ["joint1", "joint2"]

test_link_aliases = ["pos", "rot"]
test_joint_aliases = ["vel", "acc"]

test_state_names = []
for l_name in MockRobot.link_names:
    for al in test_link_aliases:
        test_state_names.append(l_name + "_" + al)

for j_name in MockRobot.joint_names:
    for al in test_joint_aliases:
        test_state_names.append(j_name + "_" + al)

# Test data for RobotState
test_robot_data = {
    "link1_pos": [1., 2., 3.],
    "link1_rot": [4., 5., 6.],
    "link2_pos": [7., 8., 9.],
    "link2_rot": [10., 11., 12.],
    "joint1_vel": [13., 14., 15.],
    "joint1_acc": [16., 17., 18.],
    "joint2_vel": [19., 20., 21.],
    "joint2_acc": [22., 23., 24.]
}

# Test RobotState initialization
def test_robot_state_init():
    robot = MockRobot()
    state = RobotState(robot, test_link_aliases, test_joint_aliases)   
    # Check if the state_df is initialized correctly
    assert isinstance(state.state_df, RobotDF)
    assert state.state_df.df.shape[0] == 0
    assert state.state_df.df.columns == test_state_names

# Test RobotState DataFrame extraction
def test_robot_state_df():
    robot = MockRobot()
    state = RobotState(robot, test_link_aliases, test_joint_aliases)  
    assert state.df().shape[0] == 0
    assert state.df().columns == test_state_names

# Test link state vec extraction
def test_robot_state_link_state_vec():
    robot = MockRobot()
    state = RobotState(robot, test_link_aliases, test_joint_aliases)  
    state.state_df.add_row(test_robot_data)
    # Test link state vector extraction
    link_name = "link1"
    type = "pos"
    vec = state.link_state_vec(state.df(), link_name, type)
    assert np.array_equal(vec, test_robot_data[link_name + "_" + type])
    # Test joint state vector extraction
    joint_name = "joint2"
    type = "vel"
    vec = state.link_state_vec(state.df(), joint_name, type)
    assert np.array_equal(vec, test_robot_data[joint_name + "_" + type])