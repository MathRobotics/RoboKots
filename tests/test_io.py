from robokots.robot_io import *

test_data = {
    "links": [
        {"id":0,"name": "link1", "mass": 1.0, "cog": [0.0, 0.0, 0.0], "inertia": [1.0, 1.0, 1.0]},
        {"id":1,"name": "link2", "mass": 2.0, "cog": [1.0, 1.0, 1.0], "inertia": [2.0, 2.0, 2.0]},
    ],
    "joints": [
        {"id":0,"name": "joint1", "type": "revolute", "parent_link_id": 0, "child_link_id": 1, "axis": [0, 0, 1]},
        {"id":1,"name": "joint2", "type": "revolute", "parent_link_id": 1, "child_link_id": 0, "axis": [1, 0, 0]},
    ]
}

# Test loading a valid target JSON file
target_data = {
    "targets": [
        {"type": "pos", "link": "link1", "pos": [0.0, 0.0, 0.0]},
        {"type": "pos", "link": "link2", "pos": [1.0, 1.0, 1.0]},
    ]
}

def test_io_load_json():
    # Test loading a valid JSON file
    with open("./file/test_robot.json", "w") as f:
        json.dump(test_data, f)
    loaded_data = io_load_json("./file/test_robot.json")
    assert loaded_data == test_data
    # Test loading a non-existent file
    try:
        io_load_json("non_existent.json")
    except FileNotFoundError as e:
        assert str(e) == "File non_existent.json not found."
    # Test loading an invalid JSON file
    with open("./file/invalid.json", "w") as f:
        f.write("{invalid_json}")
    try:
        io_load_json("./file/invalid.json")
    except ValueError as e:
        assert str(e) == "Invalid JSON format: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"
    
def test_io_save_json():
    # Test saving to a valid JSON file
    io_save_json(test_data, "./file/output.json")
    with open("./file/output.json", "r") as f:
        saved_data = json.load(f)
    assert saved_data == test_data
    # Test saving to a read-only file
    try:
        io_save_json(test_data, "./file/output.json")
    except IOError as e:
        assert str(e) == "Failed to write JSON file: [Errno 13] Permission denied: '/file/output.json'"
        
def test_io_from_json_file():
    # Test loading a valid JSON file
    with open("./file/test_robot.json", "w") as f:
        json.dump(test_data, f)
    robot = io_from_json_file("./file/test_robot.json")
    assert isinstance(robot, RobotStruct)
    assert robot.links[0].name == "link1"
    assert robot.joints[1].type == "revolute"
    
    # Test loading a non-existent file
    try:
        io_from_json_file("non_existent.json")
    except FileNotFoundError as e:
        assert str(e) == "File non_existent.json not found."
        
    # Test loading an invalid JSON file
    with open("./file/invalid.json", "w") as f:
        f.write("{invalid_json}")
    try:
        io_from_json_file("./file/invalid.json")
    except ValueError as e:
        assert str(e) == "Invalid JSON format: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"
        
def test_io_print_structure():
    # Test printing the structure of a robot
    robot = RobotStruct.from_dict(test_data)
    io_print_structure(robot)
    # Check the printed output manually or redirect stdout to capture it
    # This is a placeholder as capturing printed output requires more setup
    # assert printed_output == expected_output
    
def test_io_from_target_json():
    with open("./file/test_target.json", "w") as f:
        json.dump(target_data, f)
    target_list = io_from_target_json("./file/test_target.json")
    assert isinstance(target_list, TargetList)
    assert target_list.targets[0].type == "pos"
    assert target_list.targets[1].link_name == "link2"
    
    # Test loading a non-existent target JSON file
    try:
        io_from_target_json("non_existent.json")
    except FileNotFoundError as e:
        assert str(e) == "File non_existent.json not found."
        
    # Test loading an invalid target JSON file
    with open("./file/invalid.json", "w") as f:
        f.write("{invalid_json}")
    try:
        io_from_target_json("./file/invalid.json")
    except ValueError as e:
        assert str(e) == "Invalid JSON format: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"
        
def test_io_print_targets():
    # Test printing the targets
    target_list = TargetList.from_dict(target_data)
    io_print_targets(target_list)
    # Check the printed output manually or redirect stdout to capture it
    # This is a placeholder as capturing printed output requires more setup
    # assert printed_output == expected_output
    
