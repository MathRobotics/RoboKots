import json
from pathlib import Path
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
        {"data_type": "pos", "owner_type": "link", "owner_name": "link1", "frame_name": "world", "frame": [0.0, 0.0, 0.0]},
        {"data_type": "pos", "owner_type": "link", "owner_name": "link2", "frame_name": "world", "frame": [1.0, 1.0, 1.0]},
    ]
}

TEST_DIR = Path(__file__).resolve().parent
FILE_DIR = TEST_DIR / "file"
TEST_ROBOT_PATH = FILE_DIR / "test_robot.json"
INVALID_PATH = FILE_DIR / "invalid.json"
OUTPUT_PATH = FILE_DIR / "output.json"

def test_load_json_file():
    FILE_DIR.mkdir(parents=True, exist_ok=True)
    # Test loading a valid JSON file
    with open(TEST_ROBOT_PATH, "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    loaded_data = load_json_file(str(TEST_ROBOT_PATH))
    assert loaded_data == test_data
    # Test loading a non-existent file
    try:
        load_json_file("non_existent.json")
    except FileNotFoundError as e:
        assert str(e) == "File non_existent.json not found."
    # Test loading an invalid JSON file
    with open(INVALID_PATH, "w", encoding="utf-8") as f:
        f.write("{invalid_json}")
    try:
        load_json_file(str(INVALID_PATH))
    except ValueError as e:
        assert str(e) == "Invalid JSON format: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"

def test_save_json_file():
    FILE_DIR.mkdir(parents=True, exist_ok=True)
    # Test saving to a valid JSON file
    save_json_file(test_data, str(OUTPUT_PATH))
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    assert saved_data == test_data
    # Test saving to a read-only file
    try:
        save_json_file(test_data, str(OUTPUT_PATH))
    except IOError as e:
        assert str(e) == "Failed to write JSON file: [Errno 13] Permission denied: '/file/output.json'"
