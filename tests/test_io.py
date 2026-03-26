import json
import math
from pathlib import Path
import pytest
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

TEST_URDF = """<?xml version="1.0"?>
<robot name="two_link">
  <link name="base">
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="2.5"/>
      <inertia ixx="1.0" ixy="0.1" ixz="0.2" iyy="2.0" iyz="0.3" izz="3.0"/>
    </inertial>
  </link>
  <link name="tool"/>
  <joint name="joint1" type="continuous">
    <parent link="base"/>
    <child link="tool"/>
    <origin xyz="1 0 0" rpy="0 0 1.5707963267948966"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
"""

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


def test_load_urdf_file(tmp_path: Path):
    urdf_path = tmp_path / "test_robot.urdf"
    with open(urdf_path, "w", encoding="utf-8") as f:
        f.write(TEST_URDF)

    model = load_urdf_file(str(urdf_path))

    assert len(model["links"]) == 3
    assert model["links"][0]["name"] == "world"

    base = next(link for link in model["links"] if link["name"] == "base")
    assert base["mass"] == 2.5
    assert base["inertia"] == [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]

    joint1 = next(j for j in model["joints"] if j["name"] == "joint1")
    assert joint1["type"] == "revolute"
    assert joint1["axis"] == [0.0, 0.0, 1.0]
    assert joint1["origin"]["position"] == [1.0, 0.0, 0.0]
    quat = joint1["origin"]["orientation"]
    assert quat[0] == pytest.approx(math.sqrt(0.5), abs=1e-12)
    assert quat[1] == pytest.approx(0.0, abs=1e-12)
    assert quat[2] == pytest.approx(0.0, abs=1e-12)
    assert quat[3] == pytest.approx(math.sqrt(0.5), abs=1e-12)

    world_joint = next(j for j in model["joints"] if j["type"] == "fix" and j["parent_link_id"] == 0)
    assert world_joint["child_link_id"] == base["id"]


def test_urdf_xml_to_model_data_without_world_link():
    model = urdf_xml_to_model_data(TEST_URDF, add_world_link=False)
    assert len(model["links"]) == 2
    assert len(model["joints"]) == 1
    assert model["joints"][0]["name"] == "joint1"


def test_urdf_unsupported_joint_type():
    bad_urdf = """<robot name="bad"><link name="a"/><link name="b"/><joint name="j" type="floating"><parent link="a"/><child link="b"/></joint></robot>"""
    with pytest.raises(ValueError, match="Unsupported URDF joint type"):
        urdf_xml_to_model_data(bad_urdf)


def test_urdf_with_world_name_conflict():
    bad_urdf = """<robot name="bad"><link name="world"/><link name="tool"/><joint name="j" type="fixed"><parent link="world"/><child link="tool"/></joint></robot>"""
    with pytest.raises(ValueError, match="already contains link name 'world'"):
        urdf_xml_to_model_data(bad_urdf, add_world_link=True)
