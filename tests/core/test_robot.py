import numpy as np
import pytest

from mathrobo import SE3
from robokots.core.robot import RobotStruct, LinkStruct, JointStruct, validate_model_data


def _valid_model_data():
    return {
        "schema_version": "0.0.2",
        "links": [
            {"id": 0, "name": "world"},
            {"id": 1, "name": "base"},
            {"id": 2, "name": "tool"},
        ],
        "joints": [
            {
                "id": 0,
                "name": "root",
                "type": "fixed",
                "parent_link_id": 0,
                "child_link_id": 1,
            },
            {
                "id": 1,
                "name": "joint1",
                "type": "revolute",
                "parent_link_id": 1,
                "child_link_id": 2,
                "axis": [0, 0, 1],
            },
        ],
    }


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


def test_robot_struct_from_dict_validates_schema_version():
    data = _valid_model_data()
    data.pop("schema_version")

    with pytest.raises(ValueError, match="schema_version"):
        RobotStruct.from_dict(data)


def test_robot_struct_from_dict_accepts_out_of_order_ids():
    data = _valid_model_data()
    data["links"] = [data["links"][1], data["links"][2], data["links"][0]]
    data["joints"] = [data["joints"][1], data["joints"][0]]

    robot = RobotStruct.from_dict(data)

    assert [link.id for link in robot.links] == [0, 1, 2]
    assert [link.name for link in robot.links] == ["world", "base", "tool"]
    assert [joint.id for joint in robot.joints] == [0, 1]
    assert [joint.name for joint in robot.joints] == ["root", "joint1"]


def test_robot_struct_from_dict_validates_ids_and_names():
    data = _valid_model_data()
    data["links"][1]["id"] = 3
    with pytest.raises(ValueError, match="links.id values"):
        RobotStruct.from_dict(data)

    data = _valid_model_data()
    data["joints"][1]["id"] = 0
    with pytest.raises(ValueError, match="joints.id values"):
        RobotStruct.from_dict(data)

    data = _valid_model_data()
    data["joints"][1]["name"] = "root"
    with pytest.raises(ValueError, match="Duplicate joint name"):
        RobotStruct.from_dict(data)


def test_model_validation_allows_non_tree_topology_for_future_closed_links():
    data = _valid_model_data()
    data["joints"][1]["parent_link_id"] = 0
    data["joints"][1]["child_link_id"] = 1

    validate_model_data(data)
    with pytest.raises(NotImplementedError, match="currently supports only tree topology"):
        RobotStruct.from_dict(data)


def test_robot_struct_from_dict_rejects_currently_unsupported_unreachable_topology():
    data = _valid_model_data()
    data["links"].append({"id": 3, "name": "orphan"})
    with pytest.raises(NotImplementedError, match="reachable"):
        RobotStruct.from_dict(data)


def test_robot_struct_from_dict_validates_joint_type_and_axis():
    data = _valid_model_data()
    data["joints"][1]["type"] = "fix"
    with pytest.raises(ValueError, match="Use 'fixed' instead"):
        RobotStruct.from_dict(data)

    data = _valid_model_data()
    data["joints"][1].pop("axis")
    with pytest.raises(ValueError, match="axis is required"):
        RobotStruct.from_dict(data)

    data = _valid_model_data()
    data["joints"][1]["axis"] = [0, 0, 0]
    with pytest.raises(ValueError, match="axis must be non-zero"):
        RobotStruct.from_dict(data)


def test_robot_struct_from_dict_accepts_spherical_rotation_vector_and_floating_expmap():
    data = _valid_model_data()
    data["links"].append({"id": 3, "name": "camera"})
    data["joints"][1] = {
        "id": 1,
        "name": "joint_s",
        "type": "spherical",
        "q_representation": "rotation_vector",
        "dof": 3,
        "parent_link_id": 1,
        "child_link_id": 2,
    }
    data["joints"].append(
        {
            "id": 2,
            "name": "joint_f",
            "type": "floating",
            "q_representation": "expmap",
            "dof": 6,
            "parent_link_id": 2,
            "child_link_id": 3,
        }
    )

    robot = RobotStruct.from_dict(data)

    assert robot.joint("joint_s").dof == 3
    assert robot.joint("joint_f").dof == 6
    assert robot.dof == 9


def test_robot_struct_from_dict_accepts_spherical_axis_angular_basis():
    data = _valid_model_data()
    angular = [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ]
    data["joints"][1] = {
        "id": 1,
        "name": "joint_s",
        "type": "spherical",
        "q_representation": "rotation_vector",
        "dof": 3,
        "axis": {"angular": angular},
        "parent_link_id": 1,
        "child_link_id": 2,
    }

    robot = RobotStruct.from_dict(data)

    assert np.allclose(robot.joint("joint_s").axis, np.asarray(angular))
    assert np.allclose(
        robot.joint("joint_s").select_mat,
        np.vstack([np.asarray(angular), np.zeros((3, 3))]),
    )
    assert robot.to_dict()["joints"][1]["axis"] == {"angular": angular}


def test_robot_struct_from_dict_validates_multi_dof_q_representation_and_dof():
    data = _valid_model_data()
    data["joints"][1] = {
        "id": 1,
        "name": "joint_s",
        "type": "spherical",
        "parent_link_id": 1,
        "child_link_id": 2,
    }
    with pytest.raises(ValueError, match="q_representation must be 'rotation_vector'"):
        RobotStruct.from_dict(data)

    data = _valid_model_data()
    data["joints"][1] = {
        "id": 1,
        "name": "joint_s",
        "type": "spherical",
        "q_representation": "rotation_vector",
        "dof": 2,
        "parent_link_id": 1,
        "child_link_id": 2,
    }
    with pytest.raises(ValueError, match="dof must be 3"):
        RobotStruct.from_dict(data)

    data = _valid_model_data()
    data["joints"][1] = {
        "id": 1,
        "name": "joint_s",
        "type": "spherical",
        "q_representation": "rotation_vector",
        "axis": {"angular": [[1, 0, 0], [0, 0, 0], [0, 0, 0]]},
        "parent_link_id": 1,
        "child_link_id": 2,
    }
    with pytest.raises(ValueError, match="axis.angular must be full rank"):
        RobotStruct.from_dict(data)


def test_robot_struct_from_dict_validates_inertia_dict():
    data = _valid_model_data()
    data["links"][1]["inertia"] = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="inertia must be a dictionary"):
        RobotStruct.from_dict(data)

    data = _valid_model_data()
    data["links"][1]["inertia"] = {"ixx": 1.0, "ixy": 0.0, "ixz": 0.0, "iyy": 1.0, "iyz": 0.0}
    with pytest.raises(ValueError, match="missing required keys"):
        RobotStruct.from_dict(data)


def test_robot_motion_owners_are_centralized_by_dof_index():
    link1 = LinkStruct(0, "link1", np.zeros(3), 1.0, np.eye(6), "soft")
    link2 = LinkStruct(1, "link2", np.zeros(3), 1.0, np.eye(6), "rigid")
    joint1 = JointStruct(0, "joint1", "revolute", np.array((1, 0, 0)), 0, 1, SE3())

    robot = RobotStruct([link1, link2], [joint1])

    assert [(owner.dof, owner.dof_index) for owner in robot.motion_owners()] == [(6, 0), (1, 6)]
    assert robot.motion_owner_dofs() == [6, 1]

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
    assert np.array_equal(joint.select_indeces, [2])
    assert isinstance(joint.origin, SE3)
    assert np.allclose(joint.origin.mat(), origin.mat())


def test_joint_struct_prismatic_init():
    origin = SE3.rand()
    joint = JointStruct(0, "joint_p", "prismatic", np.array((0, 1, 0)), 0, 1, origin)

    assert joint.type == "prismatic"
    assert joint.dof == 1
    assert np.allclose(joint.select_mat, np.array([[0], [0], [0], [0], [1], [0]]))


def test_joint_struct_fixed_init():
    joint = JointStruct(0, "joint_fixed", "fixed", np.array((0, 0, 0)), 0, 1, SE3())

    assert joint.type == "fixed"
    assert joint.dof == 0
    assert np.allclose(joint.select_mat, np.zeros((6, 0)))
    assert np.array_equal(joint.select_indeces, [])


def test_joint_struct_spherical_rotation_vector_init():
    joint = JointStruct(
        0,
        "joint_s",
        "spherical",
        np.array((0, 0, 0)),
        0,
        1,
        SE3(),
        q_representation="rotation_vector",
    )

    assert joint.type == "spherical"
    assert joint.q_representation == "rotation_vector"
    assert joint.dof == 3
    assert np.allclose(joint.select_mat, np.vstack([np.eye(3), np.zeros((3, 3))]))


def test_joint_struct_spherical_rotation_vector_uses_axis_angular_basis():
    angular = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    joint = JointStruct(
        0,
        "joint_s",
        "spherical",
        angular,
        0,
        1,
        SE3(),
        q_representation="rotation_vector",
    )

    assert np.allclose(joint.axis, angular)
    assert np.allclose(joint.select_mat, np.vstack([angular, np.zeros((3, 3))]))


def test_joint_struct_floating_expmap_init():
    joint = JointStruct(
        0,
        "joint_f",
        "floating",
        np.array((0, 0, 0)),
        0,
        1,
        SE3(),
        q_representation="expmap",
    )

    assert joint.type == "floating"
    assert joint.q_representation == "expmap"
    assert joint.dof == 6
    assert np.allclose(joint.select_mat, np.eye(6))


def test_joint_struct_requires_supported_representation_for_multi_dof_joints():
    with pytest.raises(ValueError, match="q_representation must be 'rotation_vector'"):
        JointStruct(0, "joint_s", "spherical", np.array((0, 0, 0)), 0, 1, SE3())


def test_joint_struct_rejects_legacy_fix_type():
    with pytest.raises(ValueError, match="Use 'fixed' instead"):
        JointStruct(0, "joint_fix", "fix", np.array((0, 0, 0)), 0, 1, SE3())

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
    assert np.allclose(selected_mat, mat[:, [1]])  # Only the first elements should be selected
 
def test_scatter():
    # Create a mock joint
    joint = JointStruct(0, "joint1", "revolute", np.array((0,1,0)), 0, 1, SE3())
    
    # Create a mock matrix
    mat = np.array([[2]])
    
    # Apply the scatter method
    scattered_mat = joint.scatter(mat)
    
    # Check if the scattered matrix is correct
    assert np.allclose(scattered_mat, np.array([[0], [2], [0], [0], [0], [0]]))  # Only the first elements should be scattered
