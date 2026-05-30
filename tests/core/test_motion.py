import numpy as np

from robokots.core.motion import MotionTensor, RobotMotions
        
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


def test_set_aliases_with_accel_diff():
    motions = RobotMotions(test_robot_dof)
    motions.set_aliases(["coord", "veloc", "accel", "accel_diff1"])
    assert motions.aliases == ["coord", "veloc", "accel", "accel_diff1"]
    assert motions.motion_num == 4
    assert motions.motions.shape == (12,)
        
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
    assert np.array_equal(motions.gen_values("coord"), np.array([1, 4, 7]))
    assert np.array_equal(motions.gen_values("veloc"), np.array([2, 5, 8]))
    assert np.array_equal(motions.gen_values("accel"), np.array([3, 6, 9]))
  
# Test the coord, veloc, and accel methods
def test_coord_veloc_accel():
    motions = RobotMotions(test_robot_dof)
    motions.set_motion(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    assert np.array_equal(motions.coord(), np.array([1, 4, 7]))
    assert np.array_equal(motions.veloc(), np.array([2, 5, 8]))
    assert np.array_equal(motions.accel(), np.array([3, 6, 9]))
    
# Test the gen_value method
def test_gen_value():
    motions = RobotMotions(test_robot_dof)
    motions.set_motion(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    assert np.array_equal(motions.gen_value(test_dof, test_dof_index, "coord"), np.array([4]))
    assert np.array_equal(motions.gen_value(test_dof, test_dof_index, "veloc"), np.array([5]))
    assert np.array_equal(motions.gen_value(test_dof, test_dof_index, "accel"), np.array([6]))

# Test the joint_coord, joint_veloc, and joint_accel methods
def test_joint_coord_veloc_accel():
    motions = RobotMotions(test_robot_dof)
    motions.set_motion(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    assert np.array_equal(motions.joint_coord(test_dof, test_dof_index), np.array([4]))
    assert np.array_equal(motions.joint_veloc(test_dof, test_dof_index), np.array([5]))
    assert np.array_equal(motions.joint_accel(test_dof, test_dof_index), np.array([6]))
    
# Test the link_coord, link_veloc, and link_accel methods
def test_link_coord_veloc_accel():
    motions = RobotMotions(test_robot_dof)
    motions.set_motion(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    assert np.array_equal(motions.link_coord(test_dof, test_dof_index), np.array([4]))
    assert np.array_equal(motions.link_veloc(test_dof, test_dof_index), np.array([5]))
    assert np.array_equal(motions.link_accel(test_dof, test_dof_index), np.array([6]))


def test_set_motion_validates_last_dimension():
    motions = RobotMotions(test_robot_dof)
    try:
        motions.set_motion(np.zeros((2, 8)))
    except ValueError as e:
        assert "last dimension must be 9" in str(e)
    else:
        assert False, "Expected ValueError not raised"


def test_batched_motion_accessors_keep_batch_shape():
    motions = RobotMotions(test_robot_dof)
    values = np.arange(2 * 9, dtype=float).reshape(2, 9)
    motions.set_motion(values)

    assert motions.batch_shape() == (2,)
    assert np.array_equal(motions.coord(), values[:, [0, 3, 6]])
    assert np.array_equal(motions.joint_motions(1, 1), values[:, 3:6].reshape(2, 3, 1))


def test_gen_values_respects_multi_dof_owner_blocks():
    motions = RobotMotions(3, owner_dofs=[2, 1])
    motions.set_motion(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))

    assert np.array_equal(motions.coord(), np.array([1, 2, 7]))
    assert np.array_equal(motions.veloc(), np.array([3, 4, 8]))
    assert np.array_equal(motions.accel(), np.array([5, 6, 9]))


def test_to_vector_respects_order_and_cm_scaling():
    motions = RobotMotions(3, owner_dofs=[2, 1])
    motions.set_motion(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))

    assert np.array_equal(motions.to_vector(order=2), np.array([1, 2, 3, 4, 7, 8]))
    np.testing.assert_allclose(
        motions.to_vector(order=3, cm=True),
        np.array([1, 2, 3, 4, 2.5, 3, 7, 8, 4.5]),
    )


def test_to_vector_full_order_returns_copy_of_flat_storage():
    motions = RobotMotions(3, owner_dofs=[2, 1])
    motions.set_motion(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float))

    vector = motions.to_vector()
    vector[0] = 100

    assert motions.motions[0] == 1


def test_to_derivative_vector_uses_tail_and_batch_shape():
    motions = RobotMotions(3, owner_dofs=[2, 1])
    values = np.arange(2 * 9, dtype=float).reshape(2, 9)
    tail = np.array([100, 200, 300], dtype=float)
    motions.set_motion(values)

    actual = motions.to_derivative_vector(order=3, tail=tail)

    expected0 = np.array([2, 3, 4, 5, 100, 200, 7, 8, 300], dtype=float)
    expected1 = np.array([11, 12, 13, 14, 100, 200, 16, 17, 300], dtype=float)
    np.testing.assert_allclose(actual, np.stack([expected0, expected1]))


def test_to_derivative_vector_cm_scales_output_positions():
    motions = RobotMotions(3, owner_dofs=[2, 1])
    motions.set_motion(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))

    np.testing.assert_allclose(
        motions.to_derivative_vector(order=3, tail=np.array([10, 20, 30]), cm=True),
        np.array([3, 4, 5, 6, 5, 10, 8, 9, 15]),
    )


def test_motion_tensor_converts_flat_owner_major_to_dof_order():
    motions = RobotMotions(3, owner_dofs=[2, 1])
    flat = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)

    tensor = MotionTensor.from_flat_owner_major(flat, motions.owner_layout, order=3)

    assert tensor.tensor.axes == ("dof", "order")
    np.testing.assert_allclose(
        tensor.as_dof_order().data,
        np.array(
            [
                [1, 3, 5],
                [2, 4, 6],
                [7, 8, 9],
            ],
            dtype=float,
        ),
    )
    np.testing.assert_allclose(tensor.to_flat_owner_major().data, flat)


def test_motion_tensor_owner_block_is_computation_layout():
    motions = RobotMotions(3, owner_dofs=[2, 1])
    values = np.arange(2 * 9, dtype=float).reshape(2, 9)
    tensor = MotionTensor.from_flat_owner_major(values, motions.owner_layout, order=3)

    block = tensor.owner_block(motions.owner_layout[0])

    assert block.axes == ("batch0", "order", "owner_dof")
    assert block.shape == (2, 3, 2)
    np.testing.assert_allclose(block.data[0], np.array([[0, 1], [2, 3], [4, 5]], dtype=float))


def test_robot_motions_dof_order_roundtrip():
    motions = RobotMotions(3, owner_dofs=[2, 1])
    dof_order = np.array(
        [
            [1, 3, 5],
            [2, 4, 6],
            [7, 8, 9],
        ],
        dtype=float,
    )

    motions.set_dof_order(dof_order)

    np.testing.assert_allclose(motions.motions, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float))
    np.testing.assert_allclose(motions.to_dof_order(), dof_order)
