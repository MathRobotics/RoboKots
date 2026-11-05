import numpy as np

from robokots.core import RobotStruct


def joint_child_link_block_indices(robot: RobotStruct, block_size: int) -> np.ndarray:
    """Return flat link-block indices ordered by ``robot.joints``."""
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if not robot.joints:
        return np.zeros(0, dtype=int)
    return np.concatenate([
        np.arange(joint.child_link_id * block_size, (joint.child_link_id + 1) * block_size)
        for joint in robot.joints
    ])


def take_joint_child_link_blocks(
    values: np.ndarray,
    robot: RobotStruct,
    block_size: int,
    axis: int = -1,
) -> np.ndarray:
    """Gather link blocks into joint order using each joint's child link."""
    indices = joint_child_link_block_indices(robot, block_size)
    return np.take(values, indices, axis=axis)


def take_joint_child_link_matrix_blocks(
    matrix: np.ndarray,
    robot: RobotStruct,
    row_block_size: int,
    col_block_size: int,
) -> np.ndarray:
    """Gather both axes of a link-block matrix into joint order."""
    matrix = take_joint_child_link_blocks(matrix, robot, row_block_size, axis=-2)
    return take_joint_child_link_blocks(matrix, robot, col_block_size, axis=-1)


def scatter_joint_child_link_blocks(
    values: np.ndarray,
    robot: RobotStruct,
    block_size: int,
) -> np.ndarray:
    """Scatter joint-ordered blocks to their child-link slots on the last axis."""
    values = np.asarray(values)
    expected = robot.joint_num * block_size
    if values.shape[-1] != expected:
        raise ValueError(
            f"joint block length must be {expected}, got {values.shape[-1]}"
        )
    result = np.zeros(values.shape[:-1] + (robot.link_num * block_size,), dtype=values.dtype)
    for joint_index, joint in enumerate(robot.joints):
        source = joint_index * block_size
        target = joint.child_link_id * block_size
        result[..., target:target + block_size] += values[..., source:source + block_size]
    return result
