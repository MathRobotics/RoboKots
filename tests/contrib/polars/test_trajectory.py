import numpy as np
import pytest

pl = pytest.importorskip("polars")
from robokots.contrib.polars.state_table import RobotState


@pytest.mark.parametrize("quantity,dimension", [("pos", 3), ("vel", 6), ("force", 6), ("torque", 1)])
def test_trajectory_uses_quantity_dimension(quantity, dimension):
    values = np.arange(2*dimension, dtype=float).reshape(2, dimension)
    df = pl.DataFrame({f"a_joint_{quantity}": values.tolist(), f"b_joint_{quantity}": (values+10).tolist()})
    actual = RobotState.state_vecs_traj(df, ["b", "a"], "joint", quantity)
    assert actual.shape == (2, 2, dimension)
    np.testing.assert_array_equal(actual, np.stack([values+10, values]))


def test_unequal_dof_trajectory_requires_list_output():
    df = pl.DataFrame({"a_joint_torque": [[1.], [2.]], "b_joint_torque": [[3.,4.,5.], [6.,7.,8.]]})
    with pytest.raises(ValueError, match="list_output"):
        RobotState.state_vecs_traj(df, ["a", "b"], "joint", "torque")
    values = RobotState.state_vecs_traj(df, ["a", "b"], "joint", "torque", list_output=True)
    assert [v.shape for v in values] == [(2,1), (2,3)]
    assert RobotState.state_vecs_traj(df, [], "joint", "torque", list_output=True) == []
    with pytest.raises(ValueError, match="empty"):
        RobotState.state_vecs_traj(df, [], "joint", "torque")
