"""Bulk state expansion preserves ordinary constructors and ownership."""
import pytest
from robokots.core.state.spec import StateType, keys_joint_motion, keys_momentum, keys_force, keys_torque


@pytest.mark.parametrize("key", keys_joint_motion + keys_momentum + keys_force + keys_torque)
@pytest.mark.parametrize("frame", [None, "local", "world"])
def test_owner_expansion_matches_constructors(key, frame):
    names = ["shoulder", "elbow", "shoulder"]
    actual = StateType._for_owners("joint", names, key, frame)
    expected = [StateType("joint", name, key, frame) for name in names]
    assert [vars(s) for s in actual] == [vars(s) for s in expected]
    actual[0].owner_name = "changed"
    actual[0].key_order = 99
    assert vars(actual[2]) == vars(expected[2])
    assert vars(StateType._for_owners("joint", names, key, frame)[0]) == vars(expected[0])


def test_empty_owner_expansion():
    assert StateType._for_owners("joint", [], "torque") == []
