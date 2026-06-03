import numpy as np

from robokots.core.state_batch import StateBatch


def _get_value(_robot, state, state_type):
    return state[state_type]


class _StateObject:
    def __init__(self, state):
        self._state = state

    def to_state_dict(self, _robot):
        return self._state


def test_state_batch_from_state_dicts_restores_state_info_shape():
    states = [
        {"vel": np.array([1, 2]), "acc": np.array([3, 4])},
        {"vel": np.array([5, 6]), "acc": np.array([7, 8])},
    ]
    batch = StateBatch.from_states(states, (2,), robot=None)

    np.testing.assert_allclose(batch.state_info(None, "vel", _get_value), np.array([[1, 2], [5, 6]]))
    np.testing.assert_allclose(
        batch.state_info_list(None, ["vel", "acc"], _get_value),
        np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
    )


def test_state_batch_from_outward_like_states_keeps_outward_states():
    states = [
        _StateObject({"vel": np.array([1, 2])}),
        _StateObject({"vel": np.array([3, 4])}),
    ]
    batch = StateBatch.from_states(states, (2,), robot=None)

    assert batch.outward_states == states
    np.testing.assert_allclose(batch.state_dicts[0]["vel"], np.array([1, 2]))
    np.testing.assert_allclose(batch.state_dicts[1]["vel"], np.array([3, 4]))
    parts = batch.state_info_list(None, ["vel"], _get_value, list_output=True)
    assert len(parts) == 1
    np.testing.assert_allclose(parts[0], np.array([[1, 2], [3, 4]]))
