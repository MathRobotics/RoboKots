"""Compatibility of state module paths after the naming cleanup."""

import importlib
import pickle

import pytest


@pytest.mark.parametrize("old,new", [
    ("state", "state_spec"),
    ("outward_data", "outward_protocol"),
    ("state_dict", "state_dict_utils"),
    ("state_json", "state_jsonl"),
])
def test_old_and_new_paths_share_module_and_attributes(old, new):
    legacy = importlib.import_module(f"robokots.core.{old}")
    current = importlib.import_module(f"robokots.core.{new}")
    assert legacy is current
    for name in vars(current):
        assert getattr(legacy, name) is getattr(current, name)


def test_legacy_pickle_global_and_current_round_trip():
    from robokots.core.state_spec import StateType

    assert pickle.loads(b"crobokots.core.state\nStateType\n.") is StateType
    state = StateType("link", "tip", "momentum", "world")
    restored = pickle.loads(pickle.dumps(state))
    assert type(restored) is StateType
    assert vars(restored) == vars(state)
