"""Public ownership boundaries and failed outward updates."""
from pathlib import Path

import numpy as np
import pytest

from robokots import outward
from robokots.kots import Kots, StateType


MODEL = Path(__file__).parent / "test_model" / "sample_robot.json"


def make_kots(batch_shape=()):
    kots = Kots.from_json_file(str(MODEL), order=4)
    motion = np.random.default_rng(71).normal(scale=.2, size=batch_shape + (kots.dof()*4,))
    kots.import_motions(motion)
    return kots, motion


@pytest.mark.parametrize("backend", ["numpy", "rust"])
@pytest.mark.parametrize("batch_shape", [(), (2, 1)])
def test_public_motion_ownership(backend, batch_shape):
    kots, motion = make_kots(batch_shape)
    original = motion.copy()
    kots.update_state(backend=backend)
    state = StateType("link", kots.link_name_list()[-1], "pos")
    expected = kots.state_info(state).copy()
    revision = kots.motions_.revision()
    motion[...] = 9
    kots.motions()[...] = 8
    np.testing.assert_array_equal(kots.motion(), original)
    assert kots.motions_.revision() == revision
    kots.update_state(backend=backend)
    np.testing.assert_array_equal(kots.state_info(state), expected)
    kots.import_motions(original * .4)
    assert kots.motions_.revision() > revision
    kots.update_state(backend=backend)
    assert not np.allclose(kots.state_info(state), expected)


@pytest.mark.parametrize("backend", ["numpy", "rust"])
@pytest.mark.parametrize("batch_shape", [(), (2, 1)])
@pytest.mark.parametrize("quantity", ["pos", "vel", "force", "torque"])
def test_public_state_arrays_are_snapshots(backend, batch_shape, quantity):
    kots, motion = make_kots(batch_shape)
    kots.dynamics(backend=backend)
    owner = "joint" if quantity == "torque" else "link"
    name = kots.joint_name_list()[-1] if owner == "joint" else kots.link_name_list()[-1]
    state = StateType(owner, name, quantity)
    for read in (lambda: kots.state_info(state),
                 lambda: kots.state_info_list([state], list_output=True)[0]):
        value = read()
        expected = value.copy()
        value[...] = 123
        np.testing.assert_array_equal(read(), expected)
    snapshot = kots.state_info(state)
    expected = snapshot.copy()
    kots.import_motions(motion * .4)
    kots.dynamics(backend=backend)
    np.testing.assert_array_equal(snapshot, expected)


@pytest.mark.parametrize("backend", ["numpy", "rust"])
def test_public_frame_is_independent(backend):
    kots, motion = make_kots()
    kots.kinematics(backend=backend)
    state = StateType("link", kots.link_name_list()[-1], "frame")
    value = kots.state_info(state)
    reference = value.mat().copy()
    value.pos()[...] = 123
    np.testing.assert_array_equal(kots.state_info(state).mat(), reference)
    snapshot = kots.state_info(state)
    frame = snapshot.mat().copy()
    kots.import_motions(motion * .4)
    kots.kinematics(backend=backend)
    np.testing.assert_array_equal(snapshot.mat(), frame)


@pytest.mark.parametrize("backend", ["numpy", "rust"])
@pytest.mark.parametrize("failure", ["backend", "order", "low_order", "gravity"])
def test_invalid_dynamics_preserves_state_and_gravity(backend, failure):
    kots, _ = make_kots()
    old_gravity = np.array([0., 0., -9.81])
    state = kots.dynamics(backend=backend, gravity=old_gravity)
    expected = kots.to_state_dict()
    kwargs = dict(backend=backend, gravity=[0, 0, 1])
    kwargs.update({"backend": "unsupported"} if failure == "backend" else
                  {"order": 5} if failure == "order" else
                  {"order": 1} if failure == "low_order" else {"gravity": [np.nan, 0, 0]})
    with pytest.raises(ValueError):
        kots.dynamics(**kwargs)
    assert kots.outward_state_ is state
    np.testing.assert_array_equal(kots.gravity_, old_gravity)
    for key, value in kots.to_state_dict().items():
        np.testing.assert_array_equal(value, expected[key])


@pytest.mark.parametrize("batch_shape", [(), (2, 1)])
def test_numpy_kernel_failure_preserves_previous_state(monkeypatch, batch_shape):
    kots, _ = make_kots(batch_shape)
    state = kots.dynamics(gravity=[0, 0, -9.81])
    def fail(*args, **kwargs):
        raise RuntimeError("injected kernel failure")
    monkeypatch.setattr(outward, "build_dynamics_outward_state", fail)
    with pytest.raises(RuntimeError, match="injected"):
        kots.dynamics(gravity=[0, 0, 1])
    assert kots.outward_state_ is state
    np.testing.assert_array_equal(kots.gravity_, [0, 0, -9.81])


@pytest.mark.parametrize("batch_shape", [(), (2, 1)])
def test_rust_partial_failure_discards_workspace(monkeypatch, batch_shape):
    kots, _ = make_kots(batch_shape)
    state = kots.dynamics(backend="rust", gravity=[0, 0, -9.81])
    original = state.compute_dynamics
    def fail(motion, gravity=None):
        original(motion, gravity)  # Simulate failure after the workspace changed.
        raise RuntimeError("injected partial failure")
    monkeypatch.setattr(state, "compute_dynamics", fail)
    with pytest.raises(RuntimeError, match="partial failure"):
        kots.dynamics(backend="rust", gravity=[0, 0, 1])
    np.testing.assert_array_equal(kots.gravity_, [0, 0, -9.81])
    assert not kots._rust_outward_data_cache_
    assert not kots._rust_outward_data_cache_state_
    with pytest.raises(ValueError, match="No computed state"):
        kots.to_state_dict()
    recovered = kots.update_state(is_dynamics=True, backend="rust")
    assert recovered is not state
    np.testing.assert_array_equal(recovered.gravity, [0, 0, -9.81])


@pytest.mark.parametrize("backend", ["numpy", "rust"])
@pytest.mark.parametrize("batch_shape", [(), (2, 1)])
def test_export_failure_does_not_commit_new_gravity(monkeypatch, backend, batch_shape):
    from robokots.state_io import dictionary
    kots, _ = make_kots(batch_shape)
    state = kots.dynamics(backend=backend, gravity=[0, 0, -9.81])
    def fail(*args, **kwargs):
        raise RuntimeError("injected export failure")
    monkeypatch.setattr(dictionary, "export_state_dict", fail)
    with pytest.raises(RuntimeError, match="export failure"):
        kots.dynamics(backend=backend, gravity=[0, 0, 1], materialize_dict=True)
    np.testing.assert_array_equal(kots.gravity_, [0, 0, -9.81])
    assert kots.outward_state_ is (state if backend == "numpy" else None)


@pytest.mark.parametrize("batch_shape", [(), (2, 1)])
def test_import_motion_array_owns_input(batch_shape):
    kots, _ = make_kots(batch_shape)
    values = kots.motion_array()
    expected = values.copy()
    revision = kots.motions_.revision()
    kots.import_motion_array(values)
    assert kots.motions_.revision() == revision + 1
    values[...] = 100
    np.testing.assert_array_equal(kots.motion_array(), expected)


def test_rust_direct_update_commits_gravity_only_after_success():
    kots, _ = make_kots()
    state = kots.update_rust_data(is_dynamics=True, gravity=[.1, .2, -9.81])
    np.testing.assert_array_equal(kots.gravity_, state.gravity)
    with pytest.raises(ValueError):
        kots.update_rust_data(is_dynamics=True, gravity=[0, np.nan, 0])
    assert kots.outward_state_ is state
    np.testing.assert_array_equal(kots.gravity_, [.1, .2, -9.81])
