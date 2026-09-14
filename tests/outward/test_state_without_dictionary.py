"""Computational paths must work with dictionary export disabled."""

from pathlib import Path

import numpy as np
import pytest

from robokots.core.outward_state import OutwardState, ArrayOutwardState
from robokots.core.state_spec import StateType
from robokots.kots import Kots
from robokots.state_io import dictionary


MODELS = Path(__file__).resolve().parents[1] / "test_model"


def _forbid_export(*args, **kwargs):
    raise AssertionError("Computation attempted dictionary export")


@pytest.mark.parametrize("batch_shape", [(), (2, 1)])
def test_array_momentum_only_export(batch_shape):
    pytest.importorskip("robokots._rust")
    from robokots.outward.rust.state import build_dynamics_outward_state_rust

    kots = Kots.from_json_file(str(MODELS / "sample_robot.json"), order=2)
    motion = np.random.default_rng(32).normal(size=batch_shape + (kots.dof() * 2,))
    state = build_dynamics_outward_state_rust(kots.robot_, motion, dynamics_order=0)
    actual = dictionary.export_state_dict(kots.robot_, state)
    kots.import_motions(motion)
    expected = kots.dynamics(backend="numpy", materialize_dict=True)
    assert actual.keys() == expected.keys()
    assert not any("force" in key or "torque" in key for key in actual)
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-12)


def test_export_uses_only_read_interface():
    kots = Kots.from_json_file(str(MODELS / "sample_robot.json"), order=4)
    kots.import_motions(np.random.default_rng(8).normal(size=kots.dof() * 4))
    state = kots.dynamics()

    class Reader:
        cmtm = staticmethod(state.cmtm)
        quantity_series = staticmethod(state.quantity_series)

    actual = dictionary.export_state_dict(kots.robot_, Reader())
    expected = kots.to_state_dict()
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])
        assert not np.shares_memory(actual[key], expected[key])


def test_rust_minimal_export_contains_only_available_quantities():
    pytest.importorskip("robokots._rust")
    kots = Kots.from_json_file(str(MODELS / "sample_robot.json"), order=4)
    motion = np.random.default_rng(18).normal(size=kots.dof() * 4)
    kots.import_motions(motion)
    state = kots.dynamics(backend="rust")
    reference = kots.to_state_dict()
    state.compute_dynamics_minimal(motion)
    actual = dictionary.export_state_dict(kots.robot_, state)
    expected_keys = {key for key in reference if "momentum" not in key and "force" not in key}
    assert actual.keys() == expected_keys
    for key in actual:
        np.testing.assert_allclose(actual[key], reference[key], atol=1e-12)


@pytest.mark.parametrize("backend", ["numpy", "rust", "jax"])
@pytest.mark.parametrize("batch_shape", [(), (2, 1)])
def test_computation_without_dictionary_export(monkeypatch, backend, batch_shape):
    if backend == "rust":
        pytest.importorskip("robokots._rust")
        from robokots.outward.rust.data import RustOutwardState
        assert not hasattr(RustOutwardState, "to_state_dict")
    for cls in (OutwardState, ArrayOutwardState):
        assert not hasattr(cls, "to_state_dict")
    monkeypatch.setattr(Kots, "to_state_dict", _forbid_export)
    monkeypatch.setattr(dictionary, "export_state_dict", _forbid_export)

    kots = Kots.from_urdf_file(str(MODELS / "branched_fixed.urdf"), order=4)
    rng = np.random.default_rng(105)
    kots.import_motions(rng.normal(scale=0.2, size=batch_shape + (kots.dof() * 4,)))
    if backend == "jax":
        kots.kinematics(backend=backend)
        states = [StateType("link", "a_tip", "acc")]
    else:
        kots.dynamics(backend=backend, gravity=[0.2, -0.3, -9.81])
        states = [StateType("link", "a_tip", "force_diff1", "world"),
                  StateType("joint", "a_shoulder", "torque_diff1")]
    assert not hasattr(kots, "state_dict_")
    assert all(np.all(np.isfinite(value)) for value in kots.state_info_list(states, list_output=True))
    jacobian = kots.jacobian(states)
    right = rng.normal(size=jacobian.shape[-1])
    left = rng.normal(size=jacobian.shape[-2])
    np.testing.assert_allclose(kots.jacobian_mul(states, right), jacobian @ right,
                               atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(kots.jacobian_transpose_mul(states, left),
                               np.swapaxes(jacobian, -1, -2) @ left, atol=1e-9, rtol=1e-9)
    if not batch_shape:
        np.testing.assert_allclose(jacobian, kots.jacobian(states, numerical=True),
                                   atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("backend", ["numpy", "rust"])
@pytest.mark.parametrize("batch_shape", [(), (2, 1)])
def test_export_is_independent_snapshot(backend, batch_shape):
    if backend == "rust":
        pytest.importorskip("robokots._rust")
    kots = Kots.from_json_file(str(MODELS / "sample_robot.json"), order=4)
    motion = np.random.default_rng(6).normal(size=batch_shape + (kots.dof() * 4,))
    kots.import_motions(motion)
    snapshot = kots.dynamics(backend=backend, materialize_dict=True)
    reference = {name: np.array(value, copy=True) for name, value in snapshot.items()}
    for value in snapshot.values():
        np.asarray(value)[...] = np.nan
    for name, value in kots.to_state_dict().items():
        np.testing.assert_array_equal(value, reference[name])
    snapshot = kots.to_state_dict()
    kots.import_motions(motion * 0.5)
    kots.dynamics(backend=backend)
    for name, value in snapshot.items():
        np.testing.assert_array_equal(value, reference[name])


def test_soft_link_batch_reads_objects_without_export(monkeypatch):
    monkeypatch.setattr(dictionary, "export_state_dict", _forbid_export)
    kots = Kots.from_json_file(str(MODELS / "soft_rod.json"), order=3)
    motion = np.random.default_rng(24).normal(scale=0.1, size=(2, 1, kots.dof() * 3))
    kots.import_motions(motion)
    kots.kinematics()
    state = StateType("link", "rod1", "acc")
    actual = kots.state_info(state)
    for i in range(2):
        single = Kots.from_json_file(str(MODELS / "soft_rod.json"), order=3)
        single.import_motions(motion[i, 0])
        single.kinematics()
        np.testing.assert_allclose(actual[i, 0], single.state_info(state))
