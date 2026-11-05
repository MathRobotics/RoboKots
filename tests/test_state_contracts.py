"""Selection, batch export, reader capabilities, and explicit fallback contracts."""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from robokots.kots import Kots, StateType
from robokots import outward
from robokots.core.state.protocol import OutwardDataView, StateValueProvider
from robokots.core.state.batch import StateBatch
from robokots.api.state_cache import StateCache
from robokots.api.state import _batch_state_info_list
from robokots.core.state.spec import state_dict_key
from robokots.outward.rust.state import build_kinematics_outward_state_rust


MODELS = Path(__file__).parent / "test_model"


def make_kots(model="sample_robot", batch_shape=()):
    kots = Kots.from_json_file(str(MODELS / f"{model}.json"), order=4)
    motion = np.random.default_rng(82).normal(scale=.2, size=batch_shape + (kots.dof()*4,))
    kots.import_motions(motion)
    return kots, motion


@pytest.mark.parametrize("backend", ["numpy", "rust", "jax"])
@pytest.mark.parametrize("shape", [(), (2,), (2, 1)])
def test_mixed_selection_shape_order_and_empty(backend, shape):
    kots, _ = make_kots(batch_shape=shape)
    kots.kinematics(backend=backend)
    link = kots.link_name_list()[-1]
    states = [StateType("link", link, q) for q in ("vel", "pos", "rot", "frame")]
    parts = kots.state_info_list(states, list_output=True)
    expected = np.concatenate([
        np.asarray(part.mat() if hasattr(part, "mat") else part).reshape(shape + (-1,))
        for part in parts], axis=-1)
    actual = kots.state_info_list(states)
    assert actual.shape == shape + (34,)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(kots.state_tensor(states).data, expected)
    assert kots.state_info_list([]).shape == shape + (0,)
    assert kots.state_tensor([]).shape == shape + (0,)
    assert kots.state_info_list([], list_output=True) == []


@pytest.mark.parametrize("shape", [(), (2, 1)])
@pytest.mark.parametrize("backend", ["numpy", "rust"])
def test_mixed_dynamics_selection(backend, shape):
    kots, _ = make_kots(batch_shape=shape)
    kots.dynamics(backend=backend)
    states = [StateType("link", kots.link_name_list()[-1], "pos"),
              StateType("joint", kots.joint_name_list()[-1], "torque"),
              StateType("link", kots.link_name_list()[-1], "force")]
    result = kots.state_info_list(states)
    assert result.shape == shape + (10,)
    np.testing.assert_array_equal(result, np.concatenate(kots.state_info_list(states, list_output=True), axis=-1))


@pytest.mark.parametrize("backend", ["numpy", "rust", "array", "jax"])
def test_common_reader_protocol(backend):
    kots, motion = make_kots()
    state = (build_kinematics_outward_state_rust(kots.robot_, motion, order=4)
             if backend == "array" else kots.kinematics(backend=backend))
    assert isinstance(state, OutwardDataView)
    assert isinstance(state, StateValueProvider) == (backend == "rust")


@pytest.mark.parametrize("shape,count", [((), 1), ((2, 3), 1), ((0,), 0),
                                         ((-1,), 0), ((1.5,), 1), ((True,), 1)])
def test_state_batch_rejects_invalid_shape_and_count(shape, count):
    with pytest.raises(ValueError):
        StateBatch([object()] * count, shape)
    with pytest.raises(ValueError):
        StateBatch.from_states([object()] * count, shape)


def test_state_batch_empty_selection_and_matrix_values():
    batch = StateBatch.from_states([object(), object()], (np.int64(2), 1))
    assert _batch_state_info_list(batch, None, [], lambda *args: None).shape == (2, 1, 0)
    assert _batch_state_info_list(batch, None, [], lambda *args: None, list_output=True) == []


def test_soft_batch_mixes_joint_motion_and_link_values():
    model = json.loads((MODELS / "soft_rod.json").read_text())
    model["joints"][1].update(type="revolute", axis=[0, 0, 1])
    kots = Kots.from_json_data(model, order=4)
    kots.import_motions(np.zeros((2, 1, kots.dof()*4)))
    kots.kinematics()
    joint = next(joint for joint in kots.robot_.joints if joint.dof)
    states = [StateType("joint", joint.name, "coord"),
              StateType("link", kots.link_name_list()[-1], "pos")]
    actual = kots.state_info_list(states)
    parts = kots.state_info_list(states, list_output=True)
    assert actual.shape == (2, 1, joint.dof + 3)
    np.testing.assert_array_equal(actual, np.concatenate(parts, axis=-1))


@pytest.mark.parametrize("layout", ["flat", "dof_order"])
def test_empty_motion_batches_are_rejected(layout):
    kots, _ = make_kots()
    if layout == "flat":
        with pytest.raises(ValueError, match="empty batches"):
            kots.import_motions(np.zeros((2, 0, kots.dof()*4)))
    else:
        with pytest.raises(ValueError, match="empty batches"):
            kots.import_motion_array(np.zeros((0, kots.dof(), 4)))


def test_batch_export_rejects_inconsistent_keys():
    from robokots.state_io.dictionary import export_state_dict
    kots, motion = make_kots()
    first = kots.kinematics()
    kots.import_motions(motion)
    second = kots.dynamics()
    with pytest.raises(ValueError, match="identical dictionary keys"):
        export_state_dict(kots.robot_, StateBatch.from_states([first, second], (2,)))


@pytest.mark.parametrize("shape", [(2,), (2, 1)])
@pytest.mark.parametrize("model,backend", [("sample_robot", "numpy"), ("sample_robot", "rust"),
                                          ("sample_robot", "jax"), ("soft_rod", "numpy")])
def test_batch_export_is_dict_with_original_axes(model, backend, shape):
    kots, motion = make_kots(model, shape)
    actual = kots.kinematics(backend=backend, materialize_dict=True)
    assert isinstance(actual, dict)
    assert all(value.shape[:len(shape)] == shape for value in actual.values())
    for index in np.ndindex(shape):
        single, _ = make_kots(model)
        single.import_motions(motion[index])
        expected = single.kinematics(backend=backend, materialize_dict=True)
        assert actual.keys() == expected.keys()
        for key in expected:
            np.testing.assert_allclose(actual[key][index], expected[key], atol=1e-12)
    for value in actual.values():
        value[...] = 123
    fresh = kots.to_state_dict()
    key = state_dict_key("link", kots.link_name_list()[-1], "pos")
    assert not np.all(fresh[key] == 123)
    if model == "soft_rod":
        state = StateType("link", kots.link_name_list()[-1], "pos")
        np.testing.assert_allclose(kots.state_info_list([state]), fresh[key])


def test_state_cache_internal_type_error_is_not_retried():
    calls = []
    def builder(x, *, time, required):
        calls.append((time, required))
        if len(calls) > 1:
            raise TypeError("internal builder bug")
        return object()
    cache = StateCache(builder)
    pack = SimpleNamespace(revision=1, get=lambda: np.zeros(2))
    cache.update_if_needed(pack)
    old = cache.state
    pack.revision = 2
    with pytest.raises(TypeError, match="internal builder bug"):
        cache.update_if_needed(pack)
    assert len(calls) == 2
    assert cache.state is old
    assert cache.is_fresh(1) and not cache.is_fresh(2)


@pytest.mark.parametrize("error", [RuntimeError, ValueError, TypeError, AttributeError])
def test_unexpected_batched_kernel_error_propagates(monkeypatch, error):
    kots, _ = make_kots(batch_shape=(2,))
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise error("injected batch failure")
    monkeypatch.setattr(outward, "build_kinematics_outward_state", fail)
    with pytest.raises(error, match="injected batch failure"):
        kots.kinematics()
    assert len(calls) == 1


def test_explicit_batch_unsupported_uses_logged_scalar_fallback(monkeypatch, caplog):
    kots, _ = make_kots(batch_shape=(2,))
    original = outward.build_kinematics_outward_state
    calls = []
    def build(robot, motion, *args, **kwargs):
        calls.append(motion.ndim)
        if motion.ndim > 1:
            raise NotImplementedError("test unsupported batch")
        return original(robot, motion, *args, **kwargs)
    monkeypatch.setattr(outward, "build_kinematics_outward_state", build)
    with caplog.at_level("DEBUG", logger="robokots.api.state"):
        kots.kinematics()
    assert calls == [2, 1, 1]
    assert "test unsupported batch" in caplog.text


@pytest.mark.parametrize("error", [RuntimeError, ValueError, TypeError, AttributeError])
def test_unexpected_rust_derivative_error_propagates(monkeypatch, error):
    kots, _ = make_kots()
    kots.dynamics(backend="rust", order=3)
    states = [StateType("joint", kots.joint_name_list()[-1], "torque")]
    def fail(*args, **kwargs):
        raise error("injected Rust derivative failure")
    compiled = SimpleNamespace(
        dynamics_jacobian=fail, model_info=kots._rust_compiled_robot().model_info
    )
    monkeypatch.setattr(kots, "_rust_compiled_robot", lambda: compiled)
    with pytest.raises(error, match="injected Rust derivative failure"):
        kots._rust_torque_jacobian(states, 3)


def test_explicit_rust_unsupported_is_logged(monkeypatch, caplog):
    kots, _ = make_kots()
    kots.dynamics(backend="rust", order=3)
    def fail(*args, **kwargs):
        raise NotImplementedError("test unsupported Rust derivative")
    compiled = SimpleNamespace(
        dynamics_jacobian=fail, model_info=kots._rust_compiled_robot().model_info
    )
    monkeypatch.setattr(kots, "_rust_compiled_robot", lambda: compiled)
    states = [StateType("joint", kots.joint_name_list()[-1], "torque")]
    with caplog.at_level("DEBUG", logger="robokots.api.rust_derivatives"):
        assert kots._rust_torque_jacobian(states, 3) is None
    assert "test unsupported Rust derivative" in caplog.text


@pytest.mark.parametrize("operation,kernel", [
    ("dense", "outward_jacobian"), ("jvp", "outward_jacobian_matvec"),
    ("matrix", "outward_jacobian_matmul_rhs"), ("vjp", "outward_jacobian_transpose_matvec")])
def test_unexpected_python_derivative_error_is_not_retried(monkeypatch, operation, kernel):
    kots, _ = make_kots(batch_shape=(2, 1))
    kots.kinematics()
    states = [StateType("link", kots.link_name_list()[-1], "vel")]
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise ValueError("injected derivative shape bug")
    monkeypatch.setattr(outward, kernel, fail)
    n = kots.dof() * 2
    run = {"dense": lambda: kots.jacobian(states),
           "jvp": lambda: kots.jacobian_mul(states, np.ones(n)),
           "matrix": lambda: kots.jacobian_mul(states, np.ones((n, 2))),
           "vjp": lambda: kots.jacobian_transpose_mul(states, np.ones(6))}[operation]
    with pytest.raises(ValueError, match="injected derivative shape bug"):
        run()
    assert len(calls) == 1


@pytest.mark.parametrize("matrix", [False, True])
def test_batched_dynamics_vjp_does_not_build_dense_jacobian(monkeypatch, matrix):
    from robokots.outward.diff import outward_transpose_matvec as reverse
    kots, _ = make_kots(batch_shape=(3,))
    state = kots.dynamics(gravity=[.1, -.2, -9.81])
    states = [StateType("joint", joint.name, "torque_diff1") for joint in kots.robot_.joints if joint.dof]
    jacobian = kots.jacobian(states)
    shape = (3, jacobian.shape[-2], 2) if matrix else (3, jacobian.shape[-2])
    rhs = np.random.default_rng(21).normal(size=shape)
    expected = np.swapaxes(jacobian, -1, -2) @ (rhs if matrix else rhs[..., None])
    if not matrix:
        expected = expected[..., 0]
    def fail(*args, **kwargs):
        raise AssertionError("dense Jacobian must not be built for reverse products")
    monkeypatch.setattr(reverse, "outward_jacobian", fail, raising=False)
    actual = reverse.outward_jacobian_transpose_matvec(kots.robot_, state, states, rhs, max_time_order=4)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
