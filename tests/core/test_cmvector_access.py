"""Normalized state series must never round-trip through ordinary derivatives."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from mathrobo import CMVector

from robokots.core.state.access import state_cmvec
from robokots.state_io.dictionary import state_dict_to_cmvec


def read_core(state, order):
    return state_cmvec(state, "link1", "link", "momentum", order)


def read_dictionary(state, order):
    return state_dict_to_cmvec(state, "link1", "link", "momentum", order)


class CoefficientOnlyView:
    """Legacy backend with coefficients, but without truncate()."""

    def __init__(self, coefficients):
        self.coefficients = coefficients
        self._n = coefficients.shape[-2]

    def cm_vecs(self):
        return self.coefficients

    def vecs(self):
        raise AssertionError("ordinary derivatives must not be reconstructed")


@pytest.mark.parametrize("reader", [read_core, read_dictionary])
@pytest.mark.parametrize("batch", [(), (2, 3)])
@pytest.mark.parametrize("legacy", [False, True])
def test_cut_state_coefficients_without_conversion(reader, batch, legacy, monkeypatch):
    coefficients = np.arange(np.prod(batch + (5, 6)), dtype=np.float32).reshape(batch + (5, 6))
    value = CoefficientOnlyView(coefficients) if legacy else CMVector.set_cmvecs(coefficients)
    state = SimpleNamespace(cmvec=lambda *args: value)
    if not legacy:
        def forbidden(*args):
            raise AssertionError("ordinary derivatives must not be reconstructed")
        monkeypatch.setattr(CMVector, "vecs", forbidden)
        original = CMVector.truncate
        calls = []
        def tracked(self, order):
            calls.append(order)
            return original(self, order)
        monkeypatch.setattr(CMVector, "truncate", tracked)
    result = reader(state, 3)
    np.testing.assert_array_equal(result.cm_vecs(), coefficients[..., :3, :])
    assert result.cm_vecs().shape == batch + (3, 6)
    assert result.cm_vecs().dtype == np.float32
    assert np.shares_memory(result.cm_vecs(), coefficients)
    assert value._n == 5
    if not legacy:
        assert calls == [3]


@pytest.mark.parametrize("reader", [read_core, read_dictionary])
def test_read_full_order_reuses_value(reader):
    value = CMVector.set_cmvecs(np.ones((5, 6)))
    state = SimpleNamespace(cmvec=lambda *args: value)
    assert reader(state, 5) is value


@pytest.mark.parametrize("reader", [read_core, read_dictionary])
@pytest.mark.parametrize("order", [-1, 0, 6])
def test_state_series_rejects_invalid_order(reader, order):
    state = SimpleNamespace(cmvec=lambda *args: CMVector.set_cmvecs(np.ones((5, 6))))
    with pytest.raises(ValueError, match="Invalid order"):
        reader(state, order)


@pytest.mark.parametrize("batch", [(), (2, 1)])
def test_selected_numpy_dynamics_jacobian_after_truncation(batch):
    from robokots.kots import Kots, StateType

    model = Path(__file__).resolve().parents[1] / "test_model" / "branched_fixed.urdf"
    kots = Kots.from_urdf_file(str(model), order=5, backend="numpy")
    motion = np.random.default_rng(31).normal(scale=.2, size=batch + (kots.dof(), 5))
    states = [
        StateType("link", "a_tip", "force"),
        StateType("link", "a_tip", "force", "world"),
        StateType("joint", "a_shoulder", "momentum_diff1", "world"),
    ]

    def values(current):
        kots.import_motion_array(current)
        kots.dynamics(backend="numpy", gravity=(.2, -.3, -9.81))
        return kots.state_info_list(states)

    values(motion)
    analytic = kots.jacobian(states)
    columns = []
    for owner in range(kots.dof()):
        for derivative in range(StateType.max_time_order(states)):
            step = np.zeros_like(motion)
            step[..., owner, derivative] = 1e-6
            columns.append((values(motion + step) - values(motion - step)) / 2e-6)
    np.testing.assert_allclose(analytic, np.stack(columns, axis=-1), atol=2e-7, rtol=2e-7)
