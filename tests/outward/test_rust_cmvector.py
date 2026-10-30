"""Rust vector adapters follow the canonical CMVector storage contract."""
import numpy as np
import pytest
from mathrobo import CMVector
from robokots.outward.rust.data import _RustCMVectorView, _as_rust_cmvector


def test_cm_input_and_truncate_avoid_ordinary_conversion():
    class CoefficientsOnly:
        def cm_vecs(self):
            return coefficients

        def vecs(self):
            raise AssertionError("ordinary conversion")

    coefficients = np.arange(2 * 11 * 6, dtype=float).reshape(2, 11, 6)
    vector = _as_rust_cmvector(CoefficientsOnly())
    assert isinstance(vector, CMVector)
    assert vector.cm_vecs() is coefficients
    prefix = vector.truncate(5)
    assert isinstance(prefix, _RustCMVectorView)
    assert np.shares_memory(prefix.cm_vecs(), coefficients)
    prefix.cm_vecs()[..., 3, :] = 7
    np.testing.assert_array_equal(vector.vecs()[..., 3, :], 42)
    detached = vector.vecs()
    detached[...] = -1
    np.testing.assert_array_equal(vector.cm_vecs(), coefficients)
    for invalid in (-1, 12):
        with pytest.raises(ValueError):
            vector.truncate(invalid)


def test_ordinary_input_matches_cmvector_and_arithmetic():
    raw = np.arange(2 * 11 * 6, dtype=float).reshape(2, 11, 6) / 7
    vector = _RustCMVectorView(raw)
    expected = CMVector(raw)
    np.testing.assert_array_equal(vector.cm_vecs(), expected.cm_vecs())
    np.testing.assert_array_equal(vector.vecs(), expected.vecs())
    np.testing.assert_array_equal((vector + vector).cm_vecs(), 2 * expected.cm_vecs())
    np.testing.assert_array_equal((vector - vector).cm_vecs(), 0)


@pytest.mark.parametrize("order", [2, 3, 6, 11])
@pytest.mark.parametrize("shape", [(), (2, 1)])
@pytest.mark.parametrize("gravity", [[0., 0., 0.], [.2, -.3, -9.81]])
def test_native_cm_series_matches_numpy_without_ordinary_getters(order, shape, gravity, monkeypatch):
    from pathlib import Path
    from robokots.kots import Kots

    pytest.importorskip("robokots._rust")
    model = Path(__file__).resolve().parents[1] / "test_model/branched_fixed.urdf"
    rust = Kots.from_urdf_file(str(model), order=order, backend="rust")
    reference = Kots.from_urdf_file(str(model), order=order, backend="numpy")
    x = np.random.default_rng(82).normal(scale=.2, size=shape + (rust.dof() * order,))
    for k in (rust, reference):
        k.import_motions(x)
        k.dynamics(gravity=gravity)
    state = rust.outward_state_

    def forbidden(*args, **kwargs):
        raise AssertionError("CM access reconstructed ordinary state values")

    monkeypatch.setattr(state, "_momentum", forbidden)
    monkeypatch.setattr(state, "_force", forbidden)
    monkeypatch.setattr(_RustCMVectorView, "__init__", forbidden)
    for owner, names in [("link", rust.link_name_list()), ("joint", rust.joint_name_list())]:
        for family in ("momentum", "force"):
            if family == "force" and order == 2:
                with pytest.raises(ValueError):
                    state.cmvec(owner, names[0], family)
                continue
            expected_map = getattr(reference.outward_state_, f"{owner}_{family}")
            for name in names:
                expected = expected_map[name]
                vector = state.cmvec(owner, name, family)
                assert isinstance(vector, _RustCMVectorView)
                np.testing.assert_allclose(vector.cm_vecs(), expected.cm_vecs(), rtol=1e-11, atol=1e-11)
                np.testing.assert_allclose(vector.vecs(), expected.vecs(), rtol=1e-10, atol=1e-10)
                ids = state.link_ids if owner == "link" else state.joint_ids
                native_get = getattr(state.raw_data, f"{owner}_{family}_cm")
                detached = native_get(ids[name])
                detached[...] = 123
                again = native_get(ids[name]).reshape(vector.cm_vecs().shape)
                np.testing.assert_allclose(again, vector.cm_vecs(), rtol=0, atol=0)
                assert np.shares_memory(vector.truncate(1).cm_vecs(), vector.cm_vecs())
