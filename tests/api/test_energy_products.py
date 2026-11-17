from pathlib import Path

import numpy as np
import pytest

from robokots.kots import Kots, StateType


MODEL = Path(__file__).resolve().parents[1] / "test_model/branched_fixed.urdf"


@pytest.mark.parametrize("backend", ["numpy", "rust"])
@pytest.mark.parametrize("batch_shape", [(), (6,), (2, 3)])
@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("shared_rhs", [False, True])
@pytest.mark.parametrize("selection", ["energy", "repeated_energy", "mixed"])
def test_energy_products_preserve_batch_axes(backend, batch_shape, matrix, shared_rhs, selection, monkeypatch):
    if backend == "rust":
        pytest.importorskip("robokots._rust", exc_type=ImportError)
    kots = Kots.from_urdf_file(str(MODEL), order=4, backend=backend)
    rng = np.random.default_rng(762)
    kots.import_motions(rng.normal(scale=.2, size=batch_shape + (kots.dof() * 4,)))
    kots.dynamics(gravity=[.2, -.3, -9.81])
    energy = StateType("total_body", "total_body", "kinetic_energy")
    states = {
        "energy": [energy],
        "repeated_energy": [energy, energy],
        "mixed": [StateType("joint", "a_shoulder", "torque_diff1"), energy,
                  StateType("link", "a_tip", "vel"), energy],
    }[selection]
    parts = kots.jacobian(states, list_output=True)
    jacobian = np.concatenate(parts, axis=-2)
    rhs_batch = () if shared_rhs else batch_shape
    columns = (2,) if matrix else ()
    tangent = rng.normal(size=rhs_batch + (jacobian.shape[-1],) + columns)
    cotangent = rng.normal(size=rhs_batch + (jacobian.shape[-2],) + columns)
    expected_jvp = jacobian @ tangent if matrix else (jacobian @ tangent[..., None])[..., 0]
    transpose = jacobian.swapaxes(-1, -2)
    expected_vjp = transpose @ cotangent if matrix else (transpose @ cotangent[..., None])[..., 0]

    def forbid_dense(*args, **kwargs):
        raise AssertionError("products must use direct paths")

    monkeypatch.setattr(kots, "jacobian", forbid_dense)
    target = energy if selection == "energy" else states
    actual = kots.jacobian_mul(target, tangent)
    assert actual.shape == expected_jvp.shape
    np.testing.assert_allclose(actual, expected_jvp, atol=1e-10, rtol=1e-10)
    result_parts = kots.jacobian_mul(target, tangent, list_output=True)
    np.testing.assert_allclose(np.concatenate(result_parts, axis=-2 if matrix else -1), expected_jvp,
                               atol=1e-10, rtol=1e-10)
    actual = kots.jacobian_transpose_mul(target, cotangent)
    assert actual.shape == expected_vjp.shape
    np.testing.assert_allclose(actual, expected_vjp, atol=1e-10, rtol=1e-10)

    pairs = []
    row = 0
    for state, part in zip(states, parts):
        width = part.shape[-2]
        rhs = cotangent[..., row:row + width, :] if matrix else cotangent[..., row:row + width]
        pairs.append((state, rhs))
        row += width
    actual = kots.jacobian_transpose_mul_many(pairs)
    assert actual.shape == expected_vjp.shape
    np.testing.assert_allclose(actual, expected_vjp, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("backend", ["numpy", "rust"])
@pytest.mark.parametrize("batch_shape", [(), (2, 3)])
@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float32, np.float64])
def test_mixed_energy_vjp_promotes_rhs_dtype(backend, batch_shape, matrix, dtype):
    if backend == "rust":
        pytest.importorskip("robokots._rust", exc_type=ImportError)
    kots = Kots.from_urdf_file(str(MODEL), order=3, backend=backend)
    motion = np.random.default_rng(763).normal(scale=.2, size=batch_shape + (kots.dof() * 3,))
    kots.import_motions(motion)
    kots.dynamics(gravity=[.2, -.3, -9.81])
    states = [StateType("total_body", "total_body", "kinetic_energy"),
              StateType("joint", "a_shoulder", "torque")]
    jacobian = kots.jacobian(states)
    shape = batch_shape + (2,) + ((2,) if matrix else ())
    rhs = np.arange(1, np.prod(shape) + 1, dtype=dtype).reshape(shape)
    transpose = jacobian.swapaxes(-1, -2)
    expected = transpose @ rhs if matrix else (transpose @ rhs[..., None])[..., 0]
    actual = kots.jacobian_transpose_mul(states, rhs)
    assert np.issubdtype(actual.dtype, np.floating)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
