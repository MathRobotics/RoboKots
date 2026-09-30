"""Every dynamics derivative uses its actual minimum motion order, without padding."""
from pathlib import Path

import numpy as np
import pytest

from robokots import outward
from robokots.kots import Kots, StateType


MODEL = Path(__file__).resolve().parents[1] / "test_model" / "branched_fixed.urdf"


@pytest.mark.parametrize("derivative", range(5))
@pytest.mark.parametrize("family", ["momentum", "force", "torque"])
@pytest.mark.parametrize("shape", [(), (2, 1)])
def test_minimum_dynamics_order(monkeypatch, derivative, family, shape):
    pytest.importorskip("robokots._rust")
    order = derivative + (2 if family == "momentum" else 3)
    key = family if derivative == 0 else f"{family}_diff{derivative}"
    rust = Kots.from_urdf_file(str(MODEL), order=order)
    reference = Kots.from_urdf_file(str(MODEL), order=order)
    rng = np.random.default_rng(289)
    motion = rng.normal(scale=.2, size=shape + (rust.dof() * order,))
    # The batched case also exercises a zero pose with nonzero higher motions.
    if shape:
        motion[0, ..., 0::order] = 0
    gravity = [.3, -.4, -9.81]
    if family == "torque":
        states = [StateType("total_joint", "total_joint", key)]
    else:
        states = [StateType(owner, name, key, frame)
                  for owner, names in (("link", rust.link_name_list()),
                                       ("joint", rust.joint_name_list()))
                  for name in names for frame in ("local", "world")]
    for kots, backend in ((reference, "numpy"), (rust, "rust")):
        kots.import_motions(motion)
        kots.dynamics(backend=backend, gravity=gravity)
    expected = reference.jacobian(states)
    assert expected.shape[-1] == rust.dof() * order

    def forbidden(*args, **kwargs):
        raise AssertionError("Rust derivatives must not fall back to Python or dense products")

    for name in ("outward_jacobian", "outward_jacobian_matvec",
                 "outward_jacobian_matmul_rhs", "outward_jacobian_transpose_matvec"):
        monkeypatch.setattr(outward, name, forbidden)
    actual = rust.jacobian(states)
    np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-9)
    parts = rust.jacobian(states, list_output=True)
    np.testing.assert_allclose(np.concatenate(parts, axis=-2), expected, atol=2e-8, rtol=2e-9)
    # Central differences run on shared Rust values, independently of tangents.
    numerical = rust.jacobian(states, numerical=True, eps=1e-6)
    np.testing.assert_allclose(numerical, expected, atol=3e-6, rtol=2e-6)
    monkeypatch.setattr(rust, "_jacobian_from_state", forbidden)
    directions = rng.normal(size=shape + (expected.shape[-1], 2))
    weights = rng.normal(size=shape + (expected.shape[-2], 2))
    jvp = rust.jacobian_mul(states, directions)
    vjp = rust.jacobian_transpose_mul(states, weights)
    np.testing.assert_allclose(jvp, expected @ directions, atol=3e-8, rtol=3e-9)
    np.testing.assert_allclose(vjp, np.swapaxes(expected, -1, -2) @ weights, atol=3e-8, rtol=3e-9)
    np.testing.assert_allclose(rust.jacobian_mul(states, directions[..., 0]), jvp[..., 0], atol=3e-8, rtol=3e-9)
    np.testing.assert_allclose(rust.jacobian_transpose_mul(states, weights[..., 0]), vjp[..., 0], atol=3e-8, rtol=3e-9)

    # Independent NumPy value differences include the moving-frame terms.
    def value(x):
        reference.import_motions(x)
        reference.dynamics(backend="numpy", gravity=gravity)
        return np.concatenate([np.asarray(reference.state_info(st)).reshape(shape + (-1,))
                               for st in reference._state_type_list(states)], axis=-1)

    h = 1e-6
    difference = (value(motion + h * directions[..., 0])
                  - value(motion - h * directions[..., 0])) / (2 * h)
    np.testing.assert_allclose(jvp[..., 0], difference, atol=3e-6, rtol=2e-6)
