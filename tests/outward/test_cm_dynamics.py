"""Independent derivative and state checks for coefficient-only dynamics."""
import math
from pathlib import Path

import numpy as np
import pytest
from mathrobo import CMVector

from robokots.core.kernels.dynamics import link_force_cmvec, link_momentum_cmvec
from robokots.kots import Kots, StateType


MODEL = Path(__file__).resolve().parents[1] / "test_model" / "branched_fixed.urdf"


@pytest.mark.parametrize("force_order", [0, 1, 4, 10])
@pytest.mark.parametrize("batch", [(), (2, 1)])
def test_force_matches_ordinary_product_rule(force_order, batch):
    rng = np.random.default_rng(41)
    velocity = rng.normal(scale=.2, size=batch + (force_order + 1, 6))
    momentum = rng.normal(scale=.2, size=batch + (force_order + 1, 6))
    expected = momentum[..., 1:, :].copy()
    # Leibniz rule for ordinary derivatives, independently of CMTM hats.
    for k in range(force_order):
        for i in range(k + 1):
            v, p = velocity[..., i, :], momentum[..., k-i, :]
            dual_cross = np.concatenate([
                np.cross(v[..., :3], p[..., :3]) + np.cross(v[..., 3:], p[..., 3:]),
                np.cross(v[..., :3], p[..., 3:]),
            ], axis=-1)
            expected[..., k, :] += math.comb(k, i) * dual_cross
    result = link_force_cmvec(CMVector(velocity), CMVector(momentum))
    assert result.vecs().shape == batch + (force_order, 6)
    np.testing.assert_allclose(result.vecs(), expected, rtol=1e-11, atol=1e-11)


def test_force_broadcasts_velocity_and_momentum_batches():
    rng = np.random.default_rng(42)
    v = rng.normal(size=(2, 1, 5, 6))
    p = rng.normal(size=(1, 3, 5, 6))
    result = link_force_cmvec(CMVector(v), CMVector(p)).vecs()
    assert result.shape == (2, 3, 4, 6)
    for i in range(2):
        for j in range(3):
            expected = link_force_cmvec(CMVector(v[i, 0]), CMVector(p[0, j])).vecs()
            np.testing.assert_allclose(result[i, j], expected)


def test_momentum_and_force_do_not_restore_ordinary_values(monkeypatch):
    velocity = np.arange(60., dtype=float).reshape(2, 5, 6) / 10
    inertia = np.diag(np.arange(1., 7.))
    series = CMVector(velocity)
    expected = np.einsum("ij,...kj->...ki", inertia, velocity)
    factors = np.array([1, 1, 2, 6, 24])[:, None]

    def forbidden(*args):
        raise AssertionError("internal dynamics must use CM coefficients")

    monkeypatch.setattr(CMVector, "vecs", forbidden)
    monkeypatch.setattr(CMVector, "vec", forbidden)
    momentum = link_momentum_cmvec(inertia, series)
    force = link_force_cmvec(series, momentum)
    np.testing.assert_allclose(momentum.cm_vecs(), expected / factors)
    assert force.cm_vecs().shape == (2, 4, 6)


@pytest.mark.parametrize("gravity", [(0., 0., 0.), (.2, -.3, -9.81)])
def test_state_keeps_all_dynamics_in_cm(monkeypatch, gravity):
    kots = Kots.from_urdf_file(str(MODEL), order=5, backend="numpy")
    motion = np.random.default_rng(43).normal(scale=.2, size=(kots.dof(), 5))
    original = CMVector.vecs
    reads = []

    def tracked(self):
        reads.append(self._n)
        return original(self)

    def forbidden(*args):
        raise AssertionError("internal dynamics must not flatten ordinary derivatives")

    kots.import_motion_array(motion)
    monkeypatch.setattr(CMVector, "vecs", tracked)
    monkeypatch.setattr(CMVector, "vec", forbidden)
    kots.dynamics(backend="numpy", gravity=gravity)
    assert reads == []
    assert all(order == 3 for order in reads)


@pytest.mark.parametrize("derivative", [0, 2])
@pytest.mark.parametrize("batch", [(), (2, 1)])
@pytest.mark.parametrize("gravity", [(0., 0., 0.), (.2, -.3, -9.81)])
def test_force_and_torque_jacobians_match_central_differences(derivative, batch, gravity):
    order = derivative + 4  # Also exercise selecting fewer rows than stored.
    kots = Kots.from_urdf_file(str(MODEL), order=order, backend="numpy")
    motion = np.random.default_rng(44).normal(scale=.2, size=batch + (kots.dof(), order))
    force = "force" if derivative == 0 else f"force_diff{derivative}"
    torque = "torque" if derivative == 0 else f"torque_diff{derivative}"
    states = [
        StateType("link", "a_tip", force),
        StateType("link", "a_tip", force, "world"),
        StateType("joint", "a_shoulder", force, "world"),
        StateType("joint", "a_shoulder", f"momentum_diff{derivative + 1}", "world"),
        StateType("joint", "a_shoulder", torque),
    ]

    def values(current):
        kots.import_motion_array(current)
        kots.dynamics(backend="numpy", gravity=gravity)
        return kots.state_info_list(states)

    values(motion)
    analytic = kots.jacobian(states)
    columns = []
    for owner in range(kots.dof()):
        for k in range(StateType.max_time_order(states)):
            step = np.zeros_like(motion)
            step[..., owner, k] = 1e-6
            columns.append((values(motion + step) - values(motion - step)) / 2e-6)
    np.testing.assert_allclose(analytic, np.stack(columns, axis=-1), rtol=3e-7, atol=3e-7)
