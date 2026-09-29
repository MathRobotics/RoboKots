from pathlib import Path

import numpy as np
import pytest

from robokots.kots import Kots, StateType
from robokots.outward.rust.data import RustBatchOutwardState


MODEL = Path(__file__).resolve().parents[1] / "test_model" / "branched_fixed.urdf"


def _make_kots(order=4):
  pytest.importorskip("robokots._rust")
  kots = Kots.from_urdf_file(str(MODEL), order=order)
  motion = np.random.default_rng(81).normal(scale=0.2, size=(2, 1, kots.dof() * order))
  kots.import_motions(motion)
  kots.dynamics(backend="rust", gravity=[0.2, -0.3, -9.81])
  return kots, motion


def test_rust_numerical_torque_uses_one_shared_value_evaluation(monkeypatch):
  kots, _ = _make_kots(order=4)
  state = StateType("total_joint", "total_joint", "torque_diff1")
  calls = []
  original = RustBatchOutwardState.compute_dynamics

  def counted(workspace, motion, gravity=None):
    calls.append(np.asarray(motion).shape)
    return original(workspace, motion, gravity)

  monkeypatch.setattr(RustBatchOutwardState, "compute_dynamics", counted)
  numerical = kots.jacobian(state, numerical=True, eps=1e-7)
  analytic = kots.jacobian(state)

  assert calls == [(2 * 1 * 2 * kots.dof() * 4, kots.dof() * 4)]
  np.testing.assert_allclose(numerical, analytic, atol=2e-6, rtol=2e-6)


def test_rust_numerical_batch_selection_and_products_reuse_matrix():
  kots, _ = _make_kots(order=4)
  states = [
    StateType("joint", "a_elbow", "torque_diff1"),
    StateType("joint", "a_shoulder", "torque_diff1"),
  ]
  numerical = kots.jacobian(states, numerical=True, eps=1e-7)
  parts = kots.jacobian(states, numerical=True, list_output=True, eps=1e-7)
  np.testing.assert_allclose(np.concatenate(parts, axis=-2), numerical)

  rng = np.random.default_rng(82)
  rhs = rng.normal(size=(2, 1, kots.dof() * 4, 3))
  lhs = rng.normal(size=(2, 1, numerical.shape[-2], 3))
  np.testing.assert_allclose(
    kots.jacobian_mul(states, rhs, numerical=True, eps=1e-7),
    numerical @ rhs,
    atol=2e-6,
    rtol=2e-6,
  )
  np.testing.assert_allclose(
    kots.jacobian_transpose_mul(states, lhs, numerical=True, eps=1e-7),
    np.swapaxes(numerical, -1, -2) @ lhs,
    atol=2e-6,
    rtol=2e-6,
  )


def test_rust_numerical_preserves_motion_and_gravity():
  kots, motion = _make_kots(order=3)
  gravity = kots.gravity_.copy()
  kots.jacobian(
    StateType("total_joint", "total_joint", "torque"),
    numerical=True,
    eps=1e-6,
  )
  np.testing.assert_array_equal(kots.motion(3), motion)
  np.testing.assert_array_equal(kots.gravity_, gravity)
