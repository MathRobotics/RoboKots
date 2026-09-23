"""Direct cached RNEA products agree with independent dense and numerical paths."""
from pathlib import Path

import numpy as np
import pytest

from robokots.kots import Kots, StateType

MODEL = Path(__file__).resolve().parents[1] / "test_model/branched_fixed.urdf"


@pytest.mark.parametrize("shape", [(), (2, 3)])
@pytest.mark.parametrize("gravity", [[0., 0., 0.], [.2, -.3, -9.81]])
@pytest.mark.parametrize("matrix", [False, True])
def test_direct_rnea_products_without_dense_or_generic_reverse(shape, gravity, matrix, monkeypatch):
    k = Kots.from_urdf_file(str(MODEL), backend="rust")
    rng = np.random.default_rng(137)
    x = rng.normal(scale=.3, size=shape + (k.dof()*3,))
    k.import_motions(x)
    k.dynamics(gravity=gravity)
    # Include a subset, non-DOF ordering, and a repeated output.
    states = [StateType("joint", name, "torque") for name in ("a_elbow", "b_shoulder", "a_elbow")]
    jac = k.jacobian(states)
    extra = (3,) if matrix else ()
    direction = rng.normal(size=shape + (jac.shape[-1],) + extra)
    weight = rng.normal(size=shape + (jac.shape[-2],) + extra)
    expected_jvp = jac @ direction if matrix else (jac @ direction[..., None])[..., 0]
    expected_vjp = jac.swapaxes(-1, -2) @ weight if matrix else (jac.swapaxes(-1, -2) @ weight[..., None])[..., 0]
    def forbidden(*a, **kw):
        raise AssertionError("No dense Jacobian or CMTM reverse fallback")
    for name in ("_rust_torque_jacobian", "_rust_torque_jacobian_apply", "_rust_torque_jacobian_transpose_apply",
                 "_rust_cmtm_outward_dynamics_jacobian_transpose_apply", "_rust_cmtm_torque_jacobian_transpose_apply"):
        monkeypatch.setattr(k, name, forbidden)
    np.testing.assert_allclose(k.jacobian_mul(states, direction), expected_jvp, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(k.jacobian_transpose_mul(states, weight), expected_vjp, atol=2e-11, rtol=2e-11)
    parts = k.jacobian_mul(states, direction, list_output=True)
    np.testing.assert_allclose(np.concatenate(parts, axis=-2 if matrix else -1), expected_jvp, atol=2e-11)
    ws = k._rust_selected_workspace_[2]
    evaluations = ws.cache_info()[1]
    k.jacobian_mul(states, direction*.7)
    k.jacobian_transpose_mul(states, weight*.4)
    assert ws.cache_info()[1] == evaluations


def test_rnea_products_finite_differences_and_cache_invalidation():
    k = Kots.from_urdf_file(str(MODEL), backend="rust")
    robot = k._rust_compiled_robot()
    ws = robot.create_selected_workspace(3)
    ids = [j.id for j in k._model_metadata.joints if j.dof]
    fixed_id = next(j.id for j in k._model_metadata.joints if not j.dof)
    outputs = [(1, j, 2, 0, False) for j in ids + [fixed_id]]
    rng = np.random.default_rng(128)
    x = rng.normal(scale=.2, size=(1,k.dof()*3))
    x[:, ::3] = 0.
    gravity = np.array([.2,-.3,-9.81])
    v = rng.normal(size=(1,x.size,1))
    w = rng.normal(size=(1,len(outputs),1))
    def apply(motion, g):
        jvp = np.asarray(ws.apply(motion, v, outputs, gravity=g))
        vjp = np.asarray(ws.apply(motion, w, outputs, gravity=g, transpose=True))
        def torque(y):
            tau = np.asarray(k.inverse_dynamics(y[0,::3], y[0,1::3], y[0,2::3], gravity=g, backend="rust"))
            return np.r_[tau, 0.]
        h = 1e-6
        numeric = np.column_stack([(torque(motion+h*d)-torque(motion-h*d))/(2*h) for d in np.eye(x.size)])
        np.testing.assert_allclose(jvp[0], numeric @ v[0], atol=3e-8, rtol=2e-7)
        np.testing.assert_allclose(vjp[0], numeric.T @ w[0], atol=3e-8, rtol=2e-7)
        np.testing.assert_allclose(np.sum(w*jvp), np.sum(v*vjp), atol=1e-12)
    apply(x, gravity)
    count = ws.cache_info()[1]
    apply(x, gravity)
    assert ws.cache_info()[1] == count
    # Any of q/v/a or gravity changes invalidates the linearization.
    for col in (0, 1, 2):
        x[0,col] += .13
        apply(x, gravity)
        count += 1
        assert ws.cache_info()[1] == count
    gravity[0] += .4
    apply(x, gravity)
    assert ws.cache_info()[1] == count+1
    # The selected workspace may switch between kinematics and torque products.
    ws.apply(x, v, [(0,2,3,0,True)], gravity=gravity)
    apply(x, gravity)
    empty = ws.apply(np.empty((0,x.size)), np.empty((0,x.size,2)), outputs, gravity=gravity)
    assert empty.shape == (0,len(outputs),2)
