"""No ordinary-series cache or round trip while selecting state samples."""
from pathlib import Path
import numpy as np
import pytest
from mathrobo import CMTM, CMVector
from robokots.kots import Kots
from robokots.outward.data import state_sample
from robokots.outward.rust.state import build_dynamics_outward_state_rust

MODEL = Path(__file__).resolve().parents[1] / 'test_model/branched_fixed.urdf'


@pytest.mark.parametrize('backend', ['numpy', 'rust', 'array'])
def test_sample_selection_shares_cm_without_ordinary_conversion(backend, monkeypatch):
    k = Kots.from_urdf_file(str(MODEL), order=6)
    x = np.random.default_rng(14).normal(scale=.2, size=(2, 1, k.dof()*6))
    k.import_motions(x)
    if backend == 'array':
        state = build_dynamics_outward_state_rust(k.robot_, x, dynamics_order=4, gravity=[.2,-.3,-9.81])
    else:
        k.dynamics(backend=backend, gravity=[.2,-.3,-9.81])
        state = k.outward_state_
    def forbidden(*args, **kwargs):
        raise AssertionError('sample selection reconstructed an ordinary derivative series')
    with monkeypatch.context() as patch:
        patch.setattr(CMVector, 'vecs', forbidden)
        patch.setattr(CMTM, 'vecs', forbidden)
        sample = state_sample(k.robot_, state, (1, 0))
    for owner, names in [('link', k.link_name_list()), ('joint', k.joint_name_list())]:
        for name in names:
            for family in ['momentum', 'force']:
                expected = state.cmvec(owner, name, family).cm_vecs()[1, 0]
                actual = sample.cmvec(owner, name, family).cm_vecs()
                np.testing.assert_array_equal(actual, expected)
                assert np.shares_memory(actual, state.cmvec(owner, name, family).cm_vecs())
    for joint in k.robot_.joints:
        if joint.dof:
            np.testing.assert_allclose(sample.cmvec('joint',joint.name,'torque').cm_vecs(),
                state.cmvec('joint',joint.name,'torque').cm_vecs()[1,0])
    if backend == 'array':
        assert 'link_vecs' not in state.__dict__
        assert 'link_momentum_array' not in state.__dict__
        assert 'joint_torque_array' not in state.__dict__
        ordinary = state.link_vecs
        assert not np.shares_memory(ordinary, state._series_cm['link_vecs'].cm_vecs())


def test_native_bulk_cm_option_preserves_ordinary_default():
    k = Kots.from_urdf_file(str(MODEL), order=6)
    robot = k._rust_compiled_robot()
    x = np.random.default_rng(18).normal(scale=.2, size=k.dof()*6)
    gravity = np.asarray([.2,-.3,-9.81])
    ordinary = robot.dynamics_outward_cmtm(x, 4, gravity)
    coefficients = robot.dynamics_outward_cmtm(x, 4, gravity, cm_coefficients=True)
    for i in [0, 2]:
        np.testing.assert_array_equal(coefficients[i], ordinary[i])
    for i in [1,3,4,5,6,7,8]:
        np.testing.assert_allclose(CMVector.set_cmvecs(coefficients[i]).vecs(),ordinary[i],rtol=1e-15,atol=1e-15)



def test_jax_local_motion_coefficients_keep_autodiff():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    from robokots.core.kernels.kinematics_jax import local_cmtm
    select = jnp.asarray([[0.], [0.], [1.], [0.], [0.], [0.]])
    motion = jnp.asarray([.1, .2, .3, .4, .5])
    def coefficients(x):
        return local_cmtm(select, x, order=5).cmvecs().cm_vecs()
    expected = np.zeros((4, 6))
    expected[:, 2] = np.asarray(motion[1:]) / np.asarray([1., 1., 2., 6.])
    np.testing.assert_allclose(coefficients(motion), expected, rtol=1e-6, atol=1e-8)
    gradient = jax.grad(lambda x: coefficients(x).sum())(motion)
    np.testing.assert_allclose(gradient, [0., 1., 1., .5, 1/6], rtol=1e-6, atol=1e-8)
