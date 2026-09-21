"""Selection workspace invalidation and kinematics-only derivative coverage."""
from pathlib import Path

import numpy as np
import pytest
from robokots.kots import Kots, StateType
from robokots import outward

MODEL=Path(__file__).resolve().parents[1]/'test_model/branched_fixed.urdf'


@pytest.mark.parametrize('order,shape',[(1,()),(1,(2,)),(2,()),(2,(2,1)),(5,()),(5,(2,))])
def test_kinematics_only_selected_rust(order,shape,monkeypatch):
    pytest.importorskip('robokots._rust')
    rng=np.random.default_rng(527)
    k=Kots.from_urdf_file(str(MODEL),order=order)
    ref=Kots.from_urdf_file(str(MODEL),order=order)
    x=rng.normal(size=shape+(k.dof()*order,))*.25
    states=[StateType('link','a_tip','frame','world'),StateType('joint','a_elbow','rot','local'),StateType('link','b_payload','pos','world')]
    if order>1:
        key=('vel','acc','jerk','snap')[order-2]
        states+=[StateType('link','a_tip',key,'world'),StateType('joint','a_elbow','vel','world')]
    for obj,backend in [(k,'rust'),(ref,'numpy')]:
        obj.import_motions(x);obj.kinematics(backend=backend)
    expected=ref.jacobian(states)
    def forbidden(*a,**kw):raise AssertionError('Python/dense fallback')
    for name in ('outward_jacobian','outward_jacobian_matvec','outward_jacobian_matmul_rhs','outward_jacobian_transpose_matvec'):
        monkeypatch.setattr(outward,name,forbidden)
    actual=k.jacobian(states)
    np.testing.assert_allclose(actual,expected,atol=1e-11)
    workspace=k._rust_selected_workspace_[2]
    assert workspace.cache_info()==(int(np.prod(shape)) if shape else 1,0,int(np.prod(shape)) if shape else 1)
    v=rng.normal(size=shape+(actual.shape[-1],2));w=rng.normal(size=shape+(actual.shape[-2],2))
    monkeypatch.setattr(k,'_jacobian_from_state',forbidden)
    np.testing.assert_allclose(k.jacobian_mul(states,v),actual@v,atol=1e-11)
    np.testing.assert_allclose(k.jacobian_transpose_mul(states,w),np.swapaxes(actual,-1,-2)@w,atol=1e-11)
    assert workspace.cache_info()[1]==0
    assert workspace.cache_info()[0]==(int(np.prod(shape)) if shape else 1)


def test_workspace_reuses_and_invalidates_primal():
    pytest.importorskip('robokots._rust')
    k=Kots.from_urdf_file(str(MODEL),order=4)
    robot=k._rust_compiled_robot()
    ws=robot.create_selected_workspace(4)
    rng=np.random.default_rng(318)
    x=rng.normal(size=(2,k.dof()*4))*.2
    v=rng.normal(size=(2,k.dof()*4,2))
    outputs=[(0,2,3,1,True),(0,2,1,1,False)]
    gravity=np.array([.2,-.3,-9.81])
    expected=lambda:np.asarray(robot.dynamics_selected_tangent_batch(x,v,outputs,2,gravity))
    first=np.asarray(ws.apply(x,v,outputs,gravity))
    first_saved=first.copy()
    np.testing.assert_allclose(first,expected(),atol=1e-11)
    assert ws.cache_info()==(0,2,2)
    np.testing.assert_allclose(ws.apply(x,v,outputs,gravity),first,atol=1e-11)
    assert ws.cache_info()==(0,2,2)
    # Different cotangents and RHS widths reuse the same primal.
    w=rng.normal(size=(2,12,3))
    np.testing.assert_allclose(ws.apply(x,w,outputs,gravity,transpose=True),robot.dynamics_selected_transpose_batch(x,w,outputs,2,gravity),atol=1e-11)
    assert ws.cache_info()==(0,2,2)
    x[1,0]+=.15
    np.testing.assert_allclose(ws.apply(x,v,outputs,gravity),expected(),atol=1e-11)
    assert ws.cache_info()==(0,3,2)
    gravity[0]+=.7
    np.testing.assert_allclose(ws.apply(x,v,outputs,gravity),expected(),atol=1e-11)
    assert ws.cache_info()==(0,5,2)
    # Switching to pure kinematics frees dynamics state and computes no dynamics.
    kin=[outputs[0]]
    ws.apply(x,v,kin,gravity)
    assert ws.cache_info()==(2,5,2)
    ws.apply(x[:1],v[:1],kin,gravity)
    assert ws.cache_info()==(3,5,1)
    # Invalid requests must not poison the last valid cache.
    with pytest.raises(ValueError):ws.apply(x,v[:1],outputs,gravity)
    assert ws.cache_info()==(3,5,1)
    np.testing.assert_array_equal(first,first_saved)


def test_workspace_owner_changes_with_model_and_order():
    k=Kots.from_urdf_file(str(MODEL),order=4)
    k.kinematics(backend='rust')
    k.jacobian(StateType('link','a_tip','vel','world'))
    previous=k._rust_selected_workspace_
    k.jacobian(StateType('link','a_tip','acc','world'))
    assert k._rust_selected_workspace_[1]==3
    assert k._rust_selected_workspace_[2] is not previous[2]
    previous=k._rust_selected_workspace_
    k._rust_compiled_robot_=None  # same invalidation used for model replacement
    k.jacobian(StateType('link','a_tip','acc','world'))
    assert k._rust_selected_workspace_[0] is not previous[0]
    assert k._rust_selected_workspace_[2] is not previous[2]


def test_momentum_only_order_two():
    rust=Kots.from_urdf_file(str(MODEL),order=3)
    numpy=Kots.from_urdf_file(str(MODEL),order=3)
    x=np.random.default_rng(11).normal(size=rust.dof()*3)*.2
    states=[StateType('link','a_tip','momentum','world'),StateType('joint','a_elbow','momentum')]
    for k,backend in [(rust,'rust'),(numpy,'numpy')]:
        k.import_motions(x);k.dynamics(backend=backend)
    jac=numpy.jacobian(states)
    np.testing.assert_allclose(rust.jacobian(states),jac,atol=1e-11)
    v=np.ones(jac.shape[-1]);w=np.ones(jac.shape[-2])
    np.testing.assert_allclose(rust.jacobian_mul(states,v),jac@v,atol=1e-11)
    np.testing.assert_allclose(rust.jacobian_transpose_mul(states,w),jac.T@w,atol=1e-11)
