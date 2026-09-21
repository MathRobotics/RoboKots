"""Independent state differences for selected spatial/dynamics derivatives."""
from pathlib import Path

import numpy as np
import pytest

from robokots.kots import Kots, StateType
from robokots import outward

MODEL = Path(__file__).resolve().parents[1] / 'test_model' / 'branched_fixed.urdf'


def vee(a):
    return np.array([a[2,1]-a[1,2], a[0,2]-a[2,0], a[1,0]-a[0,1]])/2


def selections(order):
    spatial = ('vel', 'acc', 'jerk', 'snap', 'crackle', 'pop')
    return [StateType('joint','b_payload_fixed','acc'),
            StateType('joint','a_shoulder','acc','world'),
            StateType('link','b_payload',spatial[order-2],'world'),
            StateType('joint','a_elbow','vel'),
            StateType('link','a_tip','force','world'),
            StateType('joint','a_elbow',f'force_diff{order-3}' if order>3 else 'force','world'),
            StateType('joint','b_shoulder','momentum','world'),
            StateType('joint','a_shoulder','torque'),
            StateType('link','a_tip','vel'),
            *[StateType(owner,name,kind,frame)
              for owner,name in [('link','a_tip'),('joint','a_shoulder')]
              for frame in ['local','world'] for kind in ['pos','rot','frame']]]


def numerical_columns(k, x, states, gravity):
    def values(motion):
        k.import_motions(motion); k.dynamics(backend='numpy', gravity=gravity)
        vals = []
        for s in states:
            if s.data_type in ('pos','rot','frame'):
                vals.append(np.asarray(k.outward_state_.cmtm(s.owner_type,s.owner_name,1).elem_mat()))
            else:
                vals.append(np.asarray(k.state_info(s)))
        return vals
    base = values(x)
    h = 1e-6
    result = []
    for e in np.eye(x.size):
        plus, minus = values(x+h*e), values(x-h*e)
        parts = []
        for s,t,a,b in zip(states,base,plus,minus):
            d = (a-b)/(2*h)
            if s.data_type in ('pos','rot','frame'):
                local = np.linalg.inv(t) @ d
                tangent = d @ np.linalg.inv(t) if s.frame_name == 'world' else local
                if s.data_type == 'pos':
                    part = d[:3,3] if s.frame_name == 'world' else local[:3,3]
                elif s.data_type == 'rot':
                    part = vee(tangent)
                else:
                    part = np.r_[vee(tangent), tangent[:3,3]]
            else:
                part = d
            parts.append(part.reshape(-1))
        result.append(np.concatenate(parts))
    return np.stack(result,axis=-1)


@pytest.mark.parametrize('order,shape', [(3,()),(5,(2,1)),(7,())])
def test_selected_spatial_dense_and_direct_products(order,shape,monkeypatch):
    pytest.importorskip('robokots._rust')
    ref = Kots.from_urdf_file(str(MODEL),order=order)
    rng=np.random.default_rng(642)
    x=rng.normal(scale=.25,size=shape+(ref.dof()*order,))
    if order == 7:
        x[..., ::order] = 0
    states=selections(order)
    states += [states[2], states[0]]
    gravity=[.2,-.3,-9.81]
    expected=np.stack([numerical_columns(ref, sample, states, gravity)
                       for sample in x.reshape(-1,x.shape[-1])]).reshape(shape+(-1,x.shape[-1]))
    def forbidden(*args,**kwargs):
        raise AssertionError('dense/Python fallback')
    for backend in ('numpy','rust'):
        k=Kots.from_urdf_file(str(MODEL),order=order)
        k.import_motions(x);k.dynamics(backend=backend,gravity=gravity)
        actual=k.jacobian(states)
        np.testing.assert_allclose(actual,expected,atol=2e-7,rtol=2e-7)
        np.testing.assert_allclose(np.concatenate(k.jacobian(states,list_output=True),axis=-2),actual,atol=1e-11)
        v=rng.normal(size=shape+(x.shape[-1],2));w=rng.normal(size=shape+(actual.shape[-2],2))
        with monkeypatch.context() as patch:
            patch.setattr(k,'_jacobian_from_state',forbidden)
            if backend=='rust':
                for name in ('outward_jacobian','outward_jacobian_matvec','outward_jacobian_matmul_rhs','outward_jacobian_transpose_matvec'):
                    patch.setattr(outward,name,forbidden)
            np.testing.assert_allclose(k.jacobian_mul(states,v),actual@v,atol=1e-10)
            np.testing.assert_allclose(k.jacobian_transpose_mul(states,w),np.swapaxes(actual,-1,-2)@w,atol=1e-10)
            np.testing.assert_allclose(k.jacobian_mul(states,v[...,0]),(actual@v)[...,0],atol=1e-10)
            np.testing.assert_allclose(k.jacobian_transpose_mul(states,w[...,0]),(np.swapaxes(actual,-1,-2)@w)[...,0],atol=1e-10)


def test_numpy_joint_spatial_selection_order_and_kinematics_only():
    k=Kots.from_urdf_file(str(MODEL),order=4)
    x=np.random.default_rng(94).normal(size=k.dof()*4)*.2
    k.import_motions(x);k.dynamics()
    joint=StateType('joint','a_elbow','acc')
    link=StateType('link','a_tip','vel')
    force=StateType('link','a_tip','force_diff1')
    a=k.jacobian([joint,link,force,joint])
    b=k.jacobian([force,link,joint,joint])
    np.testing.assert_allclose(a[:6],b[12:18])
    # Fixed relative joint motion stays zero although its child link moves.
    np.testing.assert_array_equal(k.jacobian(StateType('joint','b_payload_fixed','acc')),0)
    assert np.linalg.norm(k.jacobian(StateType('link','b_payload','acc'))) > 0


@pytest.mark.parametrize('frame', ['local','world'])
def test_public_pose_numerical_and_world_adapter(frame):
    k=Kots.from_urdf_file(str(MODEL),order=3)
    k.import_motions(np.random.default_rng(5).normal(size=k.dof()*3)*.3)
    k.dynamics(backend='rust')
    for owner,name in [('link','a_tip'),('joint','a_elbow')]:
        for kind in ('pos','rot','frame'):
            states=[StateType(owner,name,kind,frame),StateType('link','a_tip','force')]
            np.testing.assert_allclose(k.jacobian(states),k.jacobian(states,numerical=True),atol=2e-6,rtol=2e-6)
        s=StateType(owner,name,'acc','world')
        np.testing.assert_allclose(k.state_info(s),k.outward_state_.state_value(s))


def test_world_velocity_is_motion_adjoint_and_acceleration_time_derivative():
    from math import factorial
    k=Kots.from_urdf_file(str(MODEL),order=4)
    x=np.random.default_rng(52).normal(size=k.dof()*4)*.4
    for owner,name in [('link','a_tip'),('joint','a_elbow')]:
        def world_velocity(t):
            series=x.reshape(-1,4)
            moved=np.zeros_like(series)
            for n in range(4):
                moved[:,n]=sum(series[:,j]*t**(j-n)/factorial(j-n) for j in range(n,4))
            k.import_motions(moved.reshape(-1));k.kinematics()
            local=k.state_info(StateType(owner,name,'vel'))
            child=name if owner=='link' else k.robot_.links[k.robot_.joint(name).child_link_id].name
            mat=np.asarray(k.outward_state_.cmtm('link',child,1).elem_mat())
            angular=mat[:3,:3]@local[:3]
            linear=mat[:3,:3]@local[3:]+np.cross(mat[:3,3],angular)
            return np.r_[angular,linear]
        h=1e-5
        acc=(world_velocity(h)-world_velocity(-h))/(2*h)
        expected=world_velocity(0)
        np.testing.assert_allclose(k.state_info(StateType(owner,name,'vel','world')),expected,atol=1e-12)
        np.testing.assert_allclose(k.state_info(StateType(owner,name,'acc','world')),acc,atol=1e-8)


def test_numpy_joint_coordinates_and_pure_spatial_products(monkeypatch):
    k=Kots.from_urdf_file(str(MODEL),order=4)
    k.import_motions(np.random.default_rng(10).normal(size=k.dof()*4)*.2)
    k.dynamics()
    cases=[
        [StateType('joint','a_elbow','acc'),StateType('link','a_tip','vel'),StateType('joint','a_elbow','frame')],
        [StateType('joint','a_elbow','coord'),StateType('joint','a_elbow','veloc'),
         StateType('joint','a_elbow','jerk'),StateType('link','a_tip','acc'),StateType('link','a_tip','force_diff1')],
    ]
    for states in cases:
        jac=k.jacobian(states)
        v=np.arange(jac.shape[-1])*.1;w=np.arange(jac.shape[-2])*.2
        with monkeypatch.context() as patch:
            patch.setattr(k,'_jacobian_from_state',lambda *a,**kw:pytest.fail('dense fallback'))
            np.testing.assert_allclose(k.jacobian_mul(states,v),jac@v,atol=1e-11)
            np.testing.assert_allclose(k.jacobian_transpose_mul(states,w),jac.T@w,atol=1e-11)
        if states[0].data_type=='coord':
            index=k.robot_.joint('a_elbow').dof_index*4
            np.testing.assert_array_equal(jac[:3],np.eye(k.dof()*4)[[index,index+1,index+3]])
