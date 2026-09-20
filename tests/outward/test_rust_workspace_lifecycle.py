"""Shared CMTM state and lazy dynamics storage lifecycle."""
from pathlib import Path

import numpy as np
import pytest

from robokots.kots import Kots, StateType

MODEL = Path(__file__).resolve().parents[1] / 'test_model/branched_fixed.urdf'


@pytest.mark.parametrize('order', [1, 2, 3, 5, 8])
@pytest.mark.parametrize('batch_shape', [(), (2, 1)])
def test_shared_workspace_switches_and_allocations(order, batch_shape):
    pytest.importorskip('robokots._rust')
    k = Kots.from_urdf_file(str(MODEL), order=order)
    ref = Kots.from_urdf_file(str(MODEL), order=order)
    robot = k._rust_compiled_robot()
    batch = int(np.prod(batch_shape)) if batch_shape else 1
    raw = robot.create_batch_outward_data(order, batch) if batch_shape else robot.create_outward_data(order)
    kin_bytes, dyn_bytes = raw._workspace_buffer_bytes()
    assert kin_bytes > 0 and dyn_bytes == 0
    rng = np.random.default_rng(715)
    motion = rng.normal(scale=.2, size=batch_shape + (k.dof()*order,))

    def flat(x):
        return x.reshape(batch, -1) if batch_shape else x

    def check_kinematics(x, dynamics=False, gravity=None):
        ref.import_motions(x)
        if dynamics:
            ref.dynamics(backend='numpy', gravity=gravity)
        else:
            ref.kinematics(backend='numpy')
        for owner, names in [('link', k.link_name_list()), ('joint', k.joint_name_list())]:
            for idx, name in enumerate(names):
                actual = getattr(raw, owner+'_mat')(idx)
                expected = ref.state_info(StateType(owner, name, 'frame'))
                if hasattr(expected, 'mat'): expected = expected.mat()
                np.testing.assert_allclose(actual.reshape(np.shape(expected)), expected, atol=2e-9, rtol=2e-9)
                for key_order in range(2, order+1):
                    # Read series through the common CMTM representation to avoid aliases.
                    source = ref.outward_state_.cmtm(owner, name, order)
                    expected = np.asarray(source.vecs())[..., key_order-2, :]
                    actual = getattr(raw, owner+'_vec')(idx, key_order)
                    np.testing.assert_allclose(actual.reshape(expected.shape), expected, atol=2e-9, rtol=2e-9)

    raw.compute_kinematics(flat(motion))
    assert raw._workspace_buffer_bytes() == (kin_bytes, 0)
    check_kinematics(motion)
    if order == 1:
        with pytest.raises(ValueError): raw.compute_dynamics(flat(motion))
        assert raw._workspace_buffer_bytes() == (kin_bytes, 0)
        return
    allocated = None
    for gravity in (np.zeros(3), np.array([.3, -.4, -9.81]), np.zeros(3)):
        raw.compute_dynamics(flat(motion), gravity)
        current = raw._workspace_buffer_bytes()
        assert current[0] == kin_bytes and current[1] > 0
        if allocated is not None: assert current == allocated
        allocated = current
        check_kinematics(motion, True, gravity)
        if order >= 3:
            for joint, name in enumerate(k.joint_name_list()):
                for n in range(order-2):
                    expected = np.asarray(ref.state_info(StateType('joint', name, 'force' if n == 0 else f'force_diff{n}')))
                    np.testing.assert_allclose(raw.joint_force(joint, n+1).reshape(expected.shape), expected, atol=2e-9, rtol=2e-9)
        motion = motion * -.7
        raw.compute_kinematics(flat(motion))
        assert raw._workspace_buffer_bytes() == allocated
        check_kinematics(motion)
        with pytest.raises(ValueError, match='compute_dynamics'):
            raw.joint_momentum(0, 1)


@pytest.mark.parametrize('batched', [False, True])
def test_minimal_dynamics_allocates_on_first_use(batched):
    pytest.importorskip('robokots._rust')
    k = Kots.from_urdf_file(str(MODEL), order=3)
    robot = k._rust_compiled_robot()
    data = robot.create_batch_outward_data(3, 2) if batched else robot.create_outward_data(3)
    x = np.zeros((2, k.dof()*3) if batched else k.dof()*3)
    assert data._workspace_buffer_bytes()[1] == 0
    data.compute_dynamics_minimal(x)
    allocated = data._workspace_buffer_bytes()
    assert allocated[1] > 0
    data.compute_kinematics(x)
    data.compute_dynamics(x)
    assert data._workspace_buffer_bytes() == allocated
    assert np.all(np.isfinite(data.joint_torque(0, 1)))
