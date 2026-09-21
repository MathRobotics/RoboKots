"""High-order spatial recurrence and shared gravity transport regressions."""
from pathlib import Path

import numpy as np
import pytest

from robokots.kots import Kots, StateType
from robokots.core.state.spec import keys_kinematics

MODEL = Path(__file__).resolve().parents[1] / 'test_model/branched_fixed.urdf'


@pytest.fixture
def oblique_model(tmp_path):
    pytest.importorskip('robokots._rust')
    path = tmp_path / 'oblique.urdf'
    path.write_text(MODEL.read_text().replace('xyz="1 0 0"', 'xyz="1 2 -3"'))
    return path


def key(family, n):
    return family if n == 0 else f'{family}_diff{n}'


@pytest.mark.parametrize('order', [4, 5, 6, 8])
@pytest.mark.parametrize('gravity', [[0, 0, 0], [.3, -.4, -9.81]])
def test_high_order_states_match_numpy_across_workspace_updates(oblique_model, order, gravity):
    rust = Kots.from_urdf_file(str(oblique_model), order=order)
    numpy = Kots.from_urdf_file(str(oblique_model), order=order)
    rng = np.random.default_rng(1942)
    motion = rng.normal(scale=.3, size=(2, 1, rust.dof() * order))
    states = []
    for name in rust.link_name_list():
        states.extend(StateType('link', name, dt) for dt in ('frame', *keys_kinematics[3:3+order-1]))
    for owner, names in [('link', rust.link_name_list()), ('joint', rust.joint_name_list())]:
        for name in names:
            for family, count in [('momentum', order-1), ('force', order-2)]:
                for n in range(count):
                    states.extend(StateType(owner, name, key(family, n), frame) for frame in (None, 'world'))
    states.extend(StateType('total_joint', 'total_joint', key('torque', n)) for n in range(order-2))
    # Reuse the same raw workspace, changing motion and gravity, then zero all
    # derivatives. Fixed joints must not inherit rotation coefficients.
    for x, g in [(motion, gravity), (motion * -.7, [0, 0, 0]), (np.zeros_like(motion), gravity)]:
        for kots, backend in [(rust, 'rust'), (numpy, 'numpy')]:
            kots.import_motions(x)
            kots.dynamics(backend=backend, gravity=g)
        for state in states:
            actual = rust.state_info(state)
            expected = numpy.state_info(state)
            np.testing.assert_allclose(actual, expected, atol=3e-9, rtol=3e-9, err_msg=str(state))


@pytest.mark.parametrize('order', [4, 6, 8])
def test_high_order_jvp_vjp_match_numpy_and_central_difference(oblique_model, order):
    rust = Kots.from_urdf_file(str(oblique_model), order=order)
    reference = Kots.from_urdf_file(str(oblique_model), order=order)
    rng = np.random.default_rng(474)
    motion = rng.normal(scale=.2, size=(2, 1, rust.dof() * order))
    gravity = [.3, -.4, -9.81]
    states = [
        StateType('link', 'a_tip', key('force', order-3), 'world'),
        StateType('joint', 'a_shoulder', key('force', order-3), 'world'),
        StateType('joint', 'b_payload_fixed', key('momentum', order-2)),
        StateType('total_joint', 'total_joint', key('torque', order-3)),
    ]
    for kots, backend in [(rust, 'rust'), (reference, 'numpy')]:
        kots.import_motions(motion)
        kots.dynamics(backend=backend, gravity=gravity)
    expected = reference.jacobian(states)
    np.testing.assert_allclose(rust.jacobian(states), expected, atol=5e-9, rtol=5e-9)
    directions = rng.normal(size=motion.shape + (2,))
    weights = rng.normal(size=expected.shape[:-1] + (2,))
    jvp = rust.jacobian_mul(states, directions)
    vjp = rust.jacobian_transpose_mul(states, weights)
    np.testing.assert_allclose(jvp, expected @ directions, atol=1e-8, rtol=5e-9)
    np.testing.assert_allclose(vjp, np.swapaxes(expected, -1, -2) @ weights, atol=1e-8, rtol=5e-9)

    def values(x):
        reference.import_motions(x)
        reference.dynamics(backend='numpy', gravity=gravity)
        return np.concatenate([np.asarray(reference.state_info(st)).reshape((2, 1, -1)) for st in reference._state_type_list(states)], axis=-1)
    eps = 1e-6
    finite_difference = (values(motion + eps * directions[..., 0]) - values(motion - eps * directions[..., 0])) / (2 * eps)
    np.testing.assert_allclose(jvp[..., 0], finite_difference, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(
        np.sum(directions[..., 0] * vjp[..., 0], axis=-1),
        np.sum(finite_difference * weights[..., 0], axis=-1), atol=3e-6, rtol=1e-6,
    )
