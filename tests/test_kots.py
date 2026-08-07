import json
import numpy as np
import pytest
from pathlib import Path

import mathrobo as mr
from robokots.kots import *
from robokots.core.target import TargetList, RobotNames
from robokots.core.models.kinematics.kinematics_jax import *
from robokots.outward.diff.outward_jax import kinematics_jax as outward_kinematics_jax

METHOD = "poly"
TEST_DIR = Path(__file__).resolve().parent
MODEL_PATH = TEST_DIR / "test_model" / "sample_robot.json"
TARGET_PATH = TEST_DIR / "target_list.json"
TARGET_LINK = "arm3"


def _make_kots(order: int = 3) -> Kots:
    return Kots.from_json_file(str(MODEL_PATH), order=order)


def test_from_urdf_file(tmp_path: Path):
    urdf = """<?xml version="1.0"?>
<robot name="urdf_robot">
  <link name="base"/>
  <link name="slider"/>
  <link name="tool"/>
  <joint name="j1" type="revolute">
    <parent link="base"/>
    <child link="slider"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
  <joint name="j2" type="prismatic">
    <parent link="slider"/>
    <child link="tool"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
  </joint>
</robot>
"""
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text(urdf, encoding="utf-8")

    kots = Kots.from_urdf_file(str(urdf_path), order=2)
    assert kots.dof() == 2
    assert "world" in kots.link_name_list()
    assert {"j1", "j2"}.issubset(set(kots.joint_name_list()))

    motion = np.array([0.2, 0.05, 0.1, 0.0], dtype=float)
    kots.import_motions(motion)
    kots.kinematics()
    frame = kots.state_info(StateType(data_type="frame", owner_type="link", owner_name="tool"))
    assert isinstance(frame, mr.SE3)


def test_state_table_is_lazy_optional():
    kots = _make_kots(order=3)
    assert kots.state_ is None

    kots.import_motions(np.zeros(kots.order() * kots.dof(), dtype=float))
    kots.kinematics()
    kots.set_state_df()

    assert kots.state_ is not None
    assert kots.state_df().shape[0] == 1
    assert not any("_link_link_force" in col for col in kots.state_df().columns)
    assert not any("_joint_joint_torque" in col for col in kots.state_df().columns)


def test_from_urdf_file_normalizes_joint_order_for_dynamics(tmp_path: Path):
    urdf = """<?xml version="1.0"?>
<robot name="misordered_tree">
  <link name="base"/>
  <link name="link1"/>
  <link name="sensor_top"/>
  <link name="sensor_bottom"/>
  <joint name="sensor_top_joint" type="fixed">
    <parent link="link1"/>
    <child link="sensor_top"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
  <joint name="sensor_bottom_joint" type="fixed">
    <parent link="link1"/>
    <child link="sensor_bottom"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
  <joint name="joint1" type="revolute">
    <parent link="base"/>
    <child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
"""
    urdf_path = tmp_path / "misordered.urdf"
    urdf_path.write_text(urdf, encoding="utf-8")

    kots = Kots.from_urdf_file(str(urdf_path), order=4)
    assert kots.joint_name_list() == [
        "world_to_base",
        "joint1",
        "sensor_top_joint",
        "sensor_bottom_joint",
    ]

    kots.import_motions(np.zeros(kots.dof() * kots.order(), dtype=float))
    kots.kinematics()
    kots.dynamics()


def test_kinematics():
    kots = _make_kots(order=3)
    motion = np.random.rand(kots.order() * kots.dof())
    kots.import_motions(motion)
    kots.kinematics()

    h_list = forward_kinematics(kots.robot_.joints, kots.motion(order=1))
    v_list = forward_kinematics_vel(kots.robot_.joints, kots.motion(order=2))
    a_list = forward_kinematics_acc(kots.robot_.joints, kots.motion(order=3))

    for i, link in enumerate(kots.link_name_list()):
        h = kots.state_info(StateType(data_type="frame", owner_type = "link", owner_name=link))
        v = kots.state_info(StateType(data_type="vel", owner_type = "link", owner_name=link))
        a = kots.state_info(StateType(data_type="acc", owner_type = "link", owner_name=link))

        assert np.allclose(h.mat(), h_list[i].mat())
        assert np.allclose(v, v_list[i])
        assert np.allclose(a, a_list[i])

def test_kinematics_backend_jax_matches_numpy():
    order = 5
    motion = np.random.default_rng(0).standard_normal(order * _make_kots(order=order).dof())

    kots = _make_kots(order=order)
    kots_jax = _make_kots(order=order)

    kots.import_motions(motion)
    kots_jax.import_motions(motion)

    kots.kinematics()
    kots_jax.kinematics(backend="jax")

    jax_frames = outward_kinematics_jax(kots_jax.robot_, kots_jax.motions_, order=1)
    assert jax_frames.names == tuple(kots_jax.link_name_list())
    target_idx = jax_frames.names.index(TARGET_LINK)
    np.testing.assert_allclose(
        np.asarray(jax_frames.state[target_idx]),
        kots.state_info(StateType(data_type="frame", owner_type="link", owner_name=TARGET_LINK)).mat(),
        atol=1e-6,
        rtol=1e-6,
    )

    for dt in ["frame", "vel", "acc", "jerk", "snap"]:
        state = StateType(data_type=dt, owner_type="link", owner_name=TARGET_LINK)
        actual = kots_jax.state_info(state)
        expected = kots.state_info(state)
        if dt == "frame":
            np.testing.assert_allclose(actual.mat(), expected.mat(), atol=1e-6, rtol=1e-6)
        else:
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)

    for dt in ["frame", "vel", "acc"]:
        state = StateType(data_type=dt, owner_type="link", owner_name=TARGET_LINK)
        np.testing.assert_allclose(kots_jax.jacobian(state), kots.jacobian(state), atol=1e-5, rtol=1e-5)


def test_kinematics_backend_rust_matches_numpy():
    pytest.importorskip("robokots._rust")
    order = 5
    motion = np.random.default_rng(24).standard_normal(order * _make_kots(order=order).dof())

    kots = _make_kots(order=order)
    kots_rust = _make_kots(order=order)
    kots.import_motions(motion)
    kots_rust.import_motions(motion)

    kots.kinematics(order=order)
    kots_rust.kinematics(order=order, backend="rust")

    for dt in ["frame", "vel", "acc", "jerk", "snap"]:
        state = StateType(data_type=dt, owner_type="link", owner_name=TARGET_LINK)
        actual = kots_rust.state_info(state)
        expected = kots.state_info(state)
        if dt == "frame":
            np.testing.assert_allclose(actual.mat(), expected.mat(), atol=1e-10, rtol=1e-10)
        else:
            np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_dynamics_backend_rust_matches_numpy():
    pytest.importorskip("robokots._rust")
    order = 5
    motion = np.random.default_rng(25).standard_normal(order * _make_kots(order=order).dof())

    kots = _make_kots(order=order)
    kots_rust = _make_kots(order=order)
    kots.import_motions(motion)
    kots_rust.import_motions(motion)

    kots.dynamics(order=order)
    kots_rust.dynamics(order=order, backend="rust")

    for state in [
        StateType("link", TARGET_LINK, "momentum"),
        StateType("link", TARGET_LINK, "force"),
        StateType("link", TARGET_LINK, "force_diff2"),
        StateType("joint", "joint3", "momentum"),
        StateType("joint", "joint3", "force"),
        StateType("joint", "joint3", "torque"),
        StateType("joint", "joint3", "torque_diff2"),
    ]:
        np.testing.assert_allclose(
            kots_rust.state_info(state),
            kots.state_info(state),
            atol=1e-10,
            rtol=1e-10,
        )


def test_rust_local_link_dynamics_jacobians_match_numpy():
    pytest.importorskip("robokots._rust")
    order = 3
    rng = np.random.default_rng(31)
    states = [
        StateType("link", TARGET_LINK, "momentum"),
        StateType("link", TARGET_LINK, "momentum_diff1"),
        StateType("link", TARGET_LINK, "force"),
    ]

    for batch_shape in [(), (3,)]:
        kots = _make_kots(order=order)
        kots_rust = _make_kots(order=order)
        motion_shape = batch_shape + (order * kots.dof(),)
        motion = rng.standard_normal(motion_shape)
        kots.import_motions(motion)
        kots_rust.import_motions(motion)
        kots.dynamics(order=order, materialize_dict=False)
        kots_rust.dynamics(order=order, backend="rust", materialize_dict=False)

        jacob = kots.jacobian(states)
        rust_jacob = kots_rust.jacobian(states)
        np.testing.assert_allclose(rust_jacob, jacob, atol=1e-10, rtol=1e-10)

        rhs = rng.standard_normal(batch_shape + (jacob.shape[-1], 4))
        np.testing.assert_allclose(
            kots_rust.jacobian_mul(states, rhs),
            kots.jacobian_mul(states, rhs),
            atol=1e-10,
            rtol=1e-10,
        )

        transpose_rhs = rng.standard_normal(batch_shape + (jacob.shape[-2], 2))
        np.testing.assert_allclose(
            kots_rust.jacobian_transpose_mul(states, transpose_rhs),
            kots.jacobian_transpose_mul(states, transpose_rhs),
            atol=1e-10,
            rtol=1e-10,
        )


def test_rust_dynamics_outward_cmtm_matches_split_outputs():
    rust_module = pytest.importorskip("robokots._rust")
    order = 5
    dynamics_order = order - 2
    model_data = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    rust_robot = rust_module.RustCompiledRobot.from_model_data(model_data)
    rng = np.random.default_rng(27)
    motion = rng.standard_normal(rust_robot.dof * order)

    combined = rust_robot.dynamics_outward_cmtm(motion, dynamics_order)
    split = (
        *rust_robot.kinematics_cmtm(motion, order),
        *rust_robot.dynamics_cmtm(motion, dynamics_order),
    )

    for actual, expected in zip(combined, split):
        np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)

    motions = rng.standard_normal((2, 3, rust_robot.dof * order))
    combined_batch = rust_robot.dynamics_outward_cmtm_batch(
        motions.reshape(-1, rust_robot.dof * order),
        dynamics_order,
    )
    split_batch = (
        *rust_robot.kinematics_cmtm_batch(motions.reshape(-1, rust_robot.dof * order), order),
        *rust_robot.dynamics_cmtm_batch(motions.reshape(-1, rust_robot.dof * order), dynamics_order),
    )

    for actual, expected in zip(combined_batch, split_batch):
        np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)


def test_rust_backend_array_state_exposes_cmtm_and_cmvector():
    pytest.importorskip("robokots._rust")
    from mathrobo import CMTM, CMVector
    from robokots.core.outward_state import ArrayOutwardState
    from robokots.core.state_dict import state_dict_to_cmvec, state_dict_to_cmtm
    from robokots.outward.rust import build_dynamics_outward_state_rust

    order = 5
    dynamics_order = order - 2
    rng = np.random.default_rng(28)
    kots = _make_kots(order=order)
    motion = rng.standard_normal(kots.dof() * order)

    state = build_dynamics_outward_state_rust(kots.robot_, motion, dynamics_order)

    assert isinstance(state, ArrayOutwardState)
    assert state._cache == {}
    assert isinstance(state.cmtm("link", TARGET_LINK, order), CMTM)
    assert isinstance(state.cmvec("joint", "joint3", "momentum"), CMVector)
    assert state.cmtm("link", TARGET_LINK, order) is state_dict_to_cmtm(
        state,
        TARGET_LINK,
        "link",
        order,
    )
    assert state.cmvec("joint", "joint3", "momentum") is state_dict_to_cmvec(
        state,
        "joint3",
        "joint",
        "momentum",
        dynamics_order + 1,
    )


def test_rust_kots_dynamics_can_defer_state_dict_materialization():
    pytest.importorskip("robokots._rust")
    from robokots.outward.rust import RustOutwardState

    order = 5
    rng = np.random.default_rng(29)
    kots = _make_kots(order=order)
    motion = rng.standard_normal(kots.dof() * order)
    kots.import_motions(motion)

    kots.dynamics(order=order, backend="rust", materialize_dict=False)

    assert isinstance(kots.outward_state_, RustOutwardState)
    assert kots.state_dict_ == {}
    for state in [
        StateType("link", TARGET_LINK, "snap"),
        StateType("link", TARGET_LINK, "force_diff2"),
        StateType("joint", "joint3", "torque_diff2"),
    ]:
        direct = outward_api.get_value(kots.robot_, kots.outward_state_, state)
        np.testing.assert_allclose(kots.state_info(state), direct, atol=1e-10, rtol=1e-10)

    state_dict = kots.to_state_dict()
    assert state_dict
    assert kots.state_dict_source_ is kots.outward_state_


def test_rust_kots_deferred_state_supports_dict_based_helpers():
    pytest.importorskip("robokots._rust")
    order = 5
    rng = np.random.default_rng(31)
    kots = _make_kots(order=order)
    motion = rng.standard_normal(kots.dof() * order)
    kots.import_motions(motion)

    kots.kinematics(order=order, backend="rust", materialize_dict=False)

    assert kots.state_dict_ == {}
    point_frame = kots.kinematics_point(0.0)
    assert point_frame is not None
    assert kots.state_dict_
    assert kots.state_dict_source_ is kots.outward_state_


def test_rust_private_fast_kots_helpers_match_compiled_robot():
    rust_module = pytest.importorskip("robokots._rust")
    order = 3
    rng = np.random.default_rng(32)
    model_data = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    rust_robot = rust_module.RustCompiledRobot.from_model_data(model_data)
    kots = _make_kots(order=order)
    q = rng.standard_normal(kots.dof())
    v = rng.standard_normal(kots.dof())
    a = rng.standard_normal(kots.dof())

    for actual, expected in zip(
        kots._rust_fast_forward_kinematics(q, v, a),
        rust_robot.forward_kinematics(q, v, a),
    ):
        np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(kots._rust_fast_rnea(q, v, a), rust_robot.rnea(q, v, a), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(kots._rust_fast_joint_jacobians(q), rust_robot.joint_jacobians(q), atol=0.0, rtol=0.0)


def test_rust_private_fast_kots_batch_helpers_match_sample_loop():
    pytest.importorskip("robokots._rust")
    order = 3
    rng = np.random.default_rng(33)
    kots = _make_kots(order=order)
    q = rng.standard_normal((2, 3, kots.dof()))
    v = rng.standard_normal(q.shape)
    a = rng.standard_normal(q.shape)
    flat_q = q.reshape(-1, kots.dof())
    flat_v = v.reshape(-1, kots.dof())
    flat_a = a.reshape(-1, kots.dof())

    fk_batch = kots._rust_fast_forward_kinematics(flat_q, flat_v, flat_a)
    fk_loop = tuple(
        np.stack(
            [
                kots._rust_fast_forward_kinematics(flat_q[i], flat_v[i], flat_a[i])[j]
                for i in range(flat_q.shape[0])
            ],
            axis=0,
        )
        for j in range(6)
    )
    for actual, expected in zip(fk_batch, fk_loop):
        np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)

    np.testing.assert_allclose(
        kots._rust_fast_rnea(flat_q, flat_v, flat_a),
        np.stack([kots._rust_fast_rnea(flat_q[i], flat_v[i], flat_a[i]) for i in range(flat_q.shape[0])]),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        kots._rust_fast_joint_jacobians(flat_q),
        np.stack([kots._rust_fast_joint_jacobians(flat_q[i]) for i in range(flat_q.shape[0])]),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize("layout", ["fortran", "permuted", "strided", "negative_stride"])
def test_rust_batch_inputs_accept_non_c_layouts(layout):
    rust_module = pytest.importorskip("robokots._rust")
    rng = np.random.default_rng(34)
    kots = _make_kots(order=3)
    model_data = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    rust_robot = rust_module.RustCompiledRobot.from_model_data(model_data)

    def arrange(values):
        if layout == "fortran":
            return np.asfortranarray(values)
        if layout == "permuted":
            permutation = np.arange(values.shape[1])[::-1]
            return values[:, permutation]
        if layout == "strided":
            storage = np.empty((values.shape[0], values.shape[1] * 2))
            storage[:, ::2] = values
            return storage[:, ::2]
        return values[:, ::-1]

    inputs = tuple(arrange(rng.standard_normal((5, kots.dof()))) for _ in range(3))
    contiguous = tuple(np.ascontiguousarray(value) for value in inputs)
    assert not all(value.flags.c_contiguous for value in inputs)
    normalized = kots._fast_qva(*inputs)
    assert all(value.flags.c_contiguous for value in normalized)
    for actual, expected in zip(normalized, contiguous):
        np.testing.assert_array_equal(actual, expected)

    expected_tau = rust_robot.rnea_batch(*contiguous)
    np.testing.assert_allclose(rust_robot.rnea_batch(*inputs), expected_tau, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(kots._rust_fast_rnea(*inputs), expected_tau, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        kots.inverse_dynamics(*inputs, gravity=np.zeros(3)),
        expected_tau,
        atol=0.0,
        rtol=0.0,
    )

    expected_fk = rust_robot.forward_kinematics_batch(*contiguous)
    for actual, expected in zip(rust_robot.forward_kinematics_batch(*inputs), expected_fk):
        np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)

    expected_jacobians = rust_robot.joint_jacobians_batch(contiguous[0])
    np.testing.assert_allclose(
        rust_robot.joint_jacobians_batch(inputs[0]),
        expected_jacobians,
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        kots._rust_fast_joint_jacobians(inputs[0]),
        expected_jacobians,
        atol=0.0,
        rtol=0.0,
    )

    scalar_tau = np.stack(
        [rust_robot.rnea(inputs[0][i], inputs[1][i], inputs[2][i]) for i in range(inputs[0].shape[0])]
    )
    np.testing.assert_allclose(scalar_tau, expected_tau, atol=0.0, rtol=0.0)


def test_rust_private_fast_kots_helpers_validate_shapes_and_backend():
    pytest.importorskip("robokots._rust")
    kots = _make_kots(order=3)
    q = np.zeros(kots.dof())
    with pytest.raises(ValueError, match="Unsupported fast backend"):
        kots._rust_fast_rnea(q, q, q, backend="numpy")
    with pytest.raises(ValueError, match="q length"):
        kots._rust_fast_joint_jacobians(np.zeros(kots.dof() + 1))
    with pytest.raises(ValueError, match="v shape"):
        kots._rust_fast_forward_kinematics(q, np.zeros((2, kots.dof())), q)


def test_inverse_dynamics_gravity_api_preserves_zero_gravity_and_batches():
    pytest.importorskip("robokots._rust")
    rng = np.random.default_rng(47)
    kots = _make_kots(order=3)
    q = rng.standard_normal((3, kots.dof()))
    v = rng.standard_normal(q.shape)
    a = rng.standard_normal(q.shape)
    gravity = np.array([0.7, -1.3, -9.2])

    np.testing.assert_allclose(
        kots.inverse_dynamics(q, v, a, gravity=gravity),
        np.stack([kots.inverse_dynamics(q[i], v[i], a[i], gravity=gravity) for i in range(q.shape[0])]),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        kots.inverse_dynamics(q, v, a, gravity=np.zeros(3)),
        kots._rust_fast_rnea(q, v, a),
        atol=0.0,
        rtol=0.0,
    )

    with pytest.raises(ValueError, match="gravity must have shape"):
        kots.inverse_dynamics(q, v, a, gravity=np.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        kots.inverse_dynamics(q, v, a, gravity=[0.0, np.nan, 0.0])


def test_inverse_dynamics_prismatic_matches_numpy_backend(tmp_path):
    pytest.importorskip("robokots._rust")
    urdf = """<robot name="prismatic_inverse_dynamics">
      <link name="base"/>
      <link name="body">
        <inertial>
          <origin xyz="0.1 -0.2 0.05" rpy="0.2 -0.1 0.3"/>
          <mass value="2.5"/>
          <inertia ixx="0.1" ixy="0.01" ixz="-0.005"
                   iyy="0.2" iyz="0.008" izz="0.3"/>
        </inertial>
      </link>
      <joint name="slide" type="prismatic">
        <parent link="base"/>
        <child link="body"/>
        <origin xyz="0.2 0.1 -0.1" rpy="0.1 -0.2 0.3"/>
        <axis xyz="1 2 3"/>
        <limit lower="-0.5" upper="0.8" effort="100" velocity="10"/>
      </joint>
    </robot>"""
    urdf_path = tmp_path / "prismatic.urdf"
    urdf_path.write_text(urdf, encoding="utf-8")
    kots = Kots.from_urdf_file(str(urdf_path), order=3)
    q = np.array([0.3])
    v = np.array([0.4])
    a = np.array([-0.5])

    kots.import_motion_array(np.stack([q, v, a], axis=-1))
    kots.dynamics(backend="numpy", materialize_dict=False)
    expected = np.asarray(
        kots.state_info(StateType("total_joint", "total_joint", "torque"))
    ).reshape(-1)

    np.testing.assert_allclose(
        kots.inverse_dynamics(q, v, a, gravity=np.zeros(3)),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_rust_fast_data_matches_allocating_private_fast_helpers():
    pytest.importorskip("robokots._rust")
    order = 3
    rng = np.random.default_rng(45)
    kots = _make_kots(order=order)
    q = rng.standard_normal(kots.dof())
    v = rng.standard_normal(kots.dof())
    a = rng.standard_normal(kots.dof())

    data = kots._create_rust_fast_data()
    data.compute_kinematics(q, v, a)
    fk = kots._rust_fast_forward_kinematics(q, v, a)
    np.testing.assert_allclose(data.rotations(), fk[0], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(data.positions(), fk[1], atol=0.0, rtol=0.0)

    data.compute_dynamics(q, v, a)
    np.testing.assert_allclose(data.tau(), kots._rust_fast_rnea(q, v, a), atol=0.0, rtol=0.0)

    data.compute_joint_jacobians(q)
    np.testing.assert_allclose(data.joint_jacobians(), kots._rust_fast_joint_jacobians(q), atol=0.0, rtol=0.0)


def test_rust_pinocchio_like_data_alias_matches_fast_data():
    pytest.importorskip("robokots._rust")
    rng = np.random.default_rng(46)
    kots = _make_kots(order=3)
    q = rng.standard_normal(kots.dof())
    v = rng.standard_normal(kots.dof())
    a = rng.standard_normal(kots.dof())

    fast_data = kots._create_rust_fast_data()
    pin_data = kots._create_rust_pinocchio_like_data()
    fast_data.compute_dynamics(q, v, a)
    pin_data.compute_dynamics(q, v, a)

    np.testing.assert_allclose(pin_data.rotations(), fast_data.rotations(), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(pin_data.positions(), fast_data.positions(), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(pin_data.tau(), fast_data.tau(), atol=0.0, rtol=0.0)


def test_rust_outward_data_getters_match_array_state():
    rust_module = pytest.importorskip("robokots._rust")
    order = 5
    rng = np.random.default_rng(34)
    motion = rng.standard_normal(_make_kots(order=order).dof() * order)
    kots = _make_kots(order=order)
    kots.import_motions(motion)
    kots.dynamics(order=order, backend="rust", materialize_dict=False)

    data = kots._create_rust_outward_state(order=order)
    assert isinstance(data.raw_data, rust_module.RustOutwardData)
    data.compute_dynamics(motion)

    state = kots.outward_state_

    np.testing.assert_allclose(data.link_mat(TARGET_LINK), state.link_mat(TARGET_LINK), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(data.link_vec(TARGET_LINK, 5), state.link_vec(TARGET_LINK, 5), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(data.joint_mat("joint3"), state.joint_mat("joint3"), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(data.joint_vec("joint3", 3), state.joint_vec("joint3", 3), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(data.link_momentum(TARGET_LINK, 4), state.link_momentum(TARGET_LINK, 4), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(data.link_force(TARGET_LINK, 3), state.link_force(TARGET_LINK, 3), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(data.joint_momentum("joint3", 4), state.joint_momentum("joint3", 4), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(data.joint_force("joint3", 3), state.joint_force("joint3", 3), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(data.joint_torque("joint3", 3), state.joint_torque("joint3", 3), atol=0.0, rtol=0.0)


def test_rust_outward_minimal_dynamics_matches_torque():
    pytest.importorskip("robokots._rust")
    order = 3
    rng = np.random.default_rng(47)
    kots = _make_kots(order=order)
    motion = rng.standard_normal(kots.dof() * order)

    full = kots._create_rust_outward_state(order=order).compute_dynamics(motion)
    minimal = kots._create_rust_outward_state(order=order).compute_dynamics_minimal(motion)

    for joint in kots.robot_.joints:
        if joint.dof <= 0:
            continue
        np.testing.assert_allclose(
            minimal.joint_torque(joint.name, 1),
            full.joint_torque(joint.name, 1),
            atol=1e-10,
            rtol=1e-10,
        )
    with pytest.raises(ValueError, match="compute_dynamics_minimal"):
        minimal.link_force(TARGET_LINK, 1)


def test_rust_outward_data_kinematics_and_validation():
    pytest.importorskip("robokots._rust")
    order = 5
    rng = np.random.default_rng(35)
    kots = _make_kots(order=order)
    motion = rng.standard_normal(kots.dof() * order)
    data = kots._create_rust_outward_state(order=order)

    with pytest.raises(ValueError, match="compute_kinematics|compute_dynamics"):
        data.link_vec(TARGET_LINK, 2)
    with pytest.raises(ValueError, match="motion must have shape"):
        data.compute_kinematics(motion.reshape(1, -1))

    data.compute_kinematics(motion)
    kots.import_motions(motion)
    kots.kinematics(order=order, backend="rust", materialize_dict=False)

    np.testing.assert_allclose(data.link_mat(TARGET_LINK), kots.outward_state_.link_mat(TARGET_LINK), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(data.link_vec(TARGET_LINK, 5), kots.outward_state_.link_vec(TARGET_LINK, 5), atol=0.0, rtol=0.0)
    with pytest.raises(ValueError, match="compute_dynamics"):
        data.joint_torque("joint3", 1)


def test_rust_outward_data_state_info_direct_kinematics_matches_array_state():
    pytest.importorskip("robokots._rust")
    order = 5
    rng = np.random.default_rng(36)
    motion = rng.standard_normal(_make_kots(order=order).dof() * order)

    expected = _make_kots(order=order)
    expected.import_motions(motion)
    expected.kinematics(order=order, backend="rust", materialize_dict=False)

    actual = _make_kots(order=order)
    actual.import_motions(motion)
    data = actual.update_rust_data(order=order)
    assert actual.outward_state_ is data

    frame_state = StateType("link", TARGET_LINK, "frame")
    np.testing.assert_allclose(
        actual.state_info(frame_state).mat(),
        expected.state_info(frame_state).mat(),
        atol=0.0,
        rtol=0.0,
    )
    for state in [
        StateType("link", TARGET_LINK, "pos"),
        StateType("link", TARGET_LINK, "rot"),
        StateType("link", TARGET_LINK, "snap"),
        StateType("joint", "joint3", "acc"),
    ]:
        np.testing.assert_allclose(
            actual.state_info(state),
            expected.state_info(state),
            atol=0.0,
            rtol=0.0,
        )


def test_rust_outward_data_state_info_direct_dynamics_matches_array_state():
    pytest.importorskip("robokots._rust")
    order = 5
    rng = np.random.default_rng(37)
    motion = rng.standard_normal(_make_kots(order=order).dof() * order)

    expected = _make_kots(order=order)
    expected.import_motions(motion)
    expected.dynamics(order=order, backend="rust", materialize_dict=False)

    actual = _make_kots(order=order)
    actual.import_motions(motion)
    actual.update_rust_data(order=order, is_dynamics=True)

    states = [
        StateType("link", TARGET_LINK, "momentum_diff3"),
        StateType("link", TARGET_LINK, "force_diff2"),
        StateType("joint", "joint3", "momentum_diff3"),
        StateType("joint", "joint3", "force_diff2"),
        StateType("joint", "joint3", "torque_diff2"),
    ]
    for state in states:
        np.testing.assert_allclose(
            actual.state_info(state),
            expected.state_info(state),
            atol=0.0,
            rtol=0.0,
        )

    total = StateType("total_joint", "total_joint", "torque_diff2")
    np.testing.assert_allclose(
        actual.state_info(total),
        expected.state_info(total),
        atol=0.0,
        rtol=0.0,
    )
    for state in [
        StateType("link", TARGET_LINK, "momentum", "world"),
        StateType("link", TARGET_LINK, "momentum_diff3", "world"),
        StateType("link", TARGET_LINK, "force", "world"),
        StateType("link", TARGET_LINK, "force_diff2", "world"),
        StateType("joint", "joint3", "momentum", "world"),
        StateType("joint", "joint3", "momentum_diff3", "world"),
        StateType("joint", "joint3", "force", "world"),
        StateType("joint", "joint3", "force_diff2", "world"),
    ]:
        np.testing.assert_allclose(
            actual.state_info(state),
            expected.state_info(state),
            atol=1e-10,
            rtol=1e-10,
        )


def test_update_rust_data_reuses_cached_workspace_for_same_order():
    pytest.importorskip("robokots._rust")
    order = 5
    rng = np.random.default_rng(38)
    kots = _make_kots(order=order)
    motion = rng.standard_normal(kots.dof() * order)
    kots.import_motions(motion)

    data = kots.update_rust_data(order=order, is_dynamics=True)
    calls = []
    original_compute_dynamics = data.compute_dynamics

    def counted_compute_dynamics(motion):
        calls.append(np.asarray(motion).copy())
        return original_compute_dynamics(motion)

    data.compute_dynamics = counted_compute_dynamics

    assert kots.update_rust_data(order=order, is_dynamics=True) is data
    assert calls == []

    kots.import_motions(rng.standard_normal(kots.dof() * order))
    assert kots.update_rust_data(order=order, is_dynamics=True) is data
    assert len(calls) == 1


def test_update_state_rust_uses_cached_rust_data_path():
    pytest.importorskip("robokots._rust")
    order = 5
    rng = np.random.default_rng(41)
    motion = rng.standard_normal(_make_kots(order=order).dof() * order)

    expected = _make_kots(order=order)
    expected.import_motions(motion)
    expected.dynamics(order=order, backend="rust", materialize_dict=False)

    actual = _make_kots(order=order)
    actual.import_motions(motion)
    state = actual.update_state(order=order, is_dynamics=True, backend="rust")

    assert actual.outward_state_ is state
    assert actual.update_state(order=order, is_dynamics=True, backend="rust") is state
    for state_type in [
        StateType("link", TARGET_LINK, "force", "world"),
        StateType("link", TARGET_LINK, "force_diff2", "world"),
        StateType("joint", "joint3", "momentum", "world"),
        StateType("joint", "joint3", "momentum_diff3", "world"),
    ]:
        np.testing.assert_allclose(
            actual.state_info(state_type),
            expected.state_info(state_type),
            atol=1e-10,
            rtol=1e-10,
        )


def test_update_rust_data_cmtm_view_matches_array_state_without_materializing_mathrobo_cmtm():
    pytest.importorskip("robokots._rust")
    order = 5
    rng = np.random.default_rng(42)
    motion = rng.standard_normal(_make_kots(order=order).dof() * order)

    expected = _make_kots(order=order)
    expected.import_motions(motion)
    expected.kinematics(order=order, backend="rust", materialize_dict=False)

    actual = _make_kots(order=order)
    actual.import_motions(motion)
    actual.update_state(order=order, backend="rust")

    expected_cmtm = expected.outward_state_.cmtm("link", TARGET_LINK, order)
    actual_cmtm = actual.outward_state_.cmtm("link", TARGET_LINK, order)
    assert type(actual_cmtm).__name__ == "_RustCMTMView"
    np.testing.assert_allclose(actual_cmtm.elem_mat(), expected_cmtm.elem_mat(), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(actual_cmtm.vecs(), expected_cmtm.vecs(), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(actual_cmtm.mat_adj(), expected_cmtm.mat_adj(), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(actual_cmtm.mat_inv_adj(), expected_cmtm.mat_inv_adj(), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(actual_cmtm.tangent_mat(), expected_cmtm.tangent_mat(), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(actual_cmtm.tangent_mat_inv(), expected_cmtm.tangent_mat_inv(), atol=1e-10, rtol=1e-10)

    child_name = actual.robot_.links[actual.robot_.joints[-1].child_link_id].name
    expected_rel = expected.outward_state_.rel_cmtm(TARGET_LINK, child_name, "link", order)
    actual_rel = actual.outward_state_.rel_cmtm(TARGET_LINK, child_name, "link", order)
    np.testing.assert_allclose(actual_rel.elem_mat(), expected_rel.elem_mat(), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(actual_rel.vecs(), expected_rel.vecs(), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(actual_rel.mat_adj(), expected_rel.mat_adj(), atol=1e-10, rtol=1e-10)


def test_update_rust_data_cmvector_view_and_cmtm_var_jacob_match_array_state():
    pytest.importorskip("robokots._rust")
    from robokots.core.state_dict import state_dict_to_cmvec

    order = 5
    dynamics_order = order - 2
    momentum_order = dynamics_order + 1
    rng = np.random.default_rng(43)
    motion = rng.standard_normal(_make_kots(order=order).dof() * order)

    expected = _make_kots(order=order)
    expected.import_motions(motion)
    expected.dynamics(order=order, backend="rust", materialize_dict=False)

    actual = _make_kots(order=order)
    actual.import_motions(motion)
    actual.update_state(order=order, is_dynamics=True, backend="rust")

    expected_vec = expected.outward_state_.cmvec("link", TARGET_LINK, "momentum")
    actual_vec = actual.outward_state_.cmvec("link", TARGET_LINK, "momentum")
    assert type(actual_vec).__name__ == "_RustCMVectorView"
    np.testing.assert_allclose(actual_vec.vecs(), expected_vec.vecs(), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(actual_vec.cm_vec(), expected_vec.cm_vec(), atol=1e-15, rtol=1e-15)

    truncated = state_dict_to_cmvec(
        actual.outward_state_,
        TARGET_LINK,
        "link",
        "momentum",
        momentum_order - 1,
    )
    assert type(truncated).__name__ == "_RustCMVectorView"
    np.testing.assert_allclose(
        truncated.vecs(),
        expected_vec.vecs()[..., : momentum_order - 1, :],
        atol=0.0,
        rtol=0.0,
    )

    expected_cmtm = expected.outward_state_.cmtm_wrench("link", TARGET_LINK, momentum_order)
    actual_cmtm = actual.outward_state_.cmtm_wrench("link", TARGET_LINK, momentum_order)
    rhs = rng.standard_normal(momentum_order * 6)
    rhs_mat = rng.standard_normal((momentum_order * 6, 4))
    np.testing.assert_allclose(
        actual_cmtm.mat_var_x_arb_vec_jacob(actual_vec, frame="bframe"),
        expected_cmtm.mat_var_x_arb_vec_jacob(expected_vec, frame="bframe"),
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        actual_cmtm.mat_inv_var_x_arb_vec_jacob(actual_vec, frame="bframe"),
        expected_cmtm.mat_inv_var_x_arb_vec_jacob(expected_vec, frame="bframe"),
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        actual_cmtm.mat_var_x_arb_vec_matvec(actual_vec, rhs, frame="bframe"),
        expected_cmtm.mat_var_x_arb_vec_jacob(expected_vec, frame="bframe") @ rhs,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        actual_cmtm.mat_var_x_arb_vec_matvec(actual_vec, rhs, frame="bframe", transpose=True),
        expected_cmtm.mat_var_x_arb_vec_jacob(expected_vec, frame="bframe").T @ rhs,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        actual_cmtm.mat_inv_var_x_arb_vec_matvec(actual_vec, rhs, frame="bframe"),
        expected_cmtm.mat_inv_var_x_arb_vec_jacob(expected_vec, frame="bframe") @ rhs,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        actual_cmtm.mat_inv_var_x_arb_vec_matvec(actual_vec, rhs, frame="bframe", transpose=True),
        expected_cmtm.mat_inv_var_x_arb_vec_jacob(expected_vec, frame="bframe").T @ rhs,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        actual_cmtm.mat_var_x_arb_vec_matmul_rhs(actual_vec, rhs_mat, frame="bframe"),
        expected_cmtm.mat_var_x_arb_vec_jacob(expected_vec, frame="bframe") @ rhs_mat,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        actual_cmtm.mat_inv_var_x_arb_vec_matmul_rhs(actual_vec, rhs_mat, frame="bframe"),
        expected_cmtm.mat_inv_var_x_arb_vec_jacob(expected_vec, frame="bframe") @ rhs_mat,
        atol=1e-10,
        rtol=1e-10,
    )

    state = StateType("link", TARGET_LINK, "momentum_diff3", "world")
    vec = rng.standard_normal(6)
    np.testing.assert_allclose(
        actual.jacobian_transpose_mul(state, vec),
        expected.jacobian_transpose_mul(state, vec),
        atol=1e-9,
        rtol=1e-9,
    )


def test_update_rust_data_batch_cmvector_view_and_cmtm_var_jacob_match_array_state():
    pytest.importorskip("robokots._rust")
    order = 5
    rng = np.random.default_rng(44)
    template = _make_kots(order=order)
    motions = rng.standard_normal((2, 3, template.dof() * order))

    expected = _make_kots(order=order)
    expected.import_motions(motions)
    expected.dynamics(order=order, backend="rust", materialize_dict=False)

    actual = _make_kots(order=order)
    actual.import_motions(motions)
    actual.update_state(order=order, is_dynamics=True, backend="rust")

    expected_vec = expected.outward_state_.cmvec("link", TARGET_LINK, "momentum")
    actual_vec = actual.outward_state_.cmvec("link", TARGET_LINK, "momentum")
    expected_cmtm = expected.outward_state_.cmtm_wrench("link", TARGET_LINK, order - 1)
    actual_cmtm = actual.outward_state_.cmtm_wrench("link", TARGET_LINK, order - 1)
    rhs = rng.standard_normal(motions.shape[:-1] + ((order - 1) * 6,))
    rhs_mat = rng.standard_normal(motions.shape[:-1] + ((order - 1) * 6, 4))

    assert type(actual_vec).__name__ == "_RustCMVectorView"
    np.testing.assert_allclose(actual_vec.vecs(), expected_vec.vecs(), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        actual_cmtm.mat_var_x_arb_vec_jacob(actual_vec, frame="bframe"),
        expected_cmtm.mat_var_x_arb_vec_jacob(expected_vec, frame="bframe"),
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        actual_cmtm.mat_var_x_arb_vec_matvec(actual_vec, rhs, frame="bframe"),
        (
            expected_cmtm.mat_var_x_arb_vec_jacob(expected_vec, frame="bframe")
            @ rhs[..., None]
        )[..., 0],
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        actual_cmtm.mat_var_x_arb_vec_matmul_rhs(actual_vec, rhs_mat, frame="bframe"),
        expected_cmtm.mat_var_x_arb_vec_jacob(expected_vec, frame="bframe") @ rhs_mat,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        actual_cmtm.mat_inv_var_x_arb_vec_jacob(actual_vec, frame="bframe"),
        expected_cmtm.mat_inv_var_x_arb_vec_jacob(expected_vec, frame="bframe"),
        atol=1e-10,
        rtol=1e-10,
    )


def test_update_rust_data_promotes_cached_kinematics_to_dynamics():
    pytest.importorskip("robokots._rust")
    order = 5
    rng = np.random.default_rng(39)
    kots = _make_kots(order=order)
    kots.import_motions(rng.standard_normal(kots.dof() * order))

    data = kots.update_rust_data(order=order, is_dynamics=False)
    calls = []
    original_compute_dynamics = data.compute_dynamics

    def counted_compute_dynamics(motion):
        calls.append(np.asarray(motion).copy())
        return original_compute_dynamics(motion)

    data.compute_dynamics = counted_compute_dynamics

    assert kots.update_rust_data(order=order, is_dynamics=True) is data
    assert len(calls) == 1
    assert kots.update_rust_data(order=order, is_dynamics=False) is data
    assert len(calls) == 1


def test_update_rust_data_batch_state_info_matches_array_state():
    rust_module = pytest.importorskip("robokots._rust")
    order = 5
    rng = np.random.default_rng(40)
    template = _make_kots(order=order)
    motions = rng.standard_normal((2, 3, template.dof() * order))

    expected = _make_kots(order=order)
    expected.import_motions(motions)
    expected.dynamics(order=order, backend="rust", materialize_dict=False)

    actual = _make_kots(order=order)
    actual.import_motions(motions)
    data = actual.update_rust_data(order=order, is_dynamics=True)

    assert isinstance(data.raw_data, rust_module.RustBatchOutwardData)
    assert actual.batch_shape_ == (2, 3)
    for state in [
        StateType("link", TARGET_LINK, "snap"),
        StateType("link", TARGET_LINK, "momentum_diff3"),
        StateType("joint", "joint3", "force_diff2"),
        StateType("joint", "joint3", "torque_diff2"),
    ]:
        np.testing.assert_allclose(
            actual.state_info(state),
            expected.state_info(state),
            atol=0.0,
            rtol=0.0,
        )

    frame_state = StateType("link", TARGET_LINK, "frame")
    np.testing.assert_allclose(
        actual.state_info(frame_state),
        expected.state_info(frame_state),
        atol=0.0,
        rtol=0.0,
    )


def test_update_rust_data_batch_reuses_workspace_per_batch_shape():
    pytest.importorskip("robokots._rust")
    order = 5
    rng = np.random.default_rng(41)
    kots = _make_kots(order=order)
    motions = rng.standard_normal((2, 3, kots.dof() * order))
    kots.import_motions(motions)

    data = kots.update_rust_data(order=order, is_dynamics=True)
    assert kots.update_rust_data(order=order, is_dynamics=True) is data

    kots.import_motions(rng.standard_normal((2, 3, kots.dof() * order)))
    assert kots.update_rust_data(order=order, is_dynamics=True) is data

    kots.import_motions(rng.standard_normal((6, kots.dof() * order)))
    assert kots.update_rust_data(order=order, is_dynamics=True) is not data


def test_public_rust_backend_uses_cached_outward_state():
    pytest.importorskip("robokots._rust")
    from robokots.outward.rust import RustOutwardState

    order = 5
    rng = np.random.default_rng(46)
    kots = _make_kots(order=order)
    kots.import_motions(rng.standard_normal(kots.dof() * order))

    kots.kinematics(order=order, backend="rust")
    data = kots.outward_state_
    assert isinstance(data, RustOutwardState)
    assert kots.state_dict_
    assert kots.state_dict_source_ is data

    kots.kinematics(order=order, backend="rust", materialize_dict=False)
    assert kots.outward_state_ is data
    assert kots.state_dict_ == {}

    kots.dynamics(order=order, backend="rust", materialize_dict=False)
    assert kots.outward_state_ is data
    np.testing.assert_allclose(
        kots.state_info(StateType("joint", "joint3", "torque_diff2")),
        outward_api.get_value(kots.robot_, data, StateType("joint", "joint3", "torque_diff2")),
        atol=0.0,
        rtol=0.0,
    )


def test_rust_kots_batch_can_defer_state_dict_materialization():
    pytest.importorskip("robokots._rust")
    from robokots.outward.rust import RustBatchOutwardState

    order = 5
    rng = np.random.default_rng(30)
    template = _make_kots(order=order)
    motions = rng.standard_normal((2, 3, template.dof() * order))
    kots = _make_kots(order=order)
    kots.import_motions(motions)

    kots.kinematics(order=order, backend="rust", materialize_dict=False)

    assert isinstance(kots.outward_state_, RustBatchOutwardState)
    assert kots.state_batch_ is None
    assert kots.state_dict_ == {}
    state = StateType("link", TARGET_LINK, "snap")
    np.testing.assert_allclose(
        kots.state_info(state),
        outward_api.get_value(kots.robot_, kots.outward_state_, state),
        atol=1e-10,
        rtol=1e-10,
    )

    state_dict = kots.to_state_dict()
    assert state_dict
    assert kots.state_dict_source_ is kots.outward_state_


def test_rust_backend_batch_matches_numpy():
    pytest.importorskip("robokots._rust")
    order = 5
    rng = np.random.default_rng(26)
    template = _make_kots(order=order)
    motions = rng.standard_normal((2, 3, template.dof() * order))

    kots = _make_kots(order=order)
    kots_rust = _make_kots(order=order)
    kots.import_motions(motions)
    kots_rust.import_motions(motions)

    kots.kinematics(order=order)
    kots_rust.kinematics(order=order, backend="rust")
    for state in [
        StateType("link", TARGET_LINK, "vel"),
        StateType("link", TARGET_LINK, "acc"),
        StateType("link", TARGET_LINK, "snap"),
    ]:
        np.testing.assert_allclose(
            kots_rust.state_info(state),
            kots.state_info(state),
            atol=1e-10,
            rtol=1e-10,
        )

    kots.dynamics(order=order)
    kots_rust.dynamics(order=order, backend="rust")
    for state in [
        StateType("link", TARGET_LINK, "force"),
        StateType("joint", "joint3", "torque"),
        StateType("joint", "joint3", "torque_diff2"),
    ]:
        np.testing.assert_allclose(
            kots_rust.state_info(state),
            kots.state_info(state),
            atol=1e-10,
            rtol=1e-10,
        )


def test_kinematics_numerical():
    kots = _make_kots(order=3)
    motion = np.random.rand(kots.order() * kots.dof())
    kots.import_motions(motion)
    kots.kinematics()

    dv = np.random.rand(kots.dof())
    vec = kots.link_diff_kinematics_numerical([TARGET_LINK], "cmtm", kots.order(), update_direction=dv)

    alias = ["frame", "vel", "acc", "jerk", "snap"]

    for i in range(kots.order()-1):
        ana_vec = kots.state_info(StateType(data_type=alias[i+1], owner_type = "link", owner_name=TARGET_LINK)) 
        num_vec = kots.link_diff_kinematics_numerical([TARGET_LINK], alias[i], order = kots.order(), update_direction=dv)

        num_vec2 = vec[:,6*i:6*(i+1)]

        assert np.allclose(ana_vec, num_vec)
        assert np.allclose(ana_vec, num_vec2)
    
def test_jacobian_numerical():
    kots = _make_kots(order=3)

    motion = np.random.rand(kots.order()*kots.dof())

    kots.import_motions(motion)
    kots.kinematics()

    for dt in ["frame", "vel", "acc"]:
        state = StateType(data_type=dt, owner_type="link", owner_name=TARGET_LINK)
        jacob = kots.jacobian(state)
        jacob_num = kots.jacobian(state, numerical=True)
        assert np.allclose(jacob, jacob_num, atol=1e-5, rtol=1e-5)


def test_jacobian_mul_vector_kinematics_matches_jacobian_product():
    kots = _make_kots(order=3)
    rng = np.random.default_rng(0)

    motion = rng.standard_normal(kots.order() * kots.dof())
    kots.import_motions(motion)
    kots.kinematics()

    states = [
        StateType(data_type="frame", owner_type="link", owner_name=TARGET_LINK),
        StateType(data_type="vel", owner_type="link", owner_name=TARGET_LINK),
        StateType(data_type="acc", owner_type="link", owner_name=TARGET_LINK),
    ]
    vec = rng.standard_normal(kots.dof() * StateType.max_time_order(states))

    np.testing.assert_allclose(
        kots.jacobian_mul(states, vec),
        kots.jacobian(states) @ vec,
        atol=1e-10,
        rtol=1e-10,
    )

    actual_parts = kots.jacobian_mul(states, vec, list_output=True)
    expected_parts = [jacob @ vec for jacob in kots.jacobian(states, list_output=True)]
    for actual, expected in zip(actual_parts, expected_parts):
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_jacobian_mul_matrix_kinematics_matches_jacobian_product():
    kots = _make_kots(order=3)
    rng = np.random.default_rng(19)

    motion = rng.standard_normal(kots.order() * kots.dof())
    kots.import_motions(motion)
    kots.kinematics()

    states = [
        StateType(data_type="frame", owner_type="link", owner_name=TARGET_LINK),
        StateType(data_type="vel", owner_type="link", owner_name=TARGET_LINK),
        StateType(data_type="acc", owner_type="link", owner_name=TARGET_LINK),
    ]
    mat = rng.standard_normal((kots.dof() * StateType.max_time_order(states), 4))

    np.testing.assert_allclose(
        kots.jacobian_mul(states, mat),
        kots.jacobian(states) @ mat,
        atol=1e-10,
        rtol=1e-10,
    )
    actual_parts = kots.jacobian_mul(states, mat, list_output=True)
    expected_parts = [jacob @ mat for jacob in kots.jacobian(states, list_output=True)]
    for actual, expected in zip(actual_parts, expected_parts):
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_jacobian_transpose_mul_vector_kinematics_matches_jacobian_product():
    kots = _make_kots(order=3)
    rng = np.random.default_rng(12)

    motion = rng.standard_normal(kots.order() * kots.dof())
    kots.import_motions(motion)
    kots.kinematics()

    states = [
        StateType(data_type="frame", owner_type="link", owner_name=TARGET_LINK),
        StateType(data_type="vel", owner_type="link", owner_name=TARGET_LINK),
        StateType(data_type="acc", owner_type="link", owner_name=TARGET_LINK),
    ]
    jacob = kots.jacobian(states)
    vec = rng.standard_normal(jacob.shape[0])

    np.testing.assert_allclose(
        kots.jacobian_transpose_mul(states, vec),
        jacob.T @ vec,
        atol=1e-10,
        rtol=1e-10,
    )


def test_jacobian_transpose_mul_matrix_kinematics_matches_jacobian_product():
    kots = _make_kots(order=3)
    rng = np.random.default_rng(22)

    motion = rng.standard_normal(kots.order() * kots.dof())
    kots.import_motions(motion)
    kots.kinematics()

    states = [
        StateType(data_type="frame", owner_type="link", owner_name=TARGET_LINK),
        StateType(data_type="vel", owner_type="link", owner_name=TARGET_LINK),
        StateType(data_type="acc", owner_type="link", owner_name=TARGET_LINK),
    ]
    jacob = kots.jacobian(states)
    mat = rng.standard_normal((jacob.shape[0], 4))

    np.testing.assert_allclose(
        kots.jacobian_transpose_mul(states, mat),
        jacob.T @ mat,
        atol=1e-10,
        rtol=1e-10,
    )

def test_jacobian_target_mul_vector_matches_jacobian_product():
    kots = _make_kots(order=3)
    rng = np.random.default_rng(1)

    kots.set_target_from_file(str(TARGET_PATH))
    motion = rng.standard_normal(kots.order() * kots.dof())
    kots.import_motions(motion)
    kots.dynamics()

    vec = rng.standard_normal(kots.dof() * StateType.max_time_order(kots.target_._targets))
    np.testing.assert_allclose(
        kots.jacobian_target_mul(vec),
        kots.jacobian_target() @ vec,
        atol=1e-10,
        rtol=1e-10,
    )


def test_jacobian_target_mul_matrix_matches_jacobian_product():
    kots = _make_kots(order=3)
    rng = np.random.default_rng(20)

    kots.set_target_from_file(str(TARGET_PATH))
    motion = rng.standard_normal(kots.order() * kots.dof())
    kots.import_motions(motion)
    kots.dynamics()

    jacob = kots.jacobian_target()
    mat = rng.standard_normal((jacob.shape[-1], 5))
    np.testing.assert_allclose(
        kots.jacobian_target_mul(mat),
        jacob @ mat,
        atol=1e-10,
        rtol=1e-10,
    )


def test_jacobian_target_transpose_mul_vector_matches_jacobian_product():
    kots = _make_kots(order=3)
    rng = np.random.default_rng(13)

    kots.set_target_from_file(str(TARGET_PATH))
    motion = rng.standard_normal(kots.order() * kots.dof())
    kots.import_motions(motion)
    kots.dynamics()

    jacob = kots.jacobian_target()
    vec = rng.standard_normal(jacob.shape[0])
    np.testing.assert_allclose(
        kots.jacobian_target_transpose_mul(vec),
        jacob.T @ vec,
        atol=1e-10,
        rtol=1e-10,
    )


def test_jacobian_target_transpose_mul_matrix_matches_jacobian_product():
    kots = _make_kots(order=3)
    rng = np.random.default_rng(23)

    kots.set_target_from_file(str(TARGET_PATH))
    motion = rng.standard_normal(kots.order() * kots.dof())
    kots.import_motions(motion)
    kots.dynamics()

    jacob = kots.jacobian_target()
    mat = rng.standard_normal((jacob.shape[0], 5))
    np.testing.assert_allclose(
        kots.jacobian_target_transpose_mul(mat),
        jacob.T @ mat,
        atol=1e-10,
        rtol=1e-10,
    )


def test_jacobian_transpose_mul_dynamics_matches_jacobian_product():
    kots = _make_kots(order=5)
    rng = np.random.default_rng(14)

    motion = rng.standard_normal(kots.order() * kots.dof())
    kots.import_motions(motion)
    kots.dynamics()

    states = [
        StateType("link", TARGET_LINK, "snap"),
        StateType("link", TARGET_LINK, "momentum_diff3"),
        StateType("joint", "joint3", "momentum_diff3"),
        StateType("joint", "joint3", "force_diff2"),
        StateType("joint", "joint3", "torque_diff2"),
    ]
    jacob = kots.jacobian(states)
    vec = rng.standard_normal(jacob.shape[0])

    np.testing.assert_allclose(
        kots.jacobian_transpose_mul(states, vec),
        jacob.T @ vec,
        atol=1e-10,
        rtol=1e-10,
    )


def test_jacobian_transpose_mul_matrix_dynamics_matches_jacobian_product():
    kots = _make_kots(order=5)
    rng = np.random.default_rng(25)

    motion = rng.standard_normal(kots.order() * kots.dof())
    kots.import_motions(motion)
    kots.dynamics()

    states = [
        StateType("link", TARGET_LINK, "snap"),
        StateType("link", TARGET_LINK, "momentum_diff3"),
        StateType("joint", "joint3", "momentum_diff3"),
        StateType("joint", "joint3", "force_diff2"),
        StateType("joint", "joint3", "torque_diff2"),
    ]
    jacob = kots.jacobian(states)
    mat = rng.standard_normal((jacob.shape[0], 4))

    np.testing.assert_allclose(
        kots.jacobian_transpose_mul(states, mat),
        jacob.T @ mat,
        atol=1e-10,
        rtol=1e-10,
    )


def test_jacobian_transpose_mul_low_order_momentum_matches_jacobian_product():
    kots = _make_kots(order=2)
    rng = np.random.default_rng(15)

    motion = rng.standard_normal(kots.order() * kots.dof())
    kots.import_motions(motion)
    kots.dynamics()

    states = [
        StateType("link", TARGET_LINK, "vel"),
        StateType("link", TARGET_LINK, "momentum"),
        StateType("joint", "joint3", "momentum"),
    ]
    jacob = kots.jacobian(states)
    vec = rng.standard_normal(jacob.shape[0])

    np.testing.assert_allclose(
        kots.jacobian_transpose_mul(states, vec),
        jacob.T @ vec,
        atol=1e-10,
        rtol=1e-10,
    )

    mat = rng.standard_normal((jacob.shape[0], 3))
    np.testing.assert_allclose(
        kots.jacobian_transpose_mul(states, mat),
        jacob.T @ mat,
        atol=1e-10,
        rtol=1e-10,
    )


def test_total_joint_dynamics_expands_to_joint_refs_for_jacobian_apis():
    kots = _make_kots(order=5)
    rng = np.random.default_rng(26)

    motion = rng.standard_normal(kots.order() * kots.dof())
    kots.import_motions(motion)
    kots.dynamics()

    total = StateType("total_joint", "total_joint", "torque_diff2")
    joint_refs = [
        StateType("joint", joint.name, "torque_diff2")
        for joint in kots.robot_.joints
        if joint.dof > 0
    ]

    np.testing.assert_allclose(
        kots.state_info(total),
        kots.state_info_list(joint_refs).reshape(-1),
        atol=1e-12,
        rtol=1e-12,
    )

    jacob = kots.jacobian(joint_refs)
    np.testing.assert_allclose(kots.jacobian(total), jacob, atol=1e-12, rtol=1e-12)

    rhs = rng.standard_normal(jacob.shape[1])
    np.testing.assert_allclose(
        kots.jacobian_mul(total, rhs),
        kots.jacobian_mul(joint_refs, rhs),
        atol=1e-10,
        rtol=1e-10,
    )

    rhs_matrix = rng.standard_normal((jacob.shape[1], 4))
    np.testing.assert_allclose(
        kots.jacobian_mul(total, rhs_matrix),
        kots.jacobian_mul(joint_refs, rhs_matrix),
        atol=1e-10,
        rtol=1e-10,
    )

    transpose_rhs = rng.standard_normal(jacob.shape[0])
    np.testing.assert_allclose(
        kots.jacobian_transpose_mul(total, transpose_rhs),
        kots.jacobian_transpose_mul(joint_refs, transpose_rhs),
        atol=1e-10,
        rtol=1e-10,
    )

    transpose_rhs_matrix = rng.standard_normal((jacob.shape[0], 3))
    np.testing.assert_allclose(
        kots.jacobian_transpose_mul(total, transpose_rhs_matrix),
        kots.jacobian_transpose_mul(joint_refs, transpose_rhs_matrix),
        atol=1e-10,
        rtol=1e-10,
    )

    state_tensor = kots.state_tensor(total)
    jacobian_tensor = kots.jacobian_tensor(total)
    assert [st.owner_name for st in state_tensor.state_types] == [st.owner_name for st in joint_refs]
    assert [st.owner_name for st in jacobian_tensor.state_types] == [st.owner_name for st in joint_refs]


def test_total_joint_motion_and_torque_derivatives_to_jerk_and_torque_diff2():
    pytest.importorskip("robokots._rust")
    order = 5
    kots = _make_kots(order=order)
    rng = np.random.default_rng(51)
    motion = rng.standard_normal(kots.order() * kots.dof())
    kots.import_motions(motion)
    kots.dynamics(order=order, backend="rust", materialize_dict=False)

    data_types = ["coord", "veloc", "accel", "jerk", "torque", "torque_diff1", "torque_diff2"]
    total_states = [StateType("total_joint", "total_joint", dt) for dt in data_types]
    joint_refs = [
        StateType("joint", joint.name, dt)
        for dt in data_types
        for joint in kots.robot_.joints
        if joint.dof > 0
    ]

    np.testing.assert_allclose(
        kots.state_info_list(total_states).reshape(-1),
        kots.state_info_list(joint_refs).reshape(-1),
        atol=1e-12,
        rtol=1e-12,
    )

    expected_motion = np.concatenate([motion[i::order] for i in range(4)])
    np.testing.assert_allclose(
        kots.state_info_list(total_states[:4]).reshape(-1),
        expected_motion,
        atol=0.0,
        rtol=0.0,
    )

    jacob = kots.jacobian(total_states)
    assert jacob.shape == (len(data_types) * kots.dof(), order * kots.dof())

    rhs = rng.standard_normal(jacob.shape[1])
    np.testing.assert_allclose(
        kots.jacobian_mul(total_states, rhs),
        jacob @ rhs,
        atol=1e-10,
        rtol=1e-10,
    )

    rhs_matrix = rng.standard_normal((jacob.shape[1], 3))
    np.testing.assert_allclose(
        kots.jacobian_mul(total_states, rhs_matrix),
        jacob @ rhs_matrix,
        atol=1e-10,
        rtol=1e-10,
    )

    transpose_rhs = rng.standard_normal(jacob.shape[0])
    np.testing.assert_allclose(
        kots.jacobian_transpose_mul(total_states, transpose_rhs),
        jacob.T @ transpose_rhs,
        atol=1e-10,
        rtol=1e-10,
    )

    transpose_rhs_matrix = rng.standard_normal((jacob.shape[0], 2))
    np.testing.assert_allclose(
        kots.jacobian_transpose_mul(total_states, transpose_rhs_matrix),
        jacob.T @ transpose_rhs_matrix,
        atol=1e-10,
        rtol=1e-10,
    )


def test_total_joint_target_from_file_expands_to_joint_refs(tmp_path: Path):
    target_path = tmp_path / "total_joint_target.json"
    target_path.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "data_type": "torque_diff2",
                        "owner_type": "total_joint",
                        "owner_name": "total_joint",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    kots = _make_kots(order=5)
    rng = np.random.default_rng(27)
    kots.set_target_from_file(str(target_path))
    kots.import_motions(rng.standard_normal(kots.order() * kots.dof()))
    kots.dynamics()

    joint_refs = [
        StateType("joint", joint.name, "torque_diff2")
        for joint in kots.robot_.joints
        if joint.dof > 0
    ]

    assert [st.owner_type for st in kots.target_._targets] == ["joint"] * len(joint_refs)
    assert [st.owner_name for st in kots.target_._targets] == [st.owner_name for st in joint_refs]

    np.testing.assert_allclose(kots.target_state_info().reshape(-1), kots.state_info_list(joint_refs).reshape(-1))
    np.testing.assert_allclose(kots.jacobian_target(), kots.jacobian(joint_refs), atol=1e-12, rtol=1e-12)


def test_total_joint_target_requires_active_joint_names_for_direct_target_list():
    with pytest.raises(ValueError, match="active_joint_names"):
        TargetList.from_dict(
            {
                "targets": [
                    {
                        "data_type": "torque_diff2",
                        "owner_type": "total_joint",
                        "owner_name": "total_joint",
                    }
                ]
            },
            RobotNames(["root", "joint1"], ["world", "link1"]),
        )


def test_batched_kinematics_matches_loop():
    order = 3
    kots = _make_kots(order=order)
    rng = np.random.default_rng(2)
    motions = rng.standard_normal((4, kots.dof() * order))

    kots.import_motions(motions)
    kots.kinematics()

    frame_state = StateType(data_type="frame", owner_type="link", owner_name=TARGET_LINK)
    acc_state = StateType(data_type="acc", owner_type="link", owner_name=TARGET_LINK)
    actual_frames = kots.state_info(frame_state)
    actual_acc = kots.state_info(acc_state)
    actual_jacob = kots.jacobian(acc_state)
    vec = rng.standard_normal(kots.dof() * order)
    actual_matvec = kots.jacobian_mul(acc_state, vec)

    assert actual_frames.shape == (4, 4, 4)
    assert actual_acc.shape == (4, 6)
    assert actual_jacob.shape == (4, 6, kots.dof() * order)
    assert actual_matvec.shape == (4, 6)

    for i, motion in enumerate(motions):
        single = _make_kots(order=order)
        single.import_motions(motion)
        single.kinematics()
        np.testing.assert_allclose(actual_frames[i], single.state_info(frame_state).mat())
        np.testing.assert_allclose(actual_acc[i], single.state_info(acc_state))
        np.testing.assert_allclose(actual_jacob[i], single.jacobian(acc_state))
        np.testing.assert_allclose(actual_matvec[i], single.jacobian(acc_state) @ vec)


def test_motion_derivative_api_aliases_old_motion_diff():
    kots = _make_kots(order=3)
    rng = np.random.default_rng(5)
    motion = rng.standard_normal(kots.dof() * kots.order())
    tail = rng.standard_normal(kots.dof())
    kots.import_motions(motion)

    np.testing.assert_allclose(kots.motion_derivative(tail=tail), kots.motion_diff(last_diff=tail))
    np.testing.assert_allclose(kots.motion_derivative_cm(tail=tail), kots.motion_diff_cm(last_diff=tail))


def test_motion_array_api_roundtrips_flat_backend_layout():
    kots = _make_kots(order=3)
    rng = np.random.default_rng(9)
    motion_array = rng.standard_normal((2, kots.dof(), kots.order()))
    tail = rng.standard_normal((2, kots.dof()))

    kots.import_motion_array(motion_array)

    np.testing.assert_allclose(kots.motion_array(), motion_array)
    np.testing.assert_allclose(kots.motion_tensor().as_dof_order().data, motion_array)

    flat = kots.motion()
    single = _make_kots(order=3)
    single.import_motions(flat)
    np.testing.assert_allclose(single.motion_array(), motion_array)

    np.testing.assert_allclose(kots.motion_derivative_array(tail=tail), single.motion_derivative_array(tail=tail))
    np.testing.assert_allclose(kots.motion_cm_array(), single.motion_cm_array())
    np.testing.assert_allclose(kots.motion_derivative_cm_array(tail=tail), single.motion_derivative_cm_array(tail=tail))


def test_batched_dynamics_and_jacobian_matches_loop():
    order = 5
    kots = _make_kots(order=order)
    rng = np.random.default_rng(3)
    motions = rng.standard_normal((3, kots.dof() * order))

    kots.import_motions(motions)
    kots.dynamics()

    force_state = StateType("link", TARGET_LINK, "force")
    torque_state = StateType("joint", "joint3", "torque")
    actual_force = kots.state_info(force_state)
    actual_jacob = kots.jacobian(torque_state)

    assert actual_force.shape == (3, 6)
    assert actual_jacob.shape[0] == 3
    assert actual_jacob.shape[2] == kots.dof() * StateType.max_time_order([torque_state])

    for i, motion in enumerate(motions):
        single = _make_kots(order=order)
        single.import_motions(motion)
        single.dynamics()
        np.testing.assert_allclose(actual_force[i], single.state_info(force_state))
        np.testing.assert_allclose(actual_jacob[i], single.jacobian(torque_state), atol=1e-12)


def test_batched_dynamics_jacobian_mul_vector_matches_jacobian_product():
    order = 5
    kots = _make_kots(order=order)
    rng = np.random.default_rng(11)
    batch_shape = (2, 3)
    motions = rng.standard_normal(batch_shape + (kots.dof() * order,))
    states = [
        StateType("link", TARGET_LINK, "momentum"),
        StateType("link", TARGET_LINK, "force"),
        StateType("joint", "joint3", "momentum"),
        StateType("joint", "joint3", "torque"),
    ]
    vecs = rng.standard_normal(batch_shape + (kots.dof() * StateType.max_time_order(states),))

    kots.import_motions(motions)
    kots.dynamics()

    actual = kots.jacobian_mul(states, vecs)
    expected = (kots.jacobian(states) @ vecs[..., None])[..., 0]
    parts = kots.jacobian_mul(states, vecs, list_output=True)

    assert actual.shape == expected.shape
    assert len(parts) == len(states)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(actual, np.concatenate(parts, axis=-1), atol=1e-10, rtol=1e-10)


def test_multidimensional_batched_kinematics_keeps_prefix_shape():
    order = 3
    kots = _make_kots(order=order)
    rng = np.random.default_rng(4)
    motions = rng.standard_normal((2, 3, kots.dof() * order))

    kots.import_motions(motions)
    kots.kinematics()

    acc_state = StateType(data_type="acc", owner_type="link", owner_name=TARGET_LINK)
    actual_acc = kots.state_info(acc_state)
    actual_jacob = kots.jacobian(acc_state)

    assert actual_acc.shape == (2, 3, 6)
    assert actual_jacob.shape == (2, 3, 6, kots.dof() * order)

    single = _make_kots(order=order)
    single.import_motions(motions[1, 2])
    single.kinematics()

    np.testing.assert_allclose(actual_acc[1, 2], single.state_info(acc_state))
    np.testing.assert_allclose(actual_jacob[1, 2], single.jacobian(acc_state))


def test_batched_state_info_list_shape_contract():
    order = 3
    kots = _make_kots(order=order)
    rng = np.random.default_rng(6)
    motions = rng.standard_normal((2, 3, kots.dof() * order))
    states = [
        StateType(data_type="vel", owner_type="link", owner_name=TARGET_LINK),
        StateType(data_type="acc", owner_type="link", owner_name=TARGET_LINK),
    ]

    kots.import_motions(motions)
    kots.kinematics()

    stacked = kots.state_info_list(states)
    parts = kots.state_info_list(states, list_output=True)

    assert stacked.shape == (2, 3, 12)
    assert len(parts) == 2
    assert parts[0].shape == (2, 3, 6)
    assert parts[1].shape == (2, 3, 6)
    np.testing.assert_allclose(stacked, np.concatenate(parts, axis=-1))

    single = _make_kots(order=order)
    single.import_motions(motions[1, 2])
    single.kinematics()
    np.testing.assert_allclose(stacked[1, 2], single.state_info_list(states).reshape(-1))


def test_batched_jacobian_mul_vector_shape_contract_with_per_sample_vecs():
    order = 3
    kots = _make_kots(order=order)
    rng = np.random.default_rng(7)
    motions = rng.standard_normal((2, 3, kots.dof() * order))
    states = [
        StateType(data_type="vel", owner_type="link", owner_name=TARGET_LINK),
        StateType(data_type="acc", owner_type="link", owner_name=TARGET_LINK),
    ]
    vecs = rng.standard_normal((2, 3, kots.dof() * StateType.max_time_order(states)))

    kots.import_motions(motions)
    kots.kinematics()

    matvec = kots.jacobian_mul(states, vecs)
    parts = kots.jacobian_mul(states, vecs, list_output=True)

    assert matvec.shape == (2, 3, 12)
    assert len(parts) == 2
    assert parts[0].shape == (2, 3, 6)
    assert parts[1].shape == (2, 3, 6)
    np.testing.assert_allclose(matvec, np.concatenate(parts, axis=-1))

    single = _make_kots(order=order)
    single.import_motions(motions[1, 2])
    single.kinematics()
    np.testing.assert_allclose(matvec[1, 2], single.jacobian(states) @ vecs[1, 2])


def test_batched_jacobian_mul_matrix_shape_contract_with_per_sample_mats():
    order = 3
    kots = _make_kots(order=order)
    rng = np.random.default_rng(21)
    motions = rng.standard_normal((2, 3, kots.dof() * order))
    states = [
        StateType(data_type="vel", owner_type="link", owner_name=TARGET_LINK),
        StateType(data_type="acc", owner_type="link", owner_name=TARGET_LINK),
    ]
    mats = rng.standard_normal((2, 3, kots.dof() * StateType.max_time_order(states), 4))

    kots.import_motions(motions)
    kots.kinematics()

    matmul = kots.jacobian_mul(states, mats)
    parts = kots.jacobian_mul(states, mats, list_output=True)
    expected = kots.jacobian(states) @ mats

    assert matmul.shape == (2, 3, 12, 4)
    assert len(parts) == 2
    assert parts[0].shape == (2, 3, 6, 4)
    assert parts[1].shape == (2, 3, 6, 4)
    np.testing.assert_allclose(matmul, expected, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(matmul, np.concatenate(parts, axis=-2), atol=1e-10, rtol=1e-10)

    shared = rng.standard_normal((kots.dof() * StateType.max_time_order(states), 2))
    shared_matmul = kots.jacobian_mul(states, shared)
    np.testing.assert_allclose(shared_matmul, kots.jacobian(states) @ shared, atol=1e-10, rtol=1e-10)


def test_batched_jacobian_transpose_mul_vector_shape_contract_with_per_sample_vecs():
    order = 3
    kots = _make_kots(order=order)
    rng = np.random.default_rng(16)
    motions = rng.standard_normal((2, 3, kots.dof() * order))
    states = [
        StateType(data_type="vel", owner_type="link", owner_name=TARGET_LINK),
        StateType(data_type="acc", owner_type="link", owner_name=TARGET_LINK),
    ]

    kots.import_motions(motions)
    kots.kinematics()

    jacob = kots.jacobian(states)
    vecs = rng.standard_normal(jacob.shape[:-2] + (jacob.shape[-2],))
    transpose_matvec = kots.jacobian_transpose_mul(states, vecs)

    assert transpose_matvec.shape == (2, 3, kots.dof() * order)
    np.testing.assert_allclose(
        transpose_matvec,
        (np.swapaxes(jacob, -1, -2) @ vecs[..., None])[..., 0],
        atol=1e-10,
        rtol=1e-10,
    )

    single = _make_kots(order=order)
    single.import_motions(motions[1, 2])
    single.kinematics()
    np.testing.assert_allclose(transpose_matvec[1, 2], single.jacobian(states).T @ vecs[1, 2])


def test_batched_jacobian_transpose_mul_matrix_shape_contract_with_per_sample_mats():
    order = 3
    kots = _make_kots(order=order)
    rng = np.random.default_rng(24)
    motions = rng.standard_normal((2, 3, kots.dof() * order))
    states = [
        StateType(data_type="vel", owner_type="link", owner_name=TARGET_LINK),
        StateType(data_type="acc", owner_type="link", owner_name=TARGET_LINK),
    ]

    kots.import_motions(motions)
    kots.kinematics()

    jacob = kots.jacobian(states)
    mats = rng.standard_normal(jacob.shape[:-2] + (jacob.shape[-2], 4))
    transpose_matmul = kots.jacobian_transpose_mul(states, mats)
    expected = np.swapaxes(jacob, -1, -2) @ mats

    assert transpose_matmul.shape == (2, 3, kots.dof() * order, 4)
    np.testing.assert_allclose(transpose_matmul, expected, atol=1e-10, rtol=1e-10)
    shared = rng.standard_normal((jacob.shape[-2], 2))
    shared_matmul = kots.jacobian_transpose_mul(states, shared)
    np.testing.assert_allclose(
        shared_matmul,
        np.swapaxes(jacob, -1, -2) @ shared,
        atol=1e-10,
        rtol=1e-10,
    )

    single = _make_kots(order=order)
    single.import_motions(motions[1, 2])
    single.kinematics()
    np.testing.assert_allclose(transpose_matmul[1, 2], single.jacobian(states).T @ mats[1, 2])


def test_batched_target_api_shape_contract():
    kots = _make_kots(order=3)
    rng = np.random.default_rng(8)
    kots.set_target_from_file(str(TARGET_PATH))
    order = kots.order()
    batch_shape = (2, 2)
    target_dim = 9 * 6
    motion_dim = kots.dof() * order
    motions = rng.standard_normal(batch_shape + (motion_dim,))
    shared_vec = rng.standard_normal(motion_dim)
    sample_vecs = rng.standard_normal(batch_shape + (motion_dim,))

    kots.import_motions(motions)
    kots.dynamics()

    target_values = kots.target_state_info()
    target_parts = kots.target_state_info(list_output=True)
    target_jacobian = kots.jacobian_target()
    shared_matvec = kots.jacobian_target_mul(shared_vec)
    sample_matvec = kots.jacobian_target_mul(sample_vecs)
    matvec_parts = kots.jacobian_target_mul(sample_vecs, list_output=True)
    shared_transpose_vec = rng.standard_normal(target_dim)
    shared_transpose_matvec = kots.jacobian_target_transpose_mul(shared_transpose_vec)
    transpose_vecs = rng.standard_normal(batch_shape + (target_dim,))
    sample_transpose_matvec = kots.jacobian_target_transpose_mul(transpose_vecs)
    shared_transpose_mat = rng.standard_normal((target_dim, 3))
    shared_transpose_matmul = kots.jacobian_target_transpose_mul(shared_transpose_mat)
    transpose_mats = rng.standard_normal(batch_shape + (target_dim, 2))
    sample_transpose_matmul = kots.jacobian_target_transpose_mul(transpose_mats)

    assert target_values.shape == batch_shape + (target_dim,)
    assert len(target_parts) == 9
    assert all(part.shape == batch_shape + (6,) for part in target_parts)
    np.testing.assert_allclose(target_values, np.concatenate(target_parts, axis=-1))

    assert target_jacobian.shape == batch_shape + (target_dim, motion_dim)
    assert shared_matvec.shape == batch_shape + (target_dim,)
    assert sample_matvec.shape == batch_shape + (target_dim,)
    assert len(matvec_parts) == 9
    assert all(part.shape == batch_shape + (6,) for part in matvec_parts)
    np.testing.assert_allclose(sample_matvec, np.concatenate(matvec_parts, axis=-1))
    assert shared_transpose_matvec.shape == batch_shape + (motion_dim,)
    assert sample_transpose_matvec.shape == batch_shape + (motion_dim,)
    np.testing.assert_allclose(
        shared_transpose_matvec,
        (np.swapaxes(target_jacobian, -1, -2) @ shared_transpose_vec[..., None])[..., 0],
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        sample_transpose_matvec,
        (np.swapaxes(target_jacobian, -1, -2) @ transpose_vecs[..., None])[..., 0],
        atol=1e-10,
        rtol=1e-10,
    )
    assert shared_transpose_matmul.shape == batch_shape + (motion_dim, 3)
    assert sample_transpose_matmul.shape == batch_shape + (motion_dim, 2)
    np.testing.assert_allclose(
        shared_transpose_matmul,
        np.swapaxes(target_jacobian, -1, -2) @ shared_transpose_mat,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        sample_transpose_matmul,
        np.swapaxes(target_jacobian, -1, -2) @ transpose_mats,
        atol=1e-10,
        rtol=1e-10,
    )

    single = _make_kots(order=3)
    single.set_target_from_file(str(TARGET_PATH))
    single.import_motions(motions[1, 1])
    single.dynamics()

    np.testing.assert_allclose(target_values[1, 1], single.target_state_info().reshape(-1))
    np.testing.assert_allclose(target_jacobian[1, 1], single.jacobian_target())
    np.testing.assert_allclose(shared_matvec[1, 1], single.jacobian_target() @ shared_vec)
    np.testing.assert_allclose(sample_matvec[1, 1], single.jacobian_target() @ sample_vecs[1, 1])


def test_batched_state_and_jacobian_tensor_shape_contract():
    order = 3
    kots = _make_kots(order=order)
    rng = np.random.default_rng(10)
    motions = rng.standard_normal((2, 3, kots.dof() * order))
    states = [
        StateType(data_type="vel", owner_type="link", owner_name=TARGET_LINK),
        StateType(data_type="acc", owner_type="link", owner_name=TARGET_LINK),
    ]

    kots.import_motions(motions)
    kots.kinematics()

    state_tensor = kots.state_tensor(states)
    jacobian_tensor = kots.jacobian_tensor(states)

    assert state_tensor.axes == ("batch0", "batch1", "state")
    assert state_tensor.shape == (2, 3, 12)
    assert state_tensor.batch_shape == (2, 3)
    assert state_tensor.state_dim == 12
    np.testing.assert_allclose(state_tensor.data, kots.state_info_list(states))

    assert jacobian_tensor.axes == ("batch0", "batch1", "state", "motion")
    assert jacobian_tensor.shape == (2, 3, 12, kots.dof() * order)
    assert jacobian_tensor.batch_shape == (2, 3)
    assert jacobian_tensor.state_dim == 12
    assert jacobian_tensor.motion_dim == kots.dof() * order
    np.testing.assert_allclose(jacobian_tensor.data, kots.jacobian(states))


def test_unbatched_state_tensor_flattens_state_list():
    kots = _make_kots(order=3)
    states = [
        StateType(data_type="vel", owner_type="link", owner_name=TARGET_LINK),
        StateType(data_type="acc", owner_type="link", owner_name=TARGET_LINK),
    ]

    kots.import_motions(np.zeros(kots.dof() * kots.order(), dtype=float))
    kots.kinematics()

    state_tensor = kots.state_tensor(states)
    jacobian_tensor = kots.jacobian_tensor(states)

    assert state_tensor.axes == ("state",)
    assert state_tensor.shape == (12,)
    np.testing.assert_allclose(state_tensor.data, kots.state_info_list(states).reshape(-1))
    assert jacobian_tensor.axes == ("state", "motion")
    assert jacobian_tensor.shape == (12, kots.dof() * kots.order())


def test_batched_target_tensor_shape_contract():
    kots = _make_kots(order=3)
    rng = np.random.default_rng(11)
    kots.set_target_from_file(str(TARGET_PATH))
    order = kots.order()
    batch_shape = (2, 2)
    motion_dim = kots.dof() * order
    target_dim = 9 * 6
    kots.import_motions(rng.standard_normal(batch_shape + (motion_dim,)))
    kots.dynamics()

    state_tensor = kots.target_state_tensor()
    jacobian_tensor = kots.jacobian_target_tensor()

    assert state_tensor.axes == ("batch0", "batch1", "state")
    assert state_tensor.shape == batch_shape + (target_dim,)
    np.testing.assert_allclose(state_tensor.data, kots.target_state_info())
    assert jacobian_tensor.axes == ("batch0", "batch1", "state", "motion")
    assert jacobian_tensor.shape == batch_shape + (target_dim, motion_dim)
    np.testing.assert_allclose(jacobian_tensor.data, kots.jacobian_target())


def test_batch_unsupported_kots_apis_raise_clear_errors():
    kots = _make_kots(order=3)
    motion = np.zeros((2, kots.dof() * kots.order()), dtype=float)
    state = StateType(data_type="acc", owner_type="link", owner_name=TARGET_LINK)

    kots.import_motions(motion)
    kots.kinematics()

    with pytest.raises(ValueError, match="state_df.*batched"):
      kots.state_df()
    with pytest.raises(ValueError, match="set_state_df.*batched"):
      kots.set_state_df()
    with pytest.raises(ValueError, match="kinematics_point.*batched"):
      kots.kinematics_point()
    with pytest.raises(ValueError, match="link_diff_kinematics_numerical.*batched"):
      kots.link_diff_kinematics_numerical([TARGET_LINK], "vel")
    with pytest.raises(ValueError, match="diff_outward_numerical.*batched"):
      kots.diff_outward_numerical(state)


def test_import_motions_invalidates_previous_batched_state():
    kots = _make_kots(order=3)
    rng = np.random.default_rng(12)
    kots.import_motions(rng.standard_normal((2, kots.dof() * kots.order())))
    kots.kinematics()

    assert kots.state_batch_ is not None
    assert kots.batch_shape_ == (2,)

    single_motion = rng.standard_normal(kots.dof() * kots.order())
    kots.import_motions(single_motion)

    assert kots.state_batch_ is None
    assert kots.outward_state_ is None
    assert kots.state_dict_ == {}
    assert kots.batch_shape_ == ()

    kots.kinematics()
    acc_state = StateType(data_type="acc", owner_type="link", owner_name=TARGET_LINK)
    assert kots.state_info(acc_state).shape == (6,)
    
# def test_cmtm_jacobian_numerical_soft():
#     kots = Kots.from_json_file("./test_model/soft_rod.json", order=5)

#     motion = np.random.rand(kots.order()*kots.dof())

#     kots.import_motions(motion)

#     kots.kinematics()  

#     jacob_cmtm = kots.jacobian(StateType('link','end','snap'))
#     jacob_cmtm_num = kots.jacobian(StateType('link','end','snap'), numerical=True)
#     print("Analytical Jacobian:\n", jacob_cmtm.shape)
#     print("Numerical Jacobian:\n", jacob_cmtm_num.shape)

#     assert np.allclose(jacob_cmtm, jacob_cmtm_num, atol=1e-5, rtol=1e-5)
