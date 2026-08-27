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
BRANCHED_FIXED_MODEL_PATH = TEST_DIR / "test_model" / "branched_fixed.urdf"
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


@pytest.mark.parametrize(("order", "data_type"), [(3, "torque"), (4, "torque_diff1")])
@pytest.mark.parametrize("add_world_link", [True, False])
def test_branched_fixed_torque_jacobian_matches_numerical_and_products(
    order,
    data_type,
    add_world_link,
):
    rng = np.random.default_rng(35 + order)
    kots = Kots.from_urdf_file(
        str(BRANCHED_FIXED_MODEL_PATH),
        order=order,
        add_world_link=add_world_link,
    )
    child_link_ids = [joint.child_link_id for joint in kots.robot_.joints]
    assert child_link_ids != sorted(child_link_ids)

    kots.import_motions(rng.standard_normal(order * kots.dof()))
    kots.dynamics(backend="numpy")
    states = [
        StateType("joint", joint.name, data_type)
        for joint in kots.robot_.joints
        if joint.dof
    ]

    jacobian = kots.jacobian(states)
    numerical = kots.jacobian(states, numerical=True)
    np.testing.assert_allclose(jacobian, numerical, atol=1e-6, rtol=1e-6)

    rhs = rng.standard_normal(order * kots.dof())
    rhs_matrix = rng.standard_normal((order * kots.dof(), 2))
    lhs = rng.standard_normal(kots.dof())
    np.testing.assert_allclose(kots.jacobian_mul(states, rhs), jacobian @ rhs, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        kots.jacobian_mul(states, rhs_matrix),
        jacobian @ rhs_matrix,
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        kots.jacobian_transpose_mul(states, lhs),
        jacobian.T @ lhs,
        atol=1e-12,
        rtol=1e-12,
    )


def test_branched_fixed_batched_torque_diff1_jacobian_matches_scalar_loop():
    rng = np.random.default_rng(40)
    order = 4
    kots = Kots.from_urdf_file(str(BRANCHED_FIXED_MODEL_PATH), order=order)
    motions = rng.standard_normal((3, kots.dof() * order))
    kots.import_motions(motions)
    kots.dynamics(backend="numpy")
    states = [
        StateType("joint", joint.name, "torque_diff1")
        for joint in kots.robot_.joints
        if joint.dof
    ]

    actual = kots.jacobian(states)
    expected = []
    for motion in motions:
        scalar = Kots.from_urdf_file(str(BRANCHED_FIXED_MODEL_PATH), order=order)
        scalar.import_motions(motion)
        scalar.dynamics(backend="numpy")
        expected.append(scalar.jacobian(states))

    np.testing.assert_allclose(actual, np.stack(expected), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(actual, kots.jacobian(states, numerical=True), atol=1e-6, rtol=1e-6)
    rhs = rng.standard_normal((motions.shape[0], order * kots.dof()))
    lhs = rng.standard_normal((motions.shape[0], kots.dof()))
    np.testing.assert_allclose(
        kots.jacobian_mul(states, rhs),
        (actual @ rhs[..., None])[..., 0],
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        kots.jacobian_transpose_mul(states, lhs),
        (np.swapaxes(actual, -1, -2) @ lhs[..., None])[..., 0],
        atol=1e-12,
        rtol=1e-12,
    )


def test_branched_fixed_rust_torque_diff1_jacobian_matches_numerical():
    pytest.importorskip("robokots._rust")
    rng = np.random.default_rng(41)
    order = 4
    kots = Kots.from_urdf_file(str(BRANCHED_FIXED_MODEL_PATH), order=order)
    kots.import_motions(rng.standard_normal(order * kots.dof()))
    kots.dynamics(backend="rust")
    states = [
        StateType("joint", joint.name, "torque_diff1")
        for joint in kots.robot_.joints
        if joint.dof
    ]

    np.testing.assert_allclose(
        kots.jacobian(states),
        kots.jacobian(states, numerical=True),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize("batched", [False, True])
def test_rust_cmtm_torque_diff1_jacobian_mul_uses_analytic_tangent_with_gravity(monkeypatch, batched):
    """Higher-order torque Jv must not fall back to Python outward CMTM."""
    pytest.importorskip("robokots._rust")
    rng = np.random.default_rng(411 + int(batched))
    order = 4
    kots = Kots.from_urdf_file(str(BRANCHED_FIXED_MODEL_PATH), order=order)
    sample_count = 3 if batched else None
    motion = rng.standard_normal(((sample_count,) if batched else ()) + (order * kots.dof(),))
    kots.import_motions(motion)
    kots.dynamics(backend="rust", gravity=(0.3, -0.8, -9.81))
    states = [StateType("total_joint", "total_joint", "torque_diff1")]
    jacobian = kots.jacobian(states)
    rhs = rng.standard_normal(((sample_count,) if batched else ()) + (order * kots.dof(), 2))
    expected = jacobian @ rhs

    def fail_outward(*args, **kwargs):
        raise AssertionError("higher-order Rust Jv fell back to outward CMTM")

    monkeypatch.setattr(outward_api, "outward_jacobian_matmul_rhs", fail_outward)
    actual = kots.jacobian_mul(states, rhs)
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)
    monkeypatch.setattr(outward_api, "outward_jacobian_matvec", fail_outward)
    np.testing.assert_allclose(
        kots.jacobian_mul(states, rhs[..., 0]),
        expected[..., 0],
        atol=1e-9,
        rtol=1e-9,
    )


@pytest.mark.parametrize("batched", [False, True])
def test_rust_cmtm_torque_diff1_vjp_uses_analytic_reverse_with_gravity(monkeypatch, batched):
    """The CMTM torque VJP must win over Python outward reverse mode.

    This is deliberately an API-forward test: source builds which predate the
    Rust reverse kernel still run the normal fallback and return here, while a
    build exporting the kernel verifies both vector and multi-RHS dispatch.
    """
    pytest.importorskip("robokots._rust")
    rng = np.random.default_rng(412 + int(batched))
    order = 4
    kots = Kots.from_urdf_file(str(BRANCHED_FIXED_MODEL_PATH), order=order)
    sample_count = 3 if batched else None
    motion = rng.standard_normal(((sample_count,) if batched else ()) + (order * kots.dof(),))
    kots.import_motions(motion)
    kots.dynamics(backend="rust", gravity=(0.3, -0.8, -9.81))
    states = [StateType("total_joint", "total_joint", "torque_diff1")]
    robot = kots._rust_compiled_robot()
    if not hasattr(robot, "dynamics_joint_torque_series_transpose_matmul_rhs"):
      # Keep this regression test usable during staged Rust/Python rollouts;
      # it automatically becomes a strict fast-path test once the extension
      # exposes the reverse kernel.
      return

    jacobian = kots.jacobian(states)
    output_dim = jacobian.shape[-2]
    vector = rng.standard_normal(((sample_count,) if batched else ()) + (output_dim,))
    matrix = rng.standard_normal(((sample_count,) if batched else ()) + (output_dim, 2))

    def fail_outward(*args, **kwargs):
      raise AssertionError("higher-order Rust VJP fell back to outward CMTM")

    monkeypatch.setattr(outward_api, "outward_jacobian_transpose_matvec", fail_outward)
    np.testing.assert_allclose(
      kots.jacobian_transpose_mul(states, vector),
      (np.swapaxes(jacobian, -1, -2) @ vector[..., None])[..., 0],
      atol=2e-9,
      rtol=2e-9,
    )
    np.testing.assert_allclose(
      kots.jacobian_transpose_mul(states, matrix),
      np.swapaxes(jacobian, -1, -2) @ matrix,
      atol=2e-9,
      rtol=2e-9,
    )


@pytest.mark.parametrize("gravity", [np.zeros(3), np.array([0.4, -1.1, -9.3])])
def test_rust_cmtm_full_torque_series_reverse_matches_basis_tangent(gravity):
    """Cover every URDF joint row, including fixed rows which are zero."""
    pytest.importorskip("robokots._rust")
    rng = np.random.default_rng(416)
    order = 5
    dynamics_order = order - 2
    rhs_cols = 3
    kots = Kots.from_urdf_file(str(BRANCHED_FIXED_MODEL_PATH), order=order)
    robot = kots._rust_compiled_robot()
    motion = rng.standard_normal(kots.dof() * order)
    cotangent = rng.standard_normal((robot.joint_num, dynamics_order, rhs_cols))
    actual = robot.dynamics_joint_torque_series_transpose_matmul_rhs(
        motion, cotangent, dynamics_order, gravity=gravity,
    )
    basis = np.eye(motion.size)
    tangent = robot.dynamics_joint_torque_series_tangent(
        motion, basis, dynamics_order, gravity=gravity,
    )
    expected = np.einsum("jti,jtr->ir", tangent, cotangent)
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("batched", [False, True])
def test_rust_cmtm_local_momentum_force_wrench_vjp_dispatches(monkeypatch, batched):
    pytest.importorskip("robokots._rust")
    rng = np.random.default_rng(413 + int(batched))
    order = 4
    kots = _make_kots(order=order)
    batch = (2,) if batched else ()
    kots.import_motions(rng.standard_normal(batch + (order * kots.dof(),)))
    kots.dynamics(backend="rust", gravity=(0.3, -0.8, -9.81))
    states = [
        StateType("link", "arm3", "momentum_diff1"),
        StateType("link", "arm3", "force_diff1"),
        StateType("joint", "joint3", "momentum"),
        StateType("joint", "joint3", "force"),
        StateType("joint", "joint3", "torque_diff1"),
    ]
    jacobian = kots.jacobian(states)
    rhs = rng.standard_normal(batch + (jacobian.shape[-2], 2))

    def fail_outward(*args, **kwargs):
        raise AssertionError("local CMTM VJP fell back to Python outward reverse mode")

    monkeypatch.setattr(outward_api, "outward_jacobian_transpose_matvec", fail_outward)
    actual = kots.jacobian_transpose_mul(states, rhs)
    np.testing.assert_allclose(
        actual,
        np.swapaxes(jacobian, -1, -2) @ rhs,
        atol=2e-9,
        rtol=2e-9,
    )


def test_rust_cmtm_link_kinematics_vjp_dispatches():
    pytest.importorskip("robokots._rust")
    rng = np.random.default_rng(414)
    order = 4
    kots = _make_kots(order=order)
    motion = rng.standard_normal(order * kots.dof())
    kots.import_motions(motion)
    kots.kinematics(backend="rust")
    state = StateType("link", "arm3", "jerk")
    rhs = rng.standard_normal((6, 2))

    actual = kots.jacobian_transpose_mul(state, rhs)
    robot = kots._rust_compiled_robot()
    _, link_vec_tangent = robot.cmtm_kinematics_tangent(
        motion, np.eye(order * kots.dof()), order,
    )
    link_id = next(i for i, link in enumerate(kots.robot_.links) if link.name == "arm3")
    expected = link_vec_tangent[link_id, 2].T @ rhs
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_rust_cmtm_mixed_local_kinematics_and_dynamics_vjp_composes():
    pytest.importorskip("robokots._rust")
    rng = np.random.default_rng(415)
    order = 4
    kots = _make_kots(order=order)
    kots.import_motions(rng.standard_normal(order * kots.dof()))
    kots.dynamics(backend="rust")
    kinematic = StateType("link", "arm3", "jerk")
    dynamic = StateType("joint", "joint3", "force_diff1")
    rhs = rng.standard_normal((12, 2))

    actual = kots.jacobian_transpose_mul([kinematic, dynamic], rhs)
    expected = (
        kots.jacobian_transpose_mul(kinematic, rhs[:6])
        + kots.jacobian_transpose_mul(dynamic, rhs[6:])
    )
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("data_type", ["force", "force_diff1"])
def test_world_link_force_dense_jvp_and_vjp_match_independent_difference(data_type):
    pytest.importorskip("robokots._rust")
    rng = np.random.default_rng(416 + (data_type == "force_diff1"))
    order = 4
    kots = _make_kots(order=order)
    motion = rng.standard_normal(order * kots.dof())
    kots.import_motions(motion)
    kots.dynamics(backend="rust")
    state = StateType("link", "arm3", data_type, "world")
    state_order = state.time_order
    reduced_motion = motion.reshape(kots.dof(), order)[:, :state_order].reshape(-1)
    direction = rng.standard_normal(reduced_motion.shape)
    eps = 1e-7

    def value_at(reduced):
        full = motion.reshape(kots.dof(), order).copy()
        full[:, :state_order] = reduced.reshape(kots.dof(), state_order)
        sample = _make_kots(order=order)
        sample.import_motions(full.reshape(-1))
        sample.dynamics(backend="rust")
        return np.asarray(sample.state_info(state)).reshape(-1)

    finite = (
        value_at(reduced_motion + eps * direction)
        - value_at(reduced_motion - eps * direction)
    ) / (2.0 * eps)
    jacobian = kots.jacobian(state)
    np.testing.assert_allclose(jacobian @ direction, finite, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(
        kots.jacobian_mul(state, direction), finite, atol=2e-6, rtol=2e-6
    )
    cotangent = rng.standard_normal(6)
    np.testing.assert_allclose(
        kots.jacobian_transpose_mul(state, cotangent),
        jacobian.T @ cotangent,
        atol=2e-9,
        rtol=2e-9,
    )


@pytest.mark.parametrize("data_type", ["force", "force_diff1"])
def test_world_joint_force_dense_jvp_and_vjp_match_independent_difference(data_type):
    rng = np.random.default_rng(418 + (data_type == "force_diff1"))
    order = 4
    kots = _make_kots(order=order)
    motion = rng.standard_normal(order * kots.dof())
    kots.import_motions(motion)
    kots.dynamics()
    state = StateType("joint", "joint3", data_type, "world")
    state_order = state.time_order
    reduced = motion.reshape(kots.dof(), order)[:, :state_order].reshape(-1)
    direction = rng.standard_normal(reduced.shape)
    eps = 1e-7

    def value_at(value):
        full = motion.reshape(kots.dof(), order).copy()
        full[:, :state_order] = value.reshape(kots.dof(), state_order)
        sample = _make_kots(order=order)
        sample.import_motions(full.reshape(-1))
        sample.dynamics()
        return np.asarray(sample.state_info(state)).reshape(-1)

    finite = (value_at(reduced + eps * direction) - value_at(reduced - eps * direction)) / (2 * eps)
    jacobian = kots.jacobian(state)
    np.testing.assert_allclose(jacobian @ direction, finite, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(kots.jacobian_mul(state, direction), finite, atol=2e-6, rtol=2e-6)
    cotangent = rng.standard_normal(6)
    np.testing.assert_allclose(
        kots.jacobian_transpose_mul(state, cotangent), jacobian.T @ cotangent,
        atol=2e-9, rtol=2e-9,
    )


@pytest.mark.parametrize("batched", [False, True])
def test_rust_world_joint_wrench_vjp_matches_dense_jacobian(batched):
    pytest.importorskip("robokots._rust")
    rng = np.random.default_rng(420 + int(batched))
    kots = _make_kots(order=4)
    shape = (2, kots.dof() * 4) if batched else (kots.dof() * 4,)
    kots.import_motions(rng.standard_normal(shape))
    kots.dynamics(backend="rust", gravity=(0.2, -0.3, -9.81))
    states = [
        StateType("joint", "joint3", "momentum", "world"),
        StateType("joint", "joint3", "force_diff1", "world"),
    ]
    jacobian = kots.jacobian(states)
    rhs = rng.standard_normal(((2,) if batched else ()) + (12, 2))
    np.testing.assert_allclose(
        kots.jacobian_transpose_mul(states, rhs),
        np.swapaxes(jacobian, -1, -2) @ rhs,
        atol=2e-8,
        rtol=2e-8,
    )


def test_branched_fixed_mixed_dynamics_jacobian_matches_numerical_and_products():
    rng = np.random.default_rng(42)
    order = 4
    kots = Kots.from_urdf_file(str(BRANCHED_FIXED_MODEL_PATH), order=order)
    kots.import_motions(rng.standard_normal(order * kots.dof()))
    kots.dynamics(backend="numpy")
    states = [
        StateType("link", "a_tip", "force_diff1"),
        StateType("joint", "b_shoulder", "torque_diff1"),
    ]

    jacobian = kots.jacobian(states)
    np.testing.assert_allclose(
        jacobian,
        kots.jacobian(states, numerical=True),
        atol=1e-6,
        rtol=1e-6,
    )
    rhs = rng.standard_normal(order * kots.dof())
    lhs = rng.standard_normal(jacobian.shape[0])
    np.testing.assert_allclose(kots.jacobian_mul(states, rhs), jacobian @ rhs, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        kots.jacobian_transpose_mul(states, lhs),
        jacobian.T @ lhs,
        atol=1e-12,
        rtol=1e-12,
    )


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


def test_numpy_gravity_aware_torque_jacobian_matches_full_numerical(monkeypatch):
    kots = _make_kots(order=3)
    rng = np.random.default_rng(31)
    motion = rng.standard_normal(kots.dof() * 3)
    gravity = np.array([1.2, -3.4, 0.7])
    torque = StateType("total_joint", "total_joint", "torque")

    kots.import_motions(motion)
    kots.dynamics(backend="numpy", gravity=gravity, materialize_dict=False)
    expected = kots.jacobian(torque, numerical=True)
    actual = kots.jacobian(torque)

    np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=5e-7)

    zero_g = _make_kots(order=3)
    zero_g.import_motions(motion)
    zero_g.dynamics(backend="numpy", gravity=np.zeros(3), materialize_dict=False)
    gravity_delta = actual - zero_g.jacobian(torque)
    configuration_columns = np.arange(0, kots.dof() * 3, 3)
    other_columns = np.setdiff1d(np.arange(kots.dof() * 3), configuration_columns)
    assert np.linalg.norm(gravity_delta[:, configuration_columns]) > 1e-3
    np.testing.assert_allclose(gravity_delta[:, other_columns], 0.0, atol=1e-10)

    def fail_full_numerical(*args, **kwargs):
        raise AssertionError("the full numerical Jacobian fallback was used")

    monkeypatch.setattr(kots, "_jacobian_numerical", fail_full_numerical)
    monkeypatch.setattr(kots, "_jacobian_mul_numerical", fail_full_numerical)
    monkeypatch.setattr(kots, "_jacobian_transpose_mul_numerical", fail_full_numerical)

    rhs = rng.standard_normal(kots.dof() * 3)
    cotangent = rng.standard_normal(kots.dof())
    np.testing.assert_allclose(kots.jacobian(torque), actual)
    np.testing.assert_allclose(kots.jacobian_mul(torque, rhs), actual @ rhs)
    np.testing.assert_allclose(
        kots.jacobian_transpose_mul(torque, cotangent),
        actual.T @ cotangent,
        atol=1e-12,
        rtol=1e-12,
    )


def test_numpy_cmtm_gravity_force_jacobians_match_full_numerical():
    from robokots.core.models.whole_body import (
        total_coord_to_joint_torque_grad_mat,
    )

    kots = _make_kots(order=3)
    rng = np.random.default_rng(33)
    kots.import_motions(rng.standard_normal(kots.dof() * 3))
    kots.dynamics(
        backend="numpy",
        gravity=np.array([0.8, -2.1, -8.7]),
        materialize_dict=False,
    )

    states = [
        StateType("link", TARGET_LINK, "force"),
        StateType("joint", "joint3", "force"),
    ]
    for state in states:
        np.testing.assert_allclose(
            kots.jacobian(state),
            kots.jacobian(state, numerical=True),
            atol=5e-6,
            rtol=5e-7,
        )

    torque = StateType("total_joint", "total_joint", "torque")
    explicit_from_dict = total_coord_to_joint_torque_grad_mat(
        kots.robot_,
        kots.to_state_dict(),
        torque_order=1,
        gravity=kots.gravity_,
    )
    np.testing.assert_allclose(
        explicit_from_dict,
        kots.jacobian(torque, numerical=True),
        atol=5e-6,
        rtol=5e-7,
    )


@pytest.mark.parametrize("backend", ["numpy", "rust"])
def test_mixed_kinematics_and_torque_gravity_jacobian_is_analytic(
    monkeypatch,
    backend,
):
    if backend == "rust":
        pytest.importorskip("robokots._rust")
    kots = _make_kots(order=3)
    rng = np.random.default_rng(35)
    kots.import_motions(rng.standard_normal(kots.dof() * 3))
    kots.dynamics(
        backend=backend,
        gravity=np.array([0.8, -2.1, -8.7]),
        materialize_dict=False,
    )
    states = [
        StateType("link", TARGET_LINK, "vel"),
        StateType("joint", "joint3", "torque"),
    ]
    expected = kots.jacobian(states, numerical=True)

    def fail_numerical(*args, **kwargs):
        raise AssertionError("a numerical Jacobian fallback was used")

    monkeypatch.setattr(kots, "_jacobian_numerical", fail_numerical)
    np.testing.assert_allclose(
        kots.jacobian(states),
        expected,
        atol=5e-6,
        rtol=5e-7,
    )


@pytest.mark.parametrize("force_order", [1, 2, 3, 4, 8])
def test_numpy_cmtm_higher_order_gravity_jacobians_are_analytic(
    monkeypatch,
    force_order,
):
    order = force_order + 2
    kots = _make_kots(order=order)
    rng = np.random.default_rng(330 + force_order)
    motion_scale = 0.2 if force_order > 4 else 1.0
    kots.import_motions(
        motion_scale * rng.standard_normal(kots.dof() * order)
    )
    kots.dynamics(
        backend="numpy",
        gravity=np.array([0.8, -2.1, -8.7]),
        materialize_dict=False,
    )

    suffix = "" if force_order == 1 else f"_diff{force_order - 1}"
    states = [
        StateType("link", TARGET_LINK, f"force{suffix}"),
        StateType("joint", "joint3", f"force{suffix}"),
        StateType("total_joint", "total_joint", f"torque{suffix}"),
    ]
    expected = [kots.jacobian(state, numerical=True) for state in states]

    def fail_numerical(*args, **kwargs):
        raise AssertionError("a numerical gravity Jacobian path was used")

    monkeypatch.setattr(kots, "_jacobian_numerical", fail_numerical)
    monkeypatch.setattr(kots, "_jacobian_mul_numerical", fail_numerical)
    monkeypatch.setattr(kots, "_jacobian_transpose_mul_numerical", fail_numerical)

    finite_difference_atol = 2e-4 if force_order > 4 else 1e-4
    for state, expected_jacobian in zip(states, expected):
        np.testing.assert_allclose(
            kots.jacobian(state),
            expected_jacobian,
            atol=finite_difference_atol,
            rtol=2e-6,
        )

    torque_state = states[-1]
    torque_jacobian = kots.jacobian(torque_state)
    tangent = rng.standard_normal(kots.dof() * order)
    cotangent = rng.standard_normal(kots.dof())
    np.testing.assert_allclose(
        kots.jacobian_mul(torque_state, tangent),
        torque_jacobian @ tangent,
        atol=finite_difference_atol,
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        kots.jacobian_transpose_mul(torque_state, cotangent),
        torque_jacobian.T @ cotangent,
        atol=finite_difference_atol,
        rtol=2e-6,
    )


def test_batched_numpy_cmtm_higher_order_gravity_torque_jacobian(monkeypatch):
    force_order = 3
    order = force_order + 2
    gravity = np.array([0.4, -1.1, -9.3])
    torque = StateType("total_joint", "total_joint", "torque_diff2")
    kots = _make_kots(order=order)
    rng = np.random.default_rng(334)
    motions = rng.standard_normal((2, kots.dof() * order))

    kots.import_motions(motions)
    kots.dynamics(backend="numpy", gravity=gravity, materialize_dict=False)

    def fail_scalar_fallback(*args, **kwargs):
        raise AssertionError("the batched analytic path fell back to scalar states")

    monkeypatch.setattr(kots, "_state_builder", fail_scalar_fallback)
    actual = kots.jacobian(torque)

    expected = []
    for motion in motions:
        single = _make_kots(order=order)
        single.import_motions(motion)
        single.dynamics(backend="numpy", gravity=gravity, materialize_dict=False)
        expected.append(single.jacobian(torque, numerical=True))

    np.testing.assert_allclose(
        actual,
        np.stack(expected),
        atol=1e-4,
        rtol=2e-6,
    )


@pytest.mark.parametrize("force_order", [1, 2, 3, 4, 8])
def test_rust_cmtm_higher_order_gravity_jacobians_match_numpy_analytic(
    monkeypatch,
    force_order,
):
    pytest.importorskip("robokots._rust")
    order = force_order + 2
    gravity = np.array([0.8, -2.1, -8.7])
    rng = np.random.default_rng(430 + force_order)
    motion_scale = 0.2 if force_order > 4 else 1.0
    motion = motion_scale * rng.standard_normal(_make_kots(order=order).dof() * order)

    expected_kots = _make_kots(order=order)
    expected_kots.import_motions(motion)
    expected_kots.dynamics(
        backend="numpy", gravity=gravity, materialize_dict=False
    )

    actual_kots = _make_kots(order=order)
    actual_kots.import_motions(motion)
    actual_kots.dynamics(
        backend="rust", gravity=gravity, materialize_dict=False
    )
    np.testing.assert_array_equal(actual_kots.outward_state_.gravity, gravity)

    suffix = "" if force_order == 1 else f"_diff{force_order - 1}"
    states = [
        StateType("link", TARGET_LINK, f"force{suffix}"),
        StateType("joint", "joint3", f"force{suffix}"),
        StateType("total_joint", "total_joint", f"torque{suffix}"),
    ]

    def fail_numerical(*args, **kwargs):
        raise AssertionError("a numerical gravity Jacobian path was used")

    monkeypatch.setattr(actual_kots, "_jacobian_numerical", fail_numerical)
    monkeypatch.setattr(actual_kots, "_jacobian_mul_numerical", fail_numerical)
    monkeypatch.setattr(
        actual_kots, "_jacobian_transpose_mul_numerical", fail_numerical
    )
    for state in states:
        np.testing.assert_allclose(
            actual_kots.jacobian(state),
            expected_kots.jacobian(state),
            atol=2e-10,
            rtol=2e-10,
        )


def test_batched_rust_cmtm_higher_order_gravity_torque_jacobian_matches_numpy(
    monkeypatch,
):
    pytest.importorskip("robokots._rust")
    force_order = 3
    order = force_order + 2
    gravity = np.array([0.4, -1.1, -9.3])
    torque = StateType("total_joint", "total_joint", "torque_diff2")
    rng = np.random.default_rng(434)
    motions = rng.standard_normal((2, _make_kots(order=order).dof() * order))

    expected = _make_kots(order=order)
    expected.import_motions(motions)
    expected.dynamics(backend="numpy", gravity=gravity, materialize_dict=False)

    actual = _make_kots(order=order)
    actual.import_motions(motions)
    actual.dynamics(backend="rust", gravity=gravity, materialize_dict=False)

    def fail_scalar_fallback(*args, **kwargs):
        raise AssertionError("the batched analytic path fell back to scalar states")

    monkeypatch.setattr(actual, "_state_builder", fail_scalar_fallback)

    np.testing.assert_allclose(
        actual.jacobian(torque),
        expected.jacobian(torque),
        atol=2e-4,
        rtol=2e-6,
    )


@pytest.mark.parametrize("backend", ["numpy", "rust"])
@pytest.mark.parametrize("batched", [False, True])
def test_gravity_jacobian_mul_uses_direct_cmtm_kernel(
    monkeypatch,
    backend,
    batched,
):
    if backend == "rust":
        pytest.importorskip("robokots._rust")
    force_order = 3
    order = force_order + 2
    gravity = np.array([0.4, -1.1, -9.3])
    states = [
        StateType("link", TARGET_LINK, "force_diff2"),
        StateType("joint", "joint3", "torque_diff2"),
    ]
    rng = np.random.default_rng(535 + int(batched))
    kots = _make_kots(order=order)
    motion_shape = (2, kots.dof() * order) if batched else (kots.dof() * order,)
    kots.import_motions(rng.standard_normal(motion_shape))
    kots.dynamics(backend=backend, gravity=gravity, materialize_dict=False)

    jacobian = kots.jacobian(states)
    vector = rng.standard_normal(kots.dof() * order)
    matrix = rng.standard_normal((kots.dof() * order, 4))

    def fail_dense_jacobian(*args, **kwargs):
        raise AssertionError("jacobian_mul assembled a dense Jacobian")

    monkeypatch.setattr(kots, "_jacobian_from_state", fail_dense_jacobian)
    np.testing.assert_allclose(
        kots.jacobian_mul(states, vector),
        (jacobian @ vector[..., None])[..., 0],
        atol=2e-10,
        rtol=2e-10,
    )
    np.testing.assert_allclose(
        kots.jacobian_mul(states, matrix),
        jacobian @ matrix,
        atol=2e-10,
        rtol=2e-10,
    )


@pytest.mark.parametrize("backend", ["numpy", "rust"])
@pytest.mark.parametrize("batched", [False, True])
def test_gravity_jacobian_transpose_mul_uses_direct_cmtm_kernel(
    monkeypatch,
    backend,
    batched,
):
    if backend == "rust":
        pytest.importorskip("robokots._rust")
    force_order = 3
    order = force_order + 2
    gravity = np.array([0.4, -1.1, -9.3])
    states = [
        StateType("link", TARGET_LINK, "force_diff2"),
        StateType("joint", "joint3", "torque_diff2"),
    ]
    rng = np.random.default_rng(635 + int(batched))
    kots = _make_kots(order=order)
    motion_shape = (2, kots.dof() * order) if batched else (kots.dof() * order,)
    kots.import_motions(rng.standard_normal(motion_shape))
    kots.dynamics(backend=backend, gravity=gravity, materialize_dict=False)

    jacobian = kots.jacobian(states)
    output_dim = jacobian.shape[-2]
    vector_shape = (2, output_dim) if batched else (output_dim,)
    matrix_shape = (2, output_dim, 4) if batched else (output_dim, 4)
    vector = rng.standard_normal(vector_shape)
    matrix = rng.standard_normal(matrix_shape)

    def fail_dense_jacobian(*args, **kwargs):
        raise AssertionError("jacobian_transpose_mul assembled a dense Jacobian")

    monkeypatch.setattr(kots, "_jacobian_from_state", fail_dense_jacobian)
    np.testing.assert_allclose(
        kots.jacobian_transpose_mul(states, vector),
        (np.swapaxes(jacobian, -1, -2) @ vector[..., None])[..., 0],
        atol=3e-10,
        rtol=3e-10,
    )
    np.testing.assert_allclose(
        kots.jacobian_transpose_mul(states, matrix),
        np.swapaxes(jacobian, -1, -2) @ matrix,
        atol=3e-10,
        rtol=3e-10,
    )


@pytest.mark.parametrize("batched", [False, True])
def test_rust_gravity_torque_jacobian_uses_rnea_kernels(monkeypatch, batched):
    pytest.importorskip("robokots._rust")
    from robokots.outward import api as outward_api

    order = 3
    gravity = np.array([0.4, -1.1, -9.3])
    torque = StateType("total_joint", "total_joint", "torque")
    rng = np.random.default_rng(734 + int(batched))
    kots = _make_kots(order=order)
    motion_shape = (2, kots.dof() * order) if batched else (kots.dof() * order,)
    kots.import_motions(rng.standard_normal(motion_shape))
    kots.dynamics(backend="rust", gravity=gravity, materialize_dict=False)
    expected = kots.jacobian(torque, numerical=True)

    # A force/torque fallback would reach one of these outward kernels.
    def fail_outward(*args, **kwargs):
        raise AssertionError("the Rust RNEA Jacobian kernel was not used")

    monkeypatch.setattr(outward_api, "outward_jacobian", fail_outward)
    monkeypatch.setattr(outward_api, "outward_jacobian_matvec", fail_outward)
    monkeypatch.setattr(outward_api, "outward_jacobian_transpose_matvec", fail_outward)

    actual = kots.jacobian(torque)
    output_dim = actual.shape[-2]
    vector_shape = (2, output_dim) if batched else (output_dim,)
    matrix_shape = (2, kots.dof() * order, 3) if batched else (kots.dof() * order, 3)
    vector_cotangent_shape = (2, output_dim) if batched else (output_dim,)
    cotangent_shape = (2, output_dim, 2) if batched else (output_dim, 2)
    tangent = rng.standard_normal(matrix_shape)
    vector_cotangent = rng.standard_normal(vector_cotangent_shape)
    cotangent = rng.standard_normal(cotangent_shape)

    np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=5e-7)
    np.testing.assert_allclose(
        kots.jacobian_mul(torque, tangent),
        actual @ tangent,
        atol=2e-10,
        rtol=2e-10,
    )
    np.testing.assert_allclose(
        kots.jacobian_transpose_mul(torque, vector_cotangent),
        (np.swapaxes(actual, -1, -2) @ vector_cotangent[..., None])[..., 0],
        atol=2e-10,
        rtol=2e-10,
    )
    np.testing.assert_allclose(
        kots.jacobian_transpose_mul(torque, cotangent),
        np.swapaxes(actual, -1, -2) @ cotangent,
        atol=2e-10,
        rtol=2e-10,
    )


@pytest.mark.parametrize("batched", [False, True])
def test_jacobian_transpose_mul_many_fuses_torque_state_refs(monkeypatch, batched):
    pytest.importorskip("robokots._rust")
    rng = np.random.default_rng(854 + int(batched))
    order = 4
    kots = _make_kots(order=order)
    motion_shape = (2, kots.dof() * order) if batched else (kots.dof() * order,)
    kots.import_motions(rng.standard_normal(motion_shape))
    kots.dynamics(backend="rust", gravity=(0.2, -0.3, -9.81), materialize_dict=False)
    torque = StateType("joint", "joint3", "torque")
    torque_d1 = StateType("joint", "joint3", "torque_diff1")
    rhs_shape = (2, 1, 3) if batched else (1, 3)
    torque_rhs = rng.standard_normal(rhs_shape)
    torque_d1_rhs = rng.standard_normal(rhs_shape)
    fused_rhs = np.concatenate([torque_rhs, torque_d1_rhs], axis=-2)
    expected = kots.jacobian_transpose_mul([torque, torque_d1], fused_rhs)

    calls = 0
    original = kots._rust_cmtm_outward_dynamics_jacobian_transpose_apply

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(kots, "_rust_cmtm_outward_dynamics_jacobian_transpose_apply", counted)
    actual = kots.jacobian_transpose_mul_many([
        (torque, torque_rhs),
        (torque_d1, torque_d1_rhs),
    ])
    assert calls == 1
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-10)


@pytest.mark.parametrize("batched", [False, True])
def test_rust_kinetic_energy_jvp_vjp_and_batch_match_finite_difference(batched):
    pytest.importorskip("robokots._rust")
    rng = np.random.default_rng(8540 + int(batched))
    order = 4
    kots = _make_kots(order=order)
    batch = 3
    motion_shape = (batch, kots.dof() * order) if batched else (kots.dof() * order,)
    motion = rng.standard_normal(motion_shape)
    kots.import_motions(motion)

    input_dim = kots.dof() * 2
    rhs_cols = 3
    direction_shape = (batch, input_dim, rhs_cols) if batched else (input_dim, rhs_cols)
    energy_rhs_shape = (batch, 1, rhs_cols) if batched else (1, rhs_cols)
    directions = rng.standard_normal(direction_shape)
    energy_rhs = rng.standard_normal(energy_rhs_shape)
    ones_shape = (batch, 1) if batched else (1,)
    gradient = kots.kinetic_energy_jacobian_transpose_mul(np.ones(ones_shape))
    jvp = kots.kinetic_energy_jacobian_mul(directions)
    vjp = kots.kinetic_energy_jacobian_transpose_mul(energy_rhs)

    expected_jvp = np.einsum("...i,...ik->...k", gradient, directions)[..., None, :]
    expected_vjp = gradient[..., :, None] * energy_rhs[..., 0, :][..., None, :]
    np.testing.assert_allclose(jvp, expected_jvp, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(vjp, expected_vjp, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(
        np.sum(jvp * energy_rhs, axis=(-2, -1)),
        np.sum(directions * vjp, axis=(-2, -1)),
        atol=2e-11,
        rtol=2e-11,
    )

    sample_motion = motion[0] if batched else motion
    sample_direction = directions[0, :, 0] if batched else directions[:, 0]
    sample = _make_kots(order=order)
    eps = 1e-6
    values = []
    for sign in (-1.0, 1.0):
        perturbed = sample_motion.reshape(kots.dof(), order).copy()
        perturbed[:, :2] += sign * eps * sample_direction.reshape(kots.dof(), 2)
        sample.import_motions(perturbed.reshape(-1))
        values.append(sample.kinetic_energy_state())
    np.testing.assert_allclose(
        (values[1] - values[0]) / (2.0 * eps),
        gradient[0] @ sample_direction if batched else gradient @ sample_direction,
        atol=2e-7,
        rtol=2e-7,
    )


@pytest.mark.parametrize("batched", [False, True])
def test_total_body_kinetic_energy_state_type_uses_energy_kernels(batched):
    pytest.importorskip("robokots._rust")
    rng = np.random.default_rng(8542 + int(batched))
    kots = _make_kots(order=4)
    motion_shape = (2, kots.dof() * 4) if batched else (kots.dof() * 4,)
    kots.import_motions(rng.standard_normal(motion_shape))
    energy = StateType("total_body", "total_body", "kinetic_energy")
    value = kots.state_info(energy)
    np.testing.assert_allclose(value, kots.kinetic_energy_state())

    jacobian = kots.jacobian(energy)
    directions = rng.standard_normal((2, kots.dof() * 2, 2) if batched else (kots.dof() * 2, 2))
    cotangent = rng.standard_normal((2, 1, 2) if batched else (1, 2))
    np.testing.assert_allclose(kots.jacobian_mul(energy, directions), jacobian @ directions)
    np.testing.assert_allclose(
        kots.jacobian_transpose_mul(energy, cotangent),
        np.swapaxes(jacobian, -1, -2) @ cotangent,
    )
    assert energy.alliance == "total_body_kinetic_energy"


def test_total_body_kinetic_energy_mixes_with_torque_vjp():
    pytest.importorskip("robokots._rust")
    rng = np.random.default_rng(8543)
    kots = _make_kots(order=4)
    kots.import_motions(rng.standard_normal(kots.dof() * 4))
    kots.dynamics(backend="rust", gravity=(0.2, -0.3, -9.81), materialize_dict=False)
    energy = StateType("total_body", "total_body", "kinetic_energy")
    torque = StateType("joint", "joint3", "torque_diff1")
    states = [energy, torque]
    jacobian = kots.jacobian(states)
    rhs = rng.standard_normal((2, 3))
    direction = rng.standard_normal(kots.dof() * 4)
    np.testing.assert_allclose(kots.jacobian_mul(states, direction), jacobian @ direction)
    np.testing.assert_allclose(
        kots.jacobian_transpose_mul(states, rhs),
        jacobian.T @ rhs,
        atol=2e-10,
        rtol=2e-10,
    )


def test_rust_cmtm_kinematics_tangent_matches_directional_difference():
    pytest.importorskip("robokots._rust")
    order = 4
    rng = np.random.default_rng(855)
    kots = _make_kots(order=order)
    motion = rng.standard_normal(kots.dof() * order)
    directions = rng.standard_normal((kots.dof() * order, 3))
    robot = kots._rust_compiled_robot()

    dmat, dvecs = robot.cmtm_kinematics_tangent(motion, directions, order)
    eps = 1e-7
    for column in range(directions.shape[-1]):
        plus = robot.kinematics_cmtm(motion + eps * directions[:, column], order)
        minus = robot.kinematics_cmtm(motion - eps * directions[:, column], order)
        np.testing.assert_allclose(
            dmat[..., column],
            (plus[0] - minus[0]) / (2.0 * eps),
            atol=2e-8,
            rtol=2e-8,
        )
        np.testing.assert_allclose(
            dvecs[..., column],
            (plus[1] - minus[1]) / (2.0 * eps),
            atol=3e-8,
            rtol=3e-8,
        )


def test_rust_cmtm_link_dynamics_tangent_matches_directional_difference():
    pytest.importorskip("robokots._rust")
    dynamics_order = 2
    order = dynamics_order + 2
    rng = np.random.default_rng(856)
    kots = _make_kots(order=order)
    motion = rng.standard_normal(kots.dof() * order)
    directions = rng.standard_normal((kots.dof() * order, 2))
    robot = kots._rust_compiled_robot()
    dmomentum, dforce = robot.cmtm_link_dynamics_tangent(
        motion, directions, dynamics_order,
    )
    eps = 1e-7
    for column in range(directions.shape[-1]):
        plus = robot.dynamics_cmtm(motion + eps * directions[:, column], dynamics_order)
        minus = robot.dynamics_cmtm(motion - eps * directions[:, column], dynamics_order)
        np.testing.assert_allclose(
            dmomentum[..., column],
            (plus[0] - minus[0]) / (2.0 * eps),
            atol=3e-7,
            rtol=3e-7,
        )
        np.testing.assert_allclose(
            dforce[..., column],
            (plus[1] - minus[1]) / (2.0 * eps),
            atol=3e-7,
            rtol=3e-7,
        )


def test_rust_cmtm_kinematics_vjp_is_tangent_transpose():
    pytest.importorskip("robokots._rust")
    order = 4
    rhs_cols = 3
    rng = np.random.default_rng(857)
    kots = _make_kots(order=order)
    robot = kots._rust_compiled_robot()
    motion = rng.standard_normal(kots.dof() * order)
    mat_rhs = rng.standard_normal((robot.link_num, 4, 4, rhs_cols))
    vec_rhs = rng.standard_normal((robot.link_num, order - 1, 6, rhs_cols))

    actual = robot.cmtm_kinematics_transpose_matmul_rhs(
        motion, mat_rhs, vec_rhs, order,
    )
    basis = np.eye(kots.dof() * order)
    dmat, dvec = robot.cmtm_kinematics_tangent(motion, basis, order)
    expected = (
        np.einsum("lpqi,lpqr->ir", dmat, mat_rhs)
        + np.einsum("ltsi,ltsr->ir", dvec, vec_rhs)
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("order", [3, 4, 5])
def test_rust_cmtm_outward_kinematics_reverse_vjp_matches_directional_difference(order):
    """The vector-only VJP is a reverse recurrence, not basis tangents."""
    pytest.importorskip("robokots._rust")
    rng = np.random.default_rng(860 + order)
    rhs_cols = 3
    kots = _make_kots(order=order)
    robot = kots._rust_compiled_robot()
    motion = rng.standard_normal(kots.dof() * order)
    direction = rng.standard_normal(motion.size)
    link_rhs = rng.standard_normal((robot.link_num, order - 1, 6, rhs_cols))
    joint_rhs = rng.standard_normal((robot.joint_num, order - 1, 6, rhs_cols))
    actual = robot.cmtm_outward_kinematics_transpose_matmul_rhs(
        motion, link_rhs, joint_rhs, order,
    )

    def loss(x):
        _, link_vec, _, joint_vec = robot.kinematics_cmtm(x, order)
        return np.einsum("ltsr,lts->r", link_rhs, link_vec) + np.einsum("jtsr,jts->r", joint_rhs, joint_vec)

    eps = 1e-6
    finite_difference = (loss(motion + eps * direction) - loss(motion - eps * direction)) / (2.0 * eps)
    np.testing.assert_allclose(direction @ actual, finite_difference, atol=6e-7, rtol=6e-7)


def test_rust_cmtm_outward_dynamics_vjp_includes_each_wrench_output():
    pytest.importorskip("robokots._rust")
    dynamics_order = 2
    order = dynamics_order + 2
    rhs_cols = 2
    rng = np.random.default_rng(858)
    kots = _make_kots(order=order)
    robot = kots._rust_compiled_robot()
    motion = rng.standard_normal(kots.dof() * order)
    link_momentum_rhs = rng.standard_normal((robot.link_num, dynamics_order + 1, 6, rhs_cols))
    link_force_rhs = rng.standard_normal((robot.link_num, dynamics_order, 6, rhs_cols))
    joint_momentum_rhs = rng.standard_normal((robot.joint_num, dynamics_order + 1, 6, rhs_cols))
    joint_force_rhs = rng.standard_normal((robot.joint_num, dynamics_order, 6, rhs_cols))
    joint_torque_rhs = rng.standard_normal((robot.joint_num, dynamics_order, rhs_cols))

    actual = robot.dynamics_cmtm_transpose_matmul_rhs(
        motion,
        link_momentum_rhs,
        link_force_rhs,
        joint_momentum_rhs,
        joint_force_rhs,
        joint_torque_rhs,
        dynamics_order,
    )
    basis = np.eye(kots.dof() * order)
    d_link_momentum, d_link_force = robot.cmtm_link_dynamics_tangent(
        motion, basis, dynamics_order,
    )
    # Isolate the joint series through the complete kernel; this guards the
    # momentum/force/torque packing independently of the link primitives.
    joint_only = robot.dynamics_cmtm_transpose_matmul_rhs(
        motion,
        np.zeros_like(link_momentum_rhs),
        np.zeros_like(link_force_rhs),
        joint_momentum_rhs,
        joint_force_rhs,
        joint_torque_rhs,
        dynamics_order,
    )
    link_only = robot.cmtm_link_dynamics_transpose_matmul_rhs(
        motion, link_momentum_rhs, link_force_rhs, dynamics_order,
    )
    np.testing.assert_allclose(actual, link_only + joint_only, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        link_only,
        np.einsum("ltsi,ltsr->ir", d_link_momentum, link_momentum_rhs)
        + np.einsum("ltsi,ltsr->ir", d_link_force, link_force_rhs),
        atol=1e-12,
        rtol=1e-12,
    )


def test_rust_cmtm_complete_reverse_vjp_matches_directional_difference_with_gravity():
    pytest.importorskip("robokots._rust")
    rng = np.random.default_rng(871)
    order = 5
    dynamics_order = order - 2
    rhs_cols = 2
    gravity = np.array([0.4, -1.1, -9.3])
    kots = Kots.from_urdf_file(str(BRANCHED_FIXED_MODEL_PATH), order=order)
    robot = kots._rust_compiled_robot()
    motion = rng.standard_normal(kots.dof() * order)
    direction = rng.standard_normal(motion.size)
    lm = rng.standard_normal((robot.link_num, dynamics_order + 1, 6, rhs_cols))
    lf = rng.standard_normal((robot.link_num, dynamics_order, 6, rhs_cols))
    jm = rng.standard_normal((robot.joint_num, dynamics_order + 1, 6, rhs_cols))
    jf = rng.standard_normal((robot.joint_num, dynamics_order, 6, rhs_cols))
    jt = rng.standard_normal((robot.joint_num, dynamics_order, rhs_cols))
    actual = robot.dynamics_cmtm_transpose_matmul_rhs(
        motion, lm, lf, jm, jf, jt, dynamics_order, gravity=gravity,
    )

    def loss(x):
        link_momentum, link_force, joint_momentum, joint_force, joint_torque = robot.dynamics_cmtm(
            x, dynamics_order, gravity=gravity,
        )
        return (
            np.einsum("ltdr,ltd->r", lm, link_momentum)
            + np.einsum("ltdr,ltd->r", lf, link_force)
            + np.einsum("jtdr,jtd->r", jm, joint_momentum)
            + np.einsum("jtdr,jtd->r", jf, joint_force)
            + np.einsum("jtr,jt->r", jt, np.asarray(joint_torque)[..., 0])
        )

    eps = 1e-6
    finite_difference = (loss(motion + eps * direction) - loss(motion - eps * direction)) / (2.0 * eps)
    np.testing.assert_allclose(direction @ actual, finite_difference, atol=8e-7, rtol=8e-7)


def test_rust_rnea_jacobian_api_accepts_gravity():
    pytest.importorskip("robokots._rust")
    kots = _make_kots(order=3)
    rng = np.random.default_rng(736)
    kots.import_motions(rng.standard_normal(kots.dof() * 3))
    q, v, a, _ = kots._rust_qva_order3()
    gravity = np.array([0.4, -1.1, -9.3])
    robot = kots._rust_compiled_robot()

    dynamics_jacobian = robot.dynamics_jacobian(q, v, a, gravity=gravity)
    expected_grouped = np.concatenate(
        [
            dynamics_jacobian[:, 0::3],
            dynamics_jacobian[:, 1::3],
            dynamics_jacobian[:, 2::3],
        ],
        axis=1,
    )
    np.testing.assert_allclose(
        robot.rnea_jacobian(q, v, a, gravity=gravity),
        expected_grouped,
        atol=2e-10,
        rtol=2e-10,
    )


def test_batched_numpy_gravity_aware_torque_jacobian_matches_scalar_loop():
    rng = np.random.default_rng(32)
    gravity = np.array([0.4, -1.1, -9.3])
    torque = StateType("total_joint", "total_joint", "torque")
    kots = _make_kots(order=3)
    motions = rng.standard_normal((2, kots.dof() * 3))

    kots.import_motions(motions)
    kots.dynamics(backend="numpy", gravity=gravity, materialize_dict=False)
    actual = kots.jacobian(torque)

    expected = []
    for motion in motions:
        single = _make_kots(order=3)
        single.import_motions(motion)
        single.dynamics(backend="numpy", gravity=gravity, materialize_dict=False)
        expected.append(single.jacobian(torque, numerical=True))

    np.testing.assert_allclose(actual, np.stack(expected), atol=5e-6, rtol=5e-7)


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
