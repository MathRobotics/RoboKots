import numpy as np
import pytest
from pathlib import Path

import mathrobo as mr
from robokots.kots import *
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


def test_jacobian_matvec_kinematics_matches_jacobian_product():
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
        kots.jacobian_matvec(states, vec),
        kots.jacobian(states) @ vec,
        atol=1e-10,
        rtol=1e-10,
    )

    actual_parts = kots.jacobian_matvec(states, vec, list_output=True)
    expected_parts = [jacob @ vec for jacob in kots.jacobian(states, list_output=True)]
    for actual, expected in zip(actual_parts, expected_parts):
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_jacobian_target_matvec_matches_jacobian_product():
    kots = _make_kots(order=3)
    rng = np.random.default_rng(1)

    kots.set_target_from_file(str(TARGET_PATH))
    motion = rng.standard_normal(kots.order() * kots.dof())
    kots.import_motions(motion)
    kots.dynamics()

    vec = rng.standard_normal(kots.dof() * StateType.max_time_order(kots.target_._targets))
    np.testing.assert_allclose(
        kots.jacobian_target_matvec(vec),
        kots.jacobian_target() @ vec,
        atol=1e-10,
        rtol=1e-10,
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
    actual_matvec = kots.jacobian_matvec(acc_state, vec)

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


def test_batched_dynamics_jacobian_matvec_matches_jacobian_product():
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

    actual = kots.jacobian_matvec(states, vecs)
    expected = (kots.jacobian(states) @ vecs[..., None])[..., 0]
    parts = kots.jacobian_matvec(states, vecs, list_output=True)

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


def test_batched_jacobian_matvec_shape_contract_with_per_sample_vecs():
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

    matvec = kots.jacobian_matvec(states, vecs)
    parts = kots.jacobian_matvec(states, vecs, list_output=True)

    assert matvec.shape == (2, 3, 12)
    assert len(parts) == 2
    assert parts[0].shape == (2, 3, 6)
    assert parts[1].shape == (2, 3, 6)
    np.testing.assert_allclose(matvec, np.concatenate(parts, axis=-1))

    single = _make_kots(order=order)
    single.import_motions(motions[1, 2])
    single.kinematics()
    np.testing.assert_allclose(matvec[1, 2], single.jacobian(states) @ vecs[1, 2])


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
    shared_matvec = kots.jacobian_target_matvec(shared_vec)
    sample_matvec = kots.jacobian_target_matvec(sample_vecs)
    matvec_parts = kots.jacobian_target_matvec(sample_vecs, list_output=True)

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
