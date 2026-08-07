from __future__ import annotations

import numpy as np
import pytest

from robokots.core.state import StateType
from robokots.kots import Kots


pin = pytest.importorskip("pinocchio", reason="Pinocchio is an optional developer dependency")


URDF = """<?xml version="1.0"?>
<robot name="inverse_dynamics_comparison">
  <link name="base"/>
  <link name="link1">
    <inertial>
      <origin xyz="0.15 -0.08 0.04" rpy="0.4 -0.2 0.3"/>
      <mass value="2.3"/>
      <inertia ixx="0.12" ixy="0.01" ixz="-0.005"
               iyy="0.18" iyz="0.008" izz="0.21"/>
    </inertial>
  </link>
  <link name="link2">
    <inertial>
      <origin xyz="-0.04 0.12 0.06" rpy="-0.25 0.15 -0.35"/>
      <mass value="1.4"/>
      <inertia ixx="0.07" ixy="-0.004" ixz="0.006"
               iyy="0.09" iyz="0.003" izz="0.11"/>
    </inertial>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base"/>
    <child link="link1"/>
    <origin xyz="0.1 0.0 0.25" rpy="0.1 -0.2 0.3"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>
  <joint name="elbow" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0.35 -0.05 0.1" rpy="-0.15 0.05 0.2"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>
</robot>
"""


BRANCHED_FIXED_URDF = """<?xml version="1.0"?>
<robot name="branched_fixed_comparison">
  <link name="base"/>
  <link name="a_upper">
    <inertial>
      <origin xyz="0.08 -0.03 0.12" rpy="0.2 -0.1 0.3"/>
      <mass value="2.1"/>
      <inertia ixx="0.10" ixy="0.006" ixz="-0.004"
               iyy="0.14" iyz="0.005" izz="0.17"/>
    </inertial>
  </link>
  <link name="a_spacer">
    <inertial>
      <origin xyz="0.05 0.02 -0.01" rpy="-0.1 0.25 0.05"/>
      <mass value="0.8"/>
      <inertia ixx="0.035" ixy="-0.002" ixz="0.001"
               iyy="0.041" iyz="0.003" izz="0.052"/>
    </inertial>
  </link>
  <link name="a_forearm">
    <inertial>
      <origin xyz="0.16 0.04 0.02" rpy="0.15 0.05 -0.2"/>
      <mass value="1.3"/>
      <inertia ixx="0.06" ixy="0.003" ixz="0.002"
               iyy="0.08" iyz="-0.004" izz="0.09"/>
    </inertial>
  </link>
  <link name="b_upper">
    <inertial>
      <origin xyz="-0.06 0.09 0.03" rpy="-0.3 0.1 0.2"/>
      <mass value="1.7"/>
      <inertia ixx="0.08" ixy="-0.005" ixz="0.004"
               iyy="0.11" iyz="0.002" izz="0.13"/>
    </inertial>
  </link>
  <link name="b_payload">
    <inertial>
      <origin xyz="0.02 -0.07 0.11" rpy="0.05 -0.15 0.35"/>
      <mass value="0.9"/>
      <inertia ixx="0.04" ixy="0.001" ixz="-0.003"
               iyy="0.05" iyz="0.002" izz="0.065"/>
    </inertial>
  </link>

  <!-- Deliberately not parent-before-child and with branch B first. -->
  <joint name="b_payload_fixed" type="fixed">
    <parent link="b_upper"/>
    <child link="b_payload"/>
    <origin xyz="0.18 -0.04 0.09" rpy="0.1 0.2 -0.1"/>
  </joint>
  <joint name="a_elbow" type="revolute">
    <parent link="a_spacer"/>
    <child link="a_forearm"/>
    <origin xyz="0.27 0.03 -0.05" rpy="-0.2 0.1 0.15"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>
  <joint name="b_shoulder" type="revolute">
    <parent link="base"/>
    <child link="b_upper"/>
    <origin xyz="-0.12 0.18 0.24" rpy="0.05 -0.25 0.2"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>
  <joint name="a_spacer_fixed" type="fixed">
    <parent link="a_upper"/>
    <child link="a_spacer"/>
    <origin xyz="0.22 -0.02 0.08" rpy="0.2 0.05 -0.15"/>
  </joint>
  <joint name="a_shoulder" type="revolute">
    <parent link="base"/>
    <child link="a_upper"/>
    <origin xyz="0.14 -0.16 0.28" rpy="-0.1 0.2 0.25"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>
</robot>
"""


MIXED_REVOLUTE_PRISMATIC_URDF = """<?xml version="1.0"?>
<robot name="mixed_joint_chain">
  <link name="base"/>
  <link name="rotor">
    <inertial>
      <origin xyz="0.08 -0.03 0.11" rpy="0.2 -0.1 0.3"/>
      <mass value="2.0"/>
      <inertia ixx="0.10" ixy="0.006" ixz="-0.004"
               iyy="0.14" iyz="0.005" izz="0.17"/>
    </inertial>
  </link>
  <link name="slider">
    <inertial>
      <origin xyz="0.04 0.07 -0.02" rpy="-0.15 0.25 0.1"/>
      <mass value="1.4"/>
      <inertia ixx="0.06" ixy="-0.003" ixz="0.002"
               iyy="0.08" iyz="0.004" izz="0.095"/>
    </inertial>
  </link>
  <link name="tool">
    <inertial>
      <origin xyz="-0.03 0.05 0.09" rpy="0.1 0.2 -0.25"/>
      <mass value="0.9"/>
      <inertia ixx="0.04" ixy="0.002" ixz="-0.001"
               iyy="0.05" iyz="-0.003" izz="0.065"/>
    </inertial>
  </link>
  <link name="payload">
    <inertial>
      <origin xyz="0.02 -0.04 0.06" rpy="-0.2 0.05 0.15"/>
      <mass value="0.6"/>
      <inertia ixx="0.025" ixy="-0.001" ixz="0.002"
               iyy="0.03" iyz="0.001" izz="0.04"/>
    </inertial>
  </link>
  <joint name="base_rotation" type="revolute">
    <parent link="base"/>
    <child link="rotor"/>
    <origin xyz="0.1 -0.2 0.25" rpy="0.1 -0.2 0.3"/>
    <axis xyz="1 2 3"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>
  <joint name="extension" type="prismatic">
    <parent link="rotor"/>
    <child link="slider"/>
    <origin xyz="0.24 0.06 -0.08" rpy="-0.2 0.15 0.1"/>
    <axis xyz="-2 1 3"/>
    <limit lower="-0.5" upper="0.8" effort="100" velocity="10"/>
  </joint>
  <joint name="wrist_rotation" type="revolute">
    <parent link="slider"/>
    <child link="tool"/>
    <origin xyz="0.13 -0.04 0.12" rpy="0.25 -0.1 -0.15"/>
    <axis xyz="2 -3 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>
  <joint name="payload_fixed" type="fixed">
    <parent link="tool"/>
    <child link="payload"/>
    <origin xyz="0.08 0.03 -0.02" rpy="0.1 0.2 0.05"/>
  </joint>
</robot>
"""


@pytest.mark.parametrize(
    "gravity",
    [
        np.array([0.0, 0.0, -9.81]),
        np.array([1.2, -3.4, 0.7]),
        np.zeros(3),
    ],
)
def test_inverse_dynamics_matches_pinocchio_for_same_urdf(tmp_path, gravity):
    urdf_path = tmp_path / "comparison.urdf"
    urdf_path.write_text(URDF, encoding="utf-8")

    kots = Kots.from_urdf_file(str(urdf_path), order=3)
    pin_model = pin.buildModelFromUrdf(str(urdf_path))
    pin_model.gravity.linear = gravity
    pin_data = pin_model.createData()

    assert [joint.name for joint in kots.robot_.joints if joint.dof] == list(pin_model.names)[1:]

    rng = np.random.default_rng(20260807)
    for _ in range(10):
        q = rng.uniform(-1.0, 1.0, pin_model.nq)
        v = rng.normal(size=pin_model.nv)
        a = rng.normal(size=pin_model.nv)

        expected = pin.rnea(pin_model, pin_data, q, v, a).copy()
        actual = kots.inverse_dynamics(q, v, a, gravity=gravity)

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_inverse_dynamics_default_gravity_matches_pinocchio(tmp_path):
    urdf_path = tmp_path / "comparison.urdf"
    urdf_path.write_text(URDF, encoding="utf-8")
    kots = Kots.from_urdf_file(str(urdf_path), order=3)
    pin_model = pin.buildModelFromUrdf(str(urdf_path))

    q = np.array([0.4, -0.2])
    v = np.array([0.3, 0.5])
    a = np.array([-0.7, 0.1])

    expected = pin.rnea(pin_model, pin_model.createData(), q, v, a)
    np.testing.assert_allclose(kots.inverse_dynamics(q, v, a), expected, rtol=1e-12, atol=1e-12)


def test_batched_inverse_dynamics_matches_pinocchio(tmp_path):
    urdf_path = tmp_path / "comparison.urdf"
    urdf_path.write_text(URDF, encoding="utf-8")
    kots = Kots.from_urdf_file(str(urdf_path), order=3)
    pin_model = pin.buildModelFromUrdf(str(urdf_path))
    pin_data = pin_model.createData()

    rng = np.random.default_rng(17)
    permutation = np.arange(pin_model.nq)[::-1]
    q = rng.uniform(-1.0, 1.0, (4, pin_model.nq))[:, permutation]
    v = rng.normal(size=(4, pin_model.nv))[:, permutation]
    a = rng.normal(size=(4, pin_model.nv))[:, permutation]
    assert q.flags.f_contiguous and not q.flags.c_contiguous

    expected = np.stack(
        [pin.rnea(pin_model, pin_data, q[i], v[i], a[i]).copy() for i in range(q.shape[0])]
    )
    np.testing.assert_allclose(kots.inverse_dynamics(q, v, a), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("gravity", [np.array([0.0, 0.0, -9.81]), np.zeros(3)])
def test_branched_fixed_joint_rnea_matches_pinocchio_by_joint_name(tmp_path, gravity):
    urdf_path = tmp_path / "branched_fixed.urdf"
    urdf_path.write_text(BRANCHED_FIXED_URDF, encoding="utf-8")

    kots = Kots.from_urdf_file(str(urdf_path), order=3)
    pin_model = pin.buildModelFromUrdf(str(urdf_path))
    pin_model.gravity.linear = gravity
    pin_data = pin_model.createData()

    kots_names = [joint.name for joint in kots.robot_.joints if joint.dof]
    pin_names = list(pin_model.names)[1:]
    assert set(kots_names) == set(pin_names)
    assert kots_names != pin_names  # Exercise name-based mapping for a branched model.
    assert sum(joint.type == "fixed" for joint in kots.robot_.joints) == 3

    rng = np.random.default_rng(314159)
    for _ in range(10):
        q_by_name = {name: rng.uniform(-1.0, 1.0) for name in kots_names}
        v_by_name = {name: rng.normal() for name in kots_names}
        a_by_name = {name: rng.normal() for name in kots_names}

        q_kots = np.array([q_by_name[name] for name in kots_names])
        v_kots = np.array([v_by_name[name] for name in kots_names])
        a_kots = np.array([a_by_name[name] for name in kots_names])
        q_pin = np.array([q_by_name[name] for name in pin_names])
        v_pin = np.array([v_by_name[name] for name in pin_names])
        a_pin = np.array([a_by_name[name] for name in pin_names])

        actual_raw = kots.inverse_dynamics(q_kots, v_kots, a_kots, gravity=gravity)
        expected_raw = pin.rnea(pin_model, pin_data, q_pin, v_pin, a_pin).copy()
        actual_by_name = dict(zip(kots_names, actual_raw))
        expected_by_name = dict(zip(pin_names, expected_raw))

        np.testing.assert_allclose(
            [actual_by_name[name] for name in sorted(kots_names)],
            [expected_by_name[name] for name in sorted(kots_names)],
            rtol=1e-12,
            atol=1e-12,
        )


def test_moving_link_without_inertial_is_massless_like_pinocchio(tmp_path):
    urdf = """<robot name="massless_link">
      <link name="base"/>
      <link name="massless_body"/>
      <joint name="joint" type="revolute">
        <parent link="base"/>
        <child link="massless_body"/>
        <axis xyz="0 1 0"/>
        <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
      </joint>
    </robot>"""
    urdf_path = tmp_path / "massless_link.urdf"
    urdf_path.write_text(urdf, encoding="utf-8")
    kots = Kots.from_urdf_file(str(urdf_path), order=3)
    pin_model = pin.buildModelFromUrdf(str(urdf_path))
    pin_model.gravity.linear[:] = 0.0

    q = np.array([0.3])
    v = np.array([0.4])
    a = np.array([-0.62])
    expected = pin.rnea(pin_model, pin_model.createData(), q, v, a)

    np.testing.assert_allclose(
        kots.inverse_dynamics(q, v, a, gravity=np.zeros(3)),
        expected,
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
def test_numpy_dynamics_normalizes_joint_axis_like_pinocchio(tmp_path, joint_type):
    urdf = f"""<robot name="non_unit_axis">
      <link name="base"/>
      <link name="body">
        <inertial>
          <origin xyz="0.15 -0.08 0.04" rpy="0.2 -0.1 0.3"/>
          <mass value="2.3"/>
          <inertia ixx="0.12" ixy="0.01" ixz="-0.005"
                   iyy="0.18" iyz="0.008" izz="0.21"/>
        </inertial>
      </link>
      <joint name="joint" type="{joint_type}">
        <parent link="base"/>
        <child link="body"/>
        <origin xyz="0.2 0.1 -0.1" rpy="0.1 -0.2 0.3"/>
        <axis xyz="1 2 3"/>
        <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
      </joint>
    </robot>"""
    urdf_path = tmp_path / f"non_unit_{joint_type}.urdf"
    urdf_path.write_text(urdf, encoding="utf-8")
    kots = Kots.from_urdf_file(str(urdf_path), order=3)
    pin_model = pin.buildModelFromUrdf(str(urdf_path))
    pin_model.gravity.linear[:] = 0.0

    q = np.array([0.31])
    v = np.array([0.43])
    a = np.array([-0.62])
    expected = pin.rnea(pin_model, pin_model.createData(), q, v, a)
    kots.import_motion_array(np.stack([q, v, a], axis=-1))
    kots.dynamics(backend="numpy", materialize_dict=False)
    actual = np.asarray(kots.state_info(StateType("total_joint", "total_joint", "torque"))).reshape(-1)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "gravity",
    [np.array([0.0, 0.0, -9.81]), np.array([1.2, -3.4, 0.7]), np.zeros(3)],
)
def test_mixed_revolute_prismatic_inverse_dynamics_matches_pinocchio(tmp_path, gravity):
    urdf_path = tmp_path / "mixed_joint_chain.urdf"
    urdf_path.write_text(MIXED_REVOLUTE_PRISMATIC_URDF, encoding="utf-8")
    kots = Kots.from_urdf_file(str(urdf_path), order=3)
    pin_model = pin.buildModelFromUrdf(str(urdf_path))
    pin_model.gravity.linear = gravity
    pin_data = pin_model.createData()

    assert [joint.name for joint in kots.robot_.joints if joint.dof] == list(pin_model.names)[1:]
    rng = np.random.default_rng(271828)
    q = rng.uniform(-0.4, 0.4, (12, pin_model.nq))
    v = rng.normal(size=(12, pin_model.nv))
    a = rng.normal(size=(12, pin_model.nv))
    expected = np.stack(
        [pin.rnea(pin_model, pin_data, q[i], v[i], a[i]).copy() for i in range(q.shape[0])]
    )

    np.testing.assert_allclose(
        kots.inverse_dynamics(q, v, a, gravity=gravity),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    for i in range(q.shape[0]):
        np.testing.assert_allclose(
            kots.inverse_dynamics(q[i], v[i], a[i], gravity=gravity),
            expected[i],
            rtol=1e-12,
            atol=1e-12,
        )

    kots.import_motion_array(np.stack([q[0], v[0], a[0]], axis=-1))
    with pytest.raises(NotImplementedError, match="prismatic"):
        kots.dynamics(backend="rust")


def test_rnea_without_added_world_handles_xml_late_root_link(tmp_path):
    urdf = """<robot name="late_root">
      <link name="tip">
        <inertial>
          <origin xyz="0.1 0.2 0.0" rpy="0.2 -0.1 0.3"/>
          <mass value="2.0"/>
          <inertia ixx="0.1" ixy="0.01" ixz="0.0"
                   iyy="0.2" iyz="-0.01" izz="0.3"/>
        </inertial>
      </link>
      <link name="base"/>
      <joint name="joint" type="revolute">
        <parent link="base"/>
        <child link="tip"/>
        <origin xyz="0.2 -0.1 0.3" rpy="0.1 0.2 -0.1"/>
        <axis xyz="0 1 0"/>
        <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
      </joint>
    </robot>"""
    urdf_path = tmp_path / "late_root.urdf"
    urdf_path.write_text(urdf, encoding="utf-8")
    kots = Kots.from_urdf_file(str(urdf_path), order=3, add_world_link=False)
    pin_model = pin.buildModelFromUrdf(str(urdf_path))
    pin_model.gravity.linear[:] = 0.0

    assert kots.robot_.links[0].name == "base"
    q = np.array([0.3])
    v = np.array([0.4])
    a = np.array([-0.5])
    expected = pin.rnea(pin_model, pin_model.createData(), q, v, a)

    np.testing.assert_allclose(
        kots.inverse_dynamics(q, v, a, gravity=np.zeros(3)),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    kots.import_motion_array(np.stack([q, v, a], axis=-1))
    kots.dynamics(backend="numpy", materialize_dict=False)
    actual_numpy = np.asarray(
        kots.state_info(StateType("total_joint", "total_joint", "torque"))
    ).reshape(-1)
    np.testing.assert_allclose(actual_numpy, expected, rtol=1e-12, atol=1e-12)


def test_inverse_dynamics_validates_gravity_shape(tmp_path):
    urdf_path = tmp_path / "comparison.urdf"
    urdf_path.write_text(URDF, encoding="utf-8")
    kots = Kots.from_urdf_file(str(urdf_path), order=3)

    with pytest.raises(ValueError, match="gravity must have shape"):
        kots.inverse_dynamics(np.zeros(2), np.zeros(2), np.zeros(2), gravity=np.zeros(2))
