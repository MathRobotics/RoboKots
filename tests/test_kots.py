import numpy as np
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
