import numpy as np

from mathrobo import SE3, CMTM
from robokots.core.state_dict import *

def test_extract_state_keys():
    state = {
        "link_pos": [],
        "link_rot": [],
        "link_vel": [],
        "link_acc": [],
        "link_acc_diff1": [],
        "link_acc_diff2": [],
        "link_jerk": [],
        "link_snap": []
    }
    name = "link"
    
    keys = extract_state_keys(state, name)
    
    assert keys == ["pos", "rot", "vel", "acc", "acc_diff1", "acc_diff2", "jerk", "snap"]

def test_count_dict_time_order():
    state = {"link_rot": []}
    assert count_dict_time_order(state) == 1

    state.update({"link_pos": []})
    assert count_dict_time_order(state) == 1

    state.update({"link_vel": []})
    assert count_dict_time_order(state) == 2

    state.update({"link_acc": []})
    assert count_dict_time_order(state) == 3

    state.update({"link_acc_diff1": []})
    assert count_dict_time_order(state) == 4

    state.update({"link_acc_diff2": []})
    assert count_dict_time_order(state) == 5

def test_cmtm_to_state_list():
    cmtm = CMTM.rand(SE3)
    name = "arm"

    state_data = cmtm_to_state_list(cmtm, "link", name)

    state_dict = {}
    state_dict.update(state_data)
    
    # Check if the state data contains the expected keys
    assert "arm_link_pos" in state_dict.keys()
    assert "arm_link_rot" in state_dict.keys()
    assert "arm_link_vel" in state_dict.keys()
    assert "arm_link_acc" in state_dict.keys()

    # Check if the state data has the expected values
    assert np.allclose(state_dict["arm_link_pos"], cmtm.elem_mat()[:3, 3])
    assert np.allclose(state_dict["arm_link_rot"], cmtm.elem_mat()[:3, :3].ravel())
    assert np.allclose(state_dict["arm_link_vel"], cmtm.elem_vecs(0))
    assert np.allclose(state_dict["arm_link_acc"], cmtm.elem_vecs(1))

def test_dict_to_rot():
    state = {
        "arm_link_rot": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    }
    name = "arm"
    
    rot = state_dict_to_rot(state, owner_name=name, owner_type="link")
    assert np.allclose(rot, np.eye(3))

def test_state_dict_to_frame():
    state = {
        "arm_link_pos": [1.0, 2.0, 3.0],
        "arm_link_rot": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    }
    name = "arm"
    
    se3 = state_dict_to_frame(state, name)
    assert np.allclose(se3.rot(), np.eye(3))
    assert np.allclose(se3.pos(), [1.0, 2.0, 3.0])

def test_state_dict_to_cmtm():
    state = {
        "arm_link_pos": [1.0, 2.0, 3.0],
        "arm_link_rot": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "arm_link_vel": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "arm_link_acc": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        "arm_link_acc_diff1" : [1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
    }
    name = "arm"
    
    cmtm = state_dict_to_cmtm(state, name)
    assert np.allclose(cmtm.elem_mat()[:3, :3], np.eye(3))
    assert np.allclose(cmtm.elem_mat()[:3, 3], [1.0, 2.0, 3.0])
    assert np.allclose(cmtm.elem_vecs(0), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert np.allclose(cmtm.elem_vecs(1), [0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    assert np.allclose(cmtm.elem_vecs(2), [1.3, 1.4, 1.5, 1.6, 1.7, 1.8])


def test_state_dict_to_cmtm_reuses_cached_object():
    state = {
        "arm_link_pos": [1.0, 2.0, 3.0],
        "arm_link_rot": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "arm_link_vel": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "arm_link_acc": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
    }

    cmtm0 = state_dict_to_cmtm(state, "arm")
    cmtm1 = state_dict_to_cmtm(state, "arm")

    assert cmtm0 is cmtm1


def test_state_dict_to_rel_cmtm_reuses_cached_object():
    state = {
        "base_link_pos": [0.0, 0.0, 0.0],
        "base_link_rot": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "base_link_vel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "base_link_acc": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "arm_link_pos": [1.0, 2.0, 3.0],
        "arm_link_rot": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "arm_link_vel": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "arm_link_acc": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
    }

    rel0 = state_dict_to_rel_cmtm(state, "base", "arm")
    rel1 = state_dict_to_rel_cmtm(state, "base", "arm")

    assert rel0 is rel1

def test_extract_dict_link_info():
    state = {
        "arm_link_pos": [1.0, 2.0, 3.0],
        "arm_link_rot": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "arm_link_vel": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "arm_link_acc": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    }
    name = "arm"
    
    assert np.allclose(extract_dict_link_info(state, "pos", name), [1.0, 2.0, 3.0])
    assert np.allclose(extract_dict_link_info(state, "rot", name), np.eye(3))
    assert np.allclose(extract_dict_link_info(state, "vel", name), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert np.allclose(extract_dict_link_info(state, "acc", name), [0.7, 0.8, 0.9, 1.0, 1.1, 1.2])


def test_extract_dict_link_info_world_wrench():
    state = {
        "arm_link_pos": [0.0, 0.0, 0.0],
        "arm_link_rot": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "arm_link_momentum": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "arm_link_force": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
    }
    name = "arm"

    assert np.allclose(
        extract_dict_link_info(state, "momentum", name, frame="world"),
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    assert np.allclose(
        extract_dict_link_info(state, "force", name, frame="world"),
        [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
    )


def test_extract_dict_joint_info_world_wrench():
    state = {
        "joint1_joint_pos": [0.0, 0.0, 0.0],
        "joint1_joint_rot": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "joint1_joint_momentum": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "joint1_joint_force": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    }

    assert np.allclose(
        extract_dict_joint_info(state, "momentum", "joint1", frame="world"),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    assert np.allclose(
        extract_dict_joint_info(state, "force", "joint1", frame="world"),
        [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    )

def test_extract_dict_info():
    state = {
        "arm_link_pos": [1.0, 2.0, 3.0],
        "arm_link_rot": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "arm_link_vel": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "arm_link_acc": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    }
    name = "arm"
    
    assert np.allclose(extract_dict_info(state, "pos", "link", name), [1.0, 2.0, 3.0])
    assert np.allclose(extract_dict_info(state, "rot", "link", name), np.eye(3))
    assert np.allclose(extract_dict_info(state, "vel", "link", name), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert np.allclose(extract_dict_info(state, "acc", "link", name), [0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
