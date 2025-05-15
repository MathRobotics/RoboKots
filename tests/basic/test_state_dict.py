import numpy as np

from mathrobo import SE3, CMTM
from robokots.outward import *

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

def test_count_dict_order():
    state = {"link_rot": []}
    assert count_dict_order(state) == 1

    state.update({"link_pos": []})
    assert count_dict_order(state) == 1

    state.update({"link_vel": []})
    assert count_dict_order(state) == 2

    state.update({"link_acc": []})
    assert count_dict_order(state) == 3

    state.update({"link_acc_diff1": []})
    assert count_dict_order(state) == 4

    state.update({"link_acc_diff2": []})
    assert count_dict_order(state) == 5

def test_cmtm_to_state_dict():
    cmtm = CMTM.rand(SE3)
    name = "link"
    
    state_data = cmtm_to_state_dict(cmtm, name)

    state_dict = {}
    state_dict.update(state_data)
    
    # Check if the state data contains the expected keys
    assert "link_pos" in state_dict.keys()
    assert "link_rot" in state_dict.keys()
    assert "link_vel" in state_dict.keys()
    assert "link_acc" in state_dict.keys()

    # Check if the state data has the expected values
    assert np.allclose(state_dict["link_pos"], cmtm.elem_mat()[:3, 3])
    assert np.allclose(state_dict["link_rot"], cmtm.elem_mat()[:3, :3].ravel())
    assert np.allclose(state_dict["link_vel"], cmtm.elem_vecs(0))
    assert np.allclose(state_dict["link_acc"], cmtm.elem_vecs(1))

def test_dict_to_rot():
    state = {
        "link_rot": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    }
    name = "link"
    
    rot = state_dict_to_rot(state, name)
    assert np.allclose(rot, np.eye(3))

def test_state_dict_to_frame():
    state = {
        "link_pos": [1.0, 2.0, 3.0],
        "link_rot": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    }
    name = "link"
    
    se3 = state_dict_to_frame(state, name)
    assert np.allclose(se3.rot(), np.eye(3))
    assert np.allclose(se3.pos(), [1.0, 2.0, 3.0])

def test_state_dict_to_cmtm():
    state = {
        "link_pos": [1.0, 2.0, 3.0],
        "link_rot": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "link_vel": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "link_acc": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        "link_acc_diff1" : [1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
    }
    name = "link"
    
    cmtm = state_dict_to_cmtm(state, name)
    assert np.allclose(cmtm.elem_mat()[:3, :3], np.eye(3))
    assert np.allclose(cmtm.elem_mat()[:3, 3], [1.0, 2.0, 3.0])
    assert np.allclose(cmtm.elem_vecs(0), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert np.allclose(cmtm.elem_vecs(1), [0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    assert np.allclose(cmtm.elem_vecs(2), [1.3, 1.4, 1.5, 1.6, 1.7, 1.8])

def test_extract_dict_link_info():
    state = {
        "link_pos": [1.0, 2.0, 3.0],
        "link_rot": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "link_vel": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "link_acc": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    }
    name = "link"
    
    assert np.allclose(extract_dict_link_info(state, "pos", name), [1.0, 2.0, 3.0])
    assert np.allclose(extract_dict_link_info(state, "rot", name), np.eye(3))
    assert np.allclose(extract_dict_link_info(state, "vel", name), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert np.allclose(extract_dict_link_info(state, "acc", name), [0.7, 0.8, 0.9, 1.0, 1.1, 1.2])

def test_extract_dict_info():
    state = {
        "link_pos": [1.0, 2.0, 3.0],
        "link_rot": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "link_vel": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "link_acc": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    }
    name = "link"
    
    assert np.allclose(extract_dict_info(state, "pos", "link", name), [1.0, 2.0, 3.0])
    assert np.allclose(extract_dict_info(state, "rot", "link", name), np.eye(3))
    assert np.allclose(extract_dict_info(state, "vel", "link", name), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert np.allclose(extract_dict_info(state, "acc", "link", name), [0.7, 0.8, 0.9, 1.0, 1.1, 1.2])