import numpy as np

from mathrobo import SE3, CMTM
from robokots.outward import *

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