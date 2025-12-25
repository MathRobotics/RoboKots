import numpy as np

import mathrobo as mr
from robokots.kots import *
from robokots.core.kinematics.kinematics_jax import *

METHOD = "poly"
kots = Kots.from_json_file("./test_model/sample_robot.json")

motion = np.random.rand(kots.order()*kots.dof())

kots.import_motions(motion)
kots.set_target_from_file("target_list.json")

kots.kinematics()

def test_kinematics():

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

def test_kinematics_numerical():
    dv = np.random.rand(kots.dof())
    motion_diff = kots.motion_diff(kots.order(), dv)

    vec = kots.link_diff_kinematics_numerical([kots.target_._targets[0].owner_name], "cmtm", kots.order(), update_direction=dv)

    alias = ["frame", "vel", "acc", "jerk", "snap"]

    for i in range(kots.order()-1):
        ana_vec = kots.state_info(StateType(data_type=alias[i+1], owner_type = "link", owner_name=kots.target_._targets[0].owner_name)) 
        num_vec = kots.link_diff_kinematics_numerical([kots.target_._targets[0].owner_name], alias[i], order = kots.order(), update_direction=dv)

        num_vec2 = vec[:,6*i:6*(i+1)]

        jacob = kots.jacobian(StateType(data_type=alias[i], owner_type = "link", owner_name=kots.target_._targets[0].owner_name))
        jac_vec = jacob @ motion_diff[:kots.dof()*(i+1)]

        assert np.allclose(ana_vec, num_vec)
        assert np.allclose(ana_vec, num_vec2)
        assert np.allclose(ana_vec, jac_vec)
    
def test_jacobian_numerical():
    kots = Kots.from_json_file("./test_model/sample_robot.json")

    motion = np.random.rand(kots.order()*kots.dof())

    kots.import_motions(motion)
    kots.set_target_from_file("target_list.json")  

    kots.kinematics()  

    jacob_cmtm = kots.jacobian_target()
    jacob_cmtm_num = kots.jacobian_target(numerical=True)

    assert np.allclose(jacob_cmtm, jacob_cmtm_num, atol=1e-5, rtol=1e-5 )
    
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