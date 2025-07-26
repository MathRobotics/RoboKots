import numpy as np

import mathrobo as mr
from robokots.kots import *
from robokots.kinematics.kinematics_jax import *

METHOD = "poly"
ORDER = 3
kots = Kots.from_json_file("./test_model/sample_robot.json", order=ORDER)

motion = np.random.rand(kots.order()*kots.dof())

kots.import_motions(motion)
kots.set_target_from_file("target_list.json")

kots.kinematics()

def test_kinematics():
    h_list = kots.state_link_info_list("frame", kots.link_name_list())
    v_list = kots.state_link_info_list("vel", kots.link_name_list())
    a_list = kots.state_link_info_list("acc", kots.link_name_list())    

    h_list2 = forward_kinematics(kots.robot_.joints, kots.motion(order=1))
    v_list2 = forward_kinematics_vel(kots.robot_.joints, kots.motion(order=2))
    a_list2 = forward_kinematics_acc(kots.robot_.joints, kots.motion(order=3))

    for i in range(len(h_list)):
        assert np.allclose(h_list[i].mat(), h_list2[i].mat())
        assert np.allclose(v_list[i], v_list2[i])
        assert np.allclose(a_list[i], a_list2[i])

def test_kinematics_numerical():
    jark = np.random.rand(kots.dof())
    motion_diff = kots.motion_diff(ORDER, jark)    

    jacob = kots.link_jacobian_target(ORDER)

    ana_vel = kots.state_target_link_info("vel")[-1]
    num_vel = kots.link_diff_kinematics_numerical(kots.target_.target_names, "frame", order = 3, update_direction=jark)

    vec = kots.link_diff_kinematics_numerical(kots.target_.target_names, "cmtm", ORDER, update_direction=jark)
    num_vel2 = vec[:,:6]

    jac_vel = jacob[:6] @ motion_diff

    assert np.allclose(ana_vel, num_vel)
    assert np.allclose(ana_vel, num_vel2)
    assert np.allclose(ana_vel, jac_vel)
    
    ana_acc = kots.state_target_link_info("acc")[-1]
    num_acc = kots.link_diff_kinematics_numerical(kots.target_.target_names, "vel", order = 3, update_direction=jark)
    num_acc2 = vec[:,6:12]
    jac_acc = jacob[6:12] @ motion_diff

    assert np.allclose(ana_acc, num_acc)
    assert np.allclose(ana_acc, num_acc2)
    assert np.allclose(ana_acc, jac_acc)
    
    num_jark = kots.link_diff_kinematics_numerical(kots.target_.target_names, "acc", order = 3, update_direction=jark)
    num_jark2 = vec[:,12:18]
    assert np.allclose(num_jark, num_jark2)
    
def test_jacobian_numerical():
    jacob = kots.link_jacobian_target(1)
    jacob_num = kots.link_jacobian_target_numerical("frame")

    assert np.allclose(jacob, jacob_num)
    
def test_cmtm_jacobian_numerical():
    kots = Kots.from_json_file("./test_model/sample_robot.json", order=5)

    motion = np.random.rand(kots.order()*kots.dof())

    kots.import_motions(motion)
    kots.set_target_from_file("target_list.json")  

    kots.kinematics()  

    jacob_cmtm = kots.link_jacobian_target(ORDER)
    jacob_cmtm_num = kots.link_jacobian_target_numerical("cmtm", ORDER)

    assert np.allclose(jacob_cmtm, jacob_cmtm_num, atol=1e-6, rtol=1e-6)
    
def test_cmtm_jacobian_numerical_soft():
    kots = Kots.from_json_file("./test_model/soft_rod.json", order=5)

    motion = np.random.rand(kots.order()*kots.dof())

    kots.import_motions(motion)

    kots.kinematics()  

    jacob_cmtm = kots.link_jacobian(["end"], ORDER)
    jacob_cmtm_num = kots.link_jacobian_numerical(["end"], "cmtm", ORDER)

    assert np.allclose(jacob_cmtm, jacob_cmtm_num, atol=1e-6, rtol=1e-6)