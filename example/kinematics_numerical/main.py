import numpy as np

import mathrobo as mr
from robokots.kots import *

def integral_mat(dof, order, dt):
    mat = np.eye(dof * order)

    vec = np.zeros((dof * order ,dof))

    for i in range(1,order): 
        for j in range(i):
            mat[j*dof:(j+1)*dof,i*dof:(i+1)*dof] = dt**(i-j) * np.eye(dof)
        
    for i in range(order):
        vec[i*dof:(i+1)*dof] = dt**(order-i) * np.eye(dof)

    return mat, vec

def link_kinematics_vel_num(kots, x_init, jark = None, delta = 1e-8):
    kots.import_motions(x_init)
    kots.kinematics()
    p0 = kots.state_target_link_info("frame")

    dp = np.zeros((len(p0), 6))
    if jark is None:
        jark = np.random.rand(kots.dof())

    x_ = x_init.copy()
    D, d = integral_mat(kots.dof(), 3, delta)
    x_ = D @ x_init + d @ jark

    kots.import_motions(x_)
    kots.kinematics()
    p1 = kots.state_target_link_info("frame")
    
    for i in range(len(p0)):
        dp[i] = mr.SE3.sub_tan_vec(p0[i], p1[i], "bframe") / delta

    return dp
    
    kots.import_motions(x_init)
def link_kinematics_acc_num(kots, x_init, jark = None, delta = 1e-8):
    kots.import_motions(x_init)
    kots.kinematics()
    a0 = kots.state_target_link_info("vel")

    da = np.zeros((len(a0), 6))
    if jark is None:
        jark = np.random.rand(kots.dof())

    x_ = x_init.copy()
    D, d = integral_mat(kots.dof(), 3, delta)
    x_ = D @ x_init + d @ jark

    kots.import_motions(x_)
    kots.kinematics()
    a1 = kots.state_target_link_info("vel")
    
    for i in range(len(a0)):
        da[i] = (a1[i]- a0[i]) / delta
    return da

def link_kinematics_jark_num(kots, x_init, jark = None, delta = 1e-8):
    kots.import_motions(x_init)
    kots.kinematics()
    a0 = kots.state_target_link_info("acc")

    da = np.zeros((len(a0), 6))
    if jark is None:
        jark = np.random.rand(kots.dof())

    x_ = x_init.copy()
    D, d = integral_mat(kots.dof(), 3, delta)
    x_ = D @ x_init + d @ jark

    kots.import_motions(x_)
    kots.kinematics()
    a1 = kots.state_target_link_info("acc")

    for i in range(len(a0)):
        da[i] = (a1[i]- a0[i]) / delta

    return da

def link_kinematics_cmtm_num(kots, x_init, order, jark = None, delta = 1e-8):
    kots.import_motions(x_init)
    kots.kinematics()
    p0 = kots.state_target_link_info("cmtm")
    a0 = kots.state_target_link_info("acc")

    dp = np.zeros((len(p0), 6*order))
    if jark is None:
        jark = np.random.rand(kots.dof())

    x_ = x_init.copy()
    D, d = integral_mat(kots.dof(), order, delta)
    x_ = D @ x_init + d @ jark

    kots.import_motions(x_)
    kots.kinematics()
    p1 = kots.state_target_link_info("cmtm")
    a1 = kots.state_target_link_info("acc")

    for i in range(len(p0)):
        dp[i] = mr.CMTM.sub_vec(p0[i], p1[i]) / delta

    return dp

ORDER = 3

def main():
    kots = Kots.from_json_file("../model/sample_robot.json", order=ORDER)

    motion = np.random.rand(kots.order()*kots.dof())

    kots.import_motions(motion)

    kots.set_target_from_file("target_list.json")

    kots.kinematics()

    jark = np.random.rand(kots.dof())

    ana_vel = kots.state_target_link_info("vel")
    num_vel = link_kinematics_vel_num(kots, motion, jark)
    vec = link_kinematics_cmtm_num(kots, motion, ORDER, jark)
    num_vel2 = vec[:,:6]
    print("velocity analytical : ", ana_vel)
    print("velocity numerical  : ", num_vel)
    print("velocity numerical2 : ", num_vel2)
    print("norm: ", np.linalg.norm(ana_vel - num_vel))

    ana_acc = kots.state_target_link_info("acc")
    num_acc = link_kinematics_acc_num(kots, motion, jark)
    num_acc2 = vec[:,6:12]
    print("accleration analytical : ", ana_acc)
    print("accleration numerical  : ", num_acc)
    print("accleration numerical2 : ", num_acc2)
    print("norm: ", np.linalg.norm(ana_acc - num_acc))

    num_jark1 = link_kinematics_jark_num(kots, motion, jark)
    num_jark2 = vec[:,12:18]
    print("jark numerical : ", num_jark1)
    print("jark numerical cmtm : ", num_jark2)
    print("norm: ", np.linalg.norm(num_jark1 - num_jark2))
    
if __name__ == "__main__":
    main()