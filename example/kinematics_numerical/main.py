import numpy as np

import mathrobo as mr
from robokots.kots import *

def link_kinematics_num(kots, delta = 1e-8):
    kots.kinematics()

    x = kots.motions()
    p0 = kots.state_target_link_info("frame")

    dp = np.zeros((len(p0), 6))

    x_ = x.copy()
    x_[0:kots.dof()] += x_[kots.dof():2*kots.dof()] * delta

    kots.import_motions(x_)
    kots.kinematics()
    p1 = kots.state_target_link_info("frame")
    
    for i in range(len(p0)):
        dp[i] = mr.SE3.sub_tan_vec(p0[i], p1[i], "bframe") / delta

    return dp

def link_kinematics_cmtm_num(kots, order, delta = 1e-8):
    kots.kinematics()

    x = kots.motions()
    p0 = kots.state_target_link_info("cmtm")

    dp = np.zeros((len(p0), 6*order))

    x_ = x.copy()
    x_[0:kots.dof()] += x_[kots.dof():2*kots.dof()] * delta

    kots.import_motions(x_)
    kots.kinematics()
    p1 = kots.state_target_link_info("cmtm")
    
    for i in range(len(p0)):
        tmp = mr.CMTM.sub_ptan_vec(p0[i], p1[i]) / delta
        dp[i] =p0[i].tan_mat_inv_adj() @ mr.CMTM.ptan_to_tan(6, order) @ tmp
    return dp

ORDER = 3

def main():
    kots = Kots.from_json_file("../model/sample_robot.json", order=ORDER)

    motion = np.random.rand(kots.order()*kots.dof())

    kots.import_motions(motion)

    kots.set_target_from_file("target_list.json")

    kots.kinematics()

    ana_vel = kots.state_target_link_info("vel")
    num_vel = link_kinematics_num(kots)

    print("velocity analytical : ", ana_vel)
    print("velocity numerical  : ", num_vel)
    print("norm: ", np.linalg.norm(ana_vel - num_vel))

    ana_acc = kots.state_target_link_info("acc")
    vec = link_kinematics_cmtm_num(kots, order=ORDER)
    num_acc = vec[:,6:12]
    print("accleration analytical : ", ana_acc)
    print("accleration numerical  : ", num_acc)
    print("norm: ", np.linalg.norm(ana_acc - num_acc))

if __name__ == "__main__":
    main()