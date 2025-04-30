import numpy as np

import mathrobo as mr
from robokots.kots import *
import time

def link_jacobian_num(kots, target, order = 3, delta = 1e-8):
    kots.kinematics()

    x = kots.motions()
    p0 = kots.state_link_info_list("cmtm", target)
    
    row = order * 6
    col = order * kots.dof()
    
    J = np.zeros((row, col))
    for i in range(col):
        x_ = x.copy()
        x_[i] += delta
        kots.import_motions(x_)
        kots.kinematics()
        p1 = kots.state_link_info_list("cmtm", target)
        for j in range(len(p0)):
            dp = mr.CMTM.sub_vec(p0[j], p1[j]) / delta
            J[j*row:(j+1)*row,i] = dp[:row]

    return J

ORDER = 3

def main():
    kots = Kots.from_json_file("../model/2dof_arm.json")
    motion = np.random.rand(kots.order()*kots.dof())
    kots.import_motions(motion)

    kots.kinematics()

    target = ["arm2"]
    jacob = kots.link_jacobian(target, ORDER)

    jacob_num = link_jacobian_num(kots, target, ORDER)
    
    print("jacobian_ana shape: ", jacob.shape)
    # print("jacobian: ", jacob)
    print("jacobian_num shape: ", jacob_num.shape)
    # print("jacobian_num: ", jacob_num)

    print("norm: ", np.linalg.norm(jacob[:,:(ORDER*kots.dof())] - jacob_num))

    for i in range(kots.order()):
        print("jacob_ana: ", jacob[i*6:(i+1)*6,:])
        print("jacob_num: ", jacob_num[i*6:(i+1)*6,:])

if __name__ == "__main__":
    main()