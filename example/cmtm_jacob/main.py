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
    
    J = np.zeros((row * len(p0), col))
    for i in range(col):
        x_ = x.copy()
        x_[i] += delta
        kots.import_motions(x_)
        kots.kinematics()
        p1 = kots.state_link_info_list("cmtm", target)
        for j in range(len(p0)):
            dp = mr.CMTM.sub_ptan_vec(p0[j], p1[j]) / delta
            J[j*row:(j+1)*row,i] = \
                CMTM.tan_to_ptan(6, p0[j]._n) @ p0[j].tan_mat_inv_adj() @ mr.CMTM.ptan_to_tan(6, order) @ dp[:row]

    return J

ORDER = 3

def main():
    kots = Kots.from_json_file("../model/sample_robot.json")

    motion = np.random.rand(kots.order()*kots.dof())
    kots.import_motions(motion)

    kots.kinematics()
    print("motion:", kots.motions())
    print("velocity:", kots.state_link_info_list("vel", ["arm3"]))
    print("acceleration:", kots.state_link_info_list("acc", ["arm3"]))

    target = ["arm1","arm2","arm3"]
    jacob = kots.link_jacobian(target, ORDER)

    # for l in kots.robot_.link_names:
    #     cmtm = kots.state_.link_rel_cmtm(l, target[0], ORDER)
    #     print("name: ", l)
    #     cmtm.print()

    jacob_num = link_jacobian_num(kots, target, ORDER)

    print("vel", jacob[0:6,:] @ motion)
    print("acc", jacob[6:12,:] @ motion)
    
    print("jacobian_ana shape: ", jacob.shape)
    # print("jacobian: ", jacob)
    print("jacobian_num shape: ", jacob_num.shape)
    # print("jacobian_num: ", jacob_num)

    print("norm: ", np.linalg.norm(jacob - jacob_num))

if __name__ == "__main__":
    main()