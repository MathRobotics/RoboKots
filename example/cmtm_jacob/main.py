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
            dp = mr.CMTM.sub_vec(p0[j], p1[j]) / delta
            J[j*row:(j+1)*row,i] = dp[:row]
    return J

ORDER = 3

def main():
    kots = Kots.from_json_file("../model/sample_robot.json")
    target = ["arm2"]

    motion = np.random.rand(kots.order()*kots.dof())
    kots.import_motions(motion)

    kots.kinematics()
    print("motion:", kots.motions())
    velocity = kots.state_link_info_list("vel", target)
    print("velocity:", velocity)
    acceleration = kots.state_link_info_list("acc", target)
    print("acceleration:", acceleration)


    jacob = kots.link_jacobian(target, ORDER)
    jacob_num = link_jacobian_num(kots, target, ORDER)

    veloc_jac = jacob[0:6,:2*kots.dof()] @ motion[kots.dof():]
    accel_jac = jacob[6:12,:2*kots.dof()] @ motion[kots.dof():]

    print("velocity", veloc_jac)
    print("acceleration", accel_jac)

    print("velocity norm: ", np.linalg.norm(velocity - veloc_jac))
    print("acceleration norm: ", np.linalg.norm(acceleration - accel_jac)) 

    print("jacobian_ana shape: ", jacob.shape)
    print("jacobian_num shape: ", jacob_num.shape)

    for i in range(6):
        print("jacobian: ", jacob[12+i:12+i+1])
        print("jacobian_num: ", jacob_num[12+i:12+i+1])
        print("norm: ", np.linalg.norm(jacob[12+i:12+i+1] - jacob_num[12+i:12+i+1]))

    print("norm: ", np.linalg.norm(jacob - jacob_num))

if __name__ == "__main__":
    main()