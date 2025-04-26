import numpy as np

import mathrobo as mr
from robokots.kots import *

def link_jacobian_num(kots, delta = 1e-8):
    kots.kinematics()

    x = kots.motions()
    p0 = kots.state_target_link_info("frame")
    
    row = 6 * len(p0)
    col = kots.dof()
    
    J = np.zeros((row, col))
    for i in range(col):
        x_ = x.copy()
        x_[i] += delta
        kots.import_motions(x_)
        kots.kinematics()
        p1 = kots.state_target_link_info("frame")
        for j in range(len(p0)):
            dp = mr.SE3.sub_tan_vec(p0[j], p1[j]) / delta
            J[j*6:(j+1)*6,i] = dp

    return J

def main():
    kots = Kots.from_json_file("../model/sample_robot.json")
    motion = np.random.rand(kots.order()*kots.dof())
    kots.import_motions(motion)

    kots.set_target_from_file("target_list.json")

    kots.kinematics()

    jacob = kots.link_jacobian_target(1)

    jacob_num = link_jacobian_num(kots)
    
    print("jacobian shape: ", jacob.shape)
    print("jacobian_num shape: ", jacob_num.shape)

    print("norm: ", np.linalg.norm(jacob - jacob_num))

    for i in range(jacob.shape[0]):
      print(jacob[i])
      print(jacob_num[i])
  
if __name__ == "__main__":
    main()