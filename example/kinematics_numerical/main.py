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

ORDER = 2

def main():
    kots = Kots.from_json_file("../model/sample_robot.json", order=ORDER)

    motion = np.random.rand(kots.order()*kots.dof())

    kots.import_motions(motion)

    kots.set_target_from_file("target_list.json")

    kots.kinematics()
    
    print("velocity analytical : ", kots.state_target_link_info("vel"))
    print("velocity numerical  : ", link_kinematics_num(kots))

if __name__ == "__main__":
    main()