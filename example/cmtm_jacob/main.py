import numpy as np

import mathrobo as mr
from robokots.kots import *
import time

ORDER = 4

def main():
    kots = Kots.from_json_file("../model/sample_robot.json", ORDER)
    target = ["arm3"]

    motion = np.random.rand(kots.order()*kots.dof())
    kots.import_motions(motion)

    kots.kinematics()
    print("motion:", kots.motions())
    velocity = kots.state_link_info_list("vel", target)
    print("velocity:", velocity)
    acceleration = kots.state_link_info_list("acc", target)
    print("acceleration:", acceleration)

    jacob = kots.link_jacobian(target, ORDER)
    jacob_num = kots.link_jacobian_numerical(target, "cmtm", ORDER)

    veloc_jac = jacob[0:6,:2*kots.dof()] @ motion[kots.dof():3*kots.dof()]
    accel_jac = jacob[6:12,:2*kots.dof()] @ motion[kots.dof():3*kots.dof()]

    print("velocity", veloc_jac)
    print("acceleration", accel_jac)

    print("velocity norm: ", np.linalg.norm(velocity - veloc_jac))
    print("acceleration norm: ", np.linalg.norm(acceleration - accel_jac)) 

    print("jacobian_ana shape: ", jacob.shape)
    print("jacobian_num shape: ", jacob_num.shape)

    for i in range(ORDER):
        print("norm: ", np.linalg.norm(jacob[6*i:6*(i+1)] - jacob_num[6*i:6*(i+1)]))

    print("norm: ", np.linalg.norm(jacob - jacob_num))

if __name__ == "__main__":
    main()