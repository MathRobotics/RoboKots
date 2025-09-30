import numpy as np

import mathrobo as mr
from robokots.kots import *

METHOD = "poly"

def main():
    kots = Kots.from_json_file("../model/sample_robot.json")
    kots.set_target_from_file("target_list.json")

    motion = np.random.rand(kots.order()*kots.dof())
    print(motion)
    kots.import_motions(motion)

    kots.kinematics()

    jark = np.random.rand(kots.dof())
    # jark = np.ones(kots.dof())
    motion_diff = kots.motion_diff(kots.order(), jark)

    jacob_cmtm = kots.jacobian_cmtm(kots.target_.target_names)

    ana_vel = kots.state_target_link_info("vel")[-1]
    num_vel = kots.link_diff_kinematics_numerical(kots.target_.target_names, "frame", order = kots.order(), update_direction=jark)

    vec = kots.link_diff_kinematics_numerical(kots.target_.target_names, "cmtm", kots.order(), update_direction=jark)
    num_vel2 = vec[:,:6]

    jac_vel = kots.jacobian_target("frame") @ kots.motion_diff(1)
    jac_vel2 = jacob_cmtm[:6] @ motion_diff

    print("velocity analytical : \n", ana_vel)
    print("velocity numerical  : \n", num_vel)
    print("velocity numerical2 : \n", num_vel2)
    print("velocity jacobian : \n", jac_vel)
    print("velocity jacobian : \n", jac_vel2)
    print("norm: ", np.linalg.norm(ana_vel - num_vel))

    ana_acc = kots.state_target_link_info("acc")[-1]
    num_acc = kots.link_diff_kinematics_numerical(kots.target_.target_names, "vel", order = 3, update_direction=jark)
    num_acc2 = vec[:,6:12]
    jac_acc = kots.jacobian_target("vel") @ kots.motion_diff(2)
    jac_acc2 = jacob_cmtm[6:12] @ motion_diff
    print("accleration analytical : \n", ana_acc)
    print("accleration numerical  : \n", num_acc)
    print("accleration numerical2 : \n", num_acc2)
    print("accleration jacobian : \n", jac_acc)
    print("accleration jacobian : \n", jac_acc2)
    print("norm: ", np.linalg.norm(ana_acc - num_acc))

    num_jark = kots.link_diff_kinematics_numerical(kots.target_.target_names, "acc", order = 3, update_direction=jark)
    num_jark2 = vec[:,12:18]
    print("jark numerical : \n", num_jark)
    print("jark numerical cmtm : \n", num_jark2)
    print("norm: ", np.linalg.norm(num_jark - num_jark2))
    
if __name__ == "__main__":
    main()