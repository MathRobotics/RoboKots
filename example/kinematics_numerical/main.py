import numpy as np

import mathrobo as mr
from robokots.kots import *

METHOD = "poly"
ORDER = 3

def main():
    kots = Kots.from_json_file("../model/sample_robot.json", order=ORDER)

    motion = np.random.rand(kots.order()*kots.dof())

    kots.import_motions(motion)

    kots.set_target_from_file("target_list.json")

    kots.kinematics()

    jark = np.random.rand(kots.dof())
    # jark = np.ones(kots.dof())

    jacob = kots.link_jacobian_target(ORDER)

    ana_vel = kots.state_target_link_info("vel")
    num_vel = kots.link_diff_kinematics_numerical(kots.target_.target_names, "frame", update_direction=jark)

    vec = kots.link_diff_kinematics_numerical(kots.target_.target_names, "cmtm", ORDER, update_direction=jark)
    num_vel2 = vec[:,:6]

    jac_vel = jacob[:6,:kots.dof()] @ motion[kots.dof():2*kots.dof()]

    print("velocity analytical : ", ana_vel)
    print("velocity numerical  : ", num_vel)
    print("velocity numerical2 : ", num_vel2)
    print("velocity jacobian : ", jac_vel)
    print("norm: ", np.linalg.norm(ana_vel - num_vel))

    ana_acc = kots.state_target_link_info("acc")
    num_acc = kots.link_diff_kinematics_numerical(kots.target_.target_names, "vel", update_direction=jark)
    num_acc2 = vec[:,6:12]
    jac_acc = jacob[6:12,:2*kots.dof()] @ motion[kots.dof():3*kots.dof()]
    print("accleration analytical : ", ana_acc)
    print("accleration numerical  : ", num_acc)
    print("accleration numerical2 : ", num_acc2)
    print("accleration jacobian : ", jac_acc)
    print("norm: ", np.linalg.norm(ana_acc - num_acc))

    num_jark = kots.link_diff_kinematics_numerical(kots.target_.target_names, "acc", ORDER, update_direction=jark)
    num_jark2 = vec[:,12:18]
    print("jark numerical : ", num_jark)
    print("jark numerical cmtm : ", num_jark2)
    print("norm: ", np.linalg.norm(num_jark - num_jark2))
    
if __name__ == "__main__":
    main()