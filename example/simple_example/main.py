import numpy as np

import mathrobo as mr
from robokots.kots import *

def main():
    kots = Kots.from_json_file("../model/sample_robot.json", 5)
    kots.print_structure()

    kots.set_target_from_file("target_list.json")
    kots.print_targets()

    motion = np.random.rand(kots.order()*kots.dof())
    kots.import_motions(motion)

    kots.kinematics()

    jacob = kots.link_jacobian_target(1)
    jacob_num = kots.link_jacobian_target_numerical("frame")
    print("norm: ", np.linalg.norm(jacob - jacob_num))

    jacob = kots.link_jacobian_target(4)
    jacob_target = kots.jacobian_target()
    print("norm: ", np.linalg.norm(jacob - jacob_target))

    fk_vel = kots.state_target_link_info('vel')
    jacob_vel = kots.link_jacobian_target(1)@kots.motion_diff(1)

    for i in range(len(fk_vel)):
      print(fk_vel[i])
      print(jacob_vel[6*i:6*(i+1)])

    fk_acc = kots.state_target_link_info('acc')
    jacob_acc = kots.link_jacobian_target(2)@kots.motion_diff(2)

    for i in range(len(fk_acc)):
      print(fk_acc[i])
      print(jacob_acc[12*i+6:12*i+12])
    
    print(kots.target_info_vecs())

    kots.dynamics()
  
    kots.show_robot()

if __name__ == "__main__":
    main()