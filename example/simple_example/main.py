import numpy as np

import mathrobo as mr
from robokots.kots import *

def main():
    kots = Kots.from_json_file("../model/sample_robot.json")
    kots.print_structure()

    kots.set_target_from_file("target_list.json")
    kots.print_targets()

    motion = np.random.rand(kots.order()*kots.dof())
    kots.import_motions(motion)

    kots.kinematics()

    jacob = kots.jacobian(["end"], "frame")
    jacob_num = kots.jacobian_numerical(["end"], "frame")
    print("norm: ", np.linalg.norm(jacob - jacob_num))

    jacob = kots.jacobian_target() 
    jacob_target = kots.jacobian_target([["vel", "acc", "jerk", "snap", "force"]])
    print("norm: ", np.linalg.norm(jacob - jacob_target))

    jacob = kots.jacobian(["end"], "force")
    jacob_num = kots.jacobian_numerical(["end"], "force")
    print("norm: ", np.linalg.norm(jacob - jacob_num))

    fk_vel = kots.state_target_link_info('vel')
    jacob_vel = kots.jacobian_target("frame")@kots.motion_diff(1)

    for i in range(len(fk_vel)):
      print(fk_vel[i])
      print(jacob_vel[6*i:6*(i+1)])

    fk_acc = kots.state_target_link_info('acc')
    jacob_acc = kots.jacobian_target("vel")@kots.motion_diff(2)
      
    print(fk_acc[0])
    print(jacob_acc)
    
    kots.dynamics()

    print(kots.target_info())
  
    kots.show_robot()

    print(kots.jacobian_target("force"))

if __name__ == "__main__":
    main()