import numpy as np

import mathrobo as mr
from robokots.kots import *

def main():
    kots = Kots.from_json_file("../model/sample_robot.json")
    motion = np.random.rand(kots.order()*kots.dof())
    kots.import_motions(motion)

    kots.set_target_from_file("target_list.json")

    kots.kinematics()

    jacob = kots.link_jacobian_target(1)

    jacob_num = kots.link_jacobian_target_numerical("frame")
    
    print("jacobian shape: ", jacob.shape)
    print("jacobian_num shape: ", jacob_num.shape)

    print("norm: ", np.linalg.norm(jacob - jacob_num))

    # for i in range(jacob.shape[0]):
    #     print("jac ana\n", jacob[i])
    #     print("jac num\n", jacob_num[i])
  
if __name__ == "__main__":
    main()