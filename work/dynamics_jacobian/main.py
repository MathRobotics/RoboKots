import numpy as np

import mathrobo as mr
from robokots.kots import *
from robokots import *

def main():
    kots = Kots.from_json_file("../model/sample_robot.json")
    kots.print_structure()

    kots.set_target_from_file("target_list.json")
    kots.print_targets()

    motion = np.random.rand(kots.order()*kots.dof())
    kots.import_motions(motion)
    
    kots.dynamics()

    # coord_to_joint = outward_matrix.coord_to_joint_mat(kots.robot_, order=kots.order())
    # print(np.count_nonzero(coord_to_joint))
    # # print(coord_to_joint[6*3:])
    # joint_to_link = outward_matrix.joint_to_link_mat(kots.robot_, kots.state_dict_, order=kots.order())
    # print(np.count_nonzero(joint_to_link))
    # joint_to_link[np.nonzero(joint_to_link)] = 1
    # # print(joint_to_link[6*3:, 6*3:])

    # # print(joint_to_link @ coord_to_joint)

    # coord_to_link_momentum = outward_matrix.coord_to_link_momentum_mat(kots.robot_, kots.state_dict_, order=kots.order())
    # print(np.count_nonzero(coord_to_link_momentum))
    # # print(coord_to_link_momentum)

    # coord_to_link_force = outward_matrix.coord_to_link_force_mat(kots.robot_, kots.state_dict_, order=kots.order())
    # print(np.count_nonzero(coord_to_link_force))
    # # print(coord_to_link_force)

    # total_momentum_to_force = outward_matrix.total_momentum_to_force_mat(kots.robot_, kots.state_dict_)
    # print(np.count_nonzero(total_momentum_to_force))
    # print(total_momentum_to_force.shape)
  
    kots.print_state_dict()

    print("momentum")
    id_momentum = kots.state_target_link_info('momentum')
    id_momentum_diff = kots.state_target_link_info('momentum_diff1')
    jac_momentum = outward_matrix.link_jacobian_momentum(kots.robot_, kots.state_dict_, kots.target_.target_names, order=kots.order()-1)@kots.motion_diff(kots.order()-1)

    print(id_momentum[0])
    print(id_momentum_diff[0])
    print(jac_momentum)

    print("force")
    id_force = kots.state_target_link_info('force')
    jac_force = outward_matrix.link_jacobian_force(kots.robot_, kots.state_dict_, kots.target_.target_names,)@kots.motion_diff(2)

    print(id_force[0])
    print(jac_force)

if __name__ == "__main__":
    main()