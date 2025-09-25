import numpy as np

import mathrobo as mr
from robokots import kots
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
  
    # kots.print_state_dict()

    print("velocity")
    print(kots.state_target_link_info('vel')[0])
    print(kots.jacobian_target('frame')@kots.motion_diff(1))
    print(kots.state_target_link_info('acc')[0])
    print(kots.jacobian_target('vel')@kots.motion_diff(2))
    print(kots.state_target_link_info('jerk')[0])
    print(kots.jacobian_target('acc')@kots.motion_diff(3))
    print(kots.state_target_link_info('snap')[0])
    print(kots.jacobian_target('jerk')@kots.motion_diff(4))

    print("momentum")
    print(kots.state_target_link_info('momentum_diff1')[0])
    print(kots.jacobian_target('momentum')@kots.motion_diff(2))
    print(kots.state_target_link_info('momentum_diff2')[0])
    print(kots.jacobian_target('momentum_diff1')@kots.motion_diff(3))
    print(kots.state_target_link_info('momentum_diff3')[0])
    print(kots.jacobian_target('momentum_diff2')@kots.motion_diff(4))

    print("force")
    print(kots.state_target_link_info('force_diff1')[0])
    print(kots.jacobian_target('force')@kots.motion_diff(3))
    print(kots.state_target_link_info('force_diff2')[0])
    print(kots.jacobian_target('force_diff1')@kots.motion_diff(4))
    print(kots.state_target_link_info('force_diff3')[0])
    print(kots.jacobian_target('force_diff2')@kots.motion_diff(5))

    print("joint_momentum")
    # jac_joint_momentum = outward_matrix.link_jacobian_joint_momentum(kots.robot_, kots.state_dict_, kots.target_.target_names,kots.order()-2)@kots.motion_diff(order=kots.order())
    print(kots.state_joint_info('momentum', 'joint4'))
    print(kots.state_joint_info('momentum_diff1', 'joint4'))
    print(kots.state_joint_info('momentum_diff2', 'joint4'))
    print(kots.state_joint_info('momentum_diff3', 'joint4'))
    # print(kots.state_target_link_info('joint_momentum_diff1')[0])
    # print(kots.state_target_link_info('joint_momentum_diff2')[0])
    # print(kots.state_target_link_info('joint_momentum_diff3')[0])
    # print(jac_joint_momentum)

    print("joint_force")
    print(kots.state_joint_info('force', 'joint4'))
    print(kots.state_joint_info('force_diff1', 'joint4'))
    print(kots.state_joint_info('force_diff2', 'joint4'))
    print(kots.state_joint_info('force_diff3', 'joint4'))

if __name__ == "__main__":
    main()