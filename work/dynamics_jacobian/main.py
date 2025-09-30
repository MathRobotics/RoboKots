import numpy as np

import mathrobo as mr
from robokots import kots
from robokots.kots import *
from robokots import *

def main():
    kots = Kots.from_json_file("../model/sample_robot.json")
    kots.set_target_from_file("target_list.json")

    motion = np.random.rand(kots.order()*kots.dof())
    kots.import_motions(motion)
    
    kots.dynamics()

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
    jac_joint_momentum = outward_matrix.joint_jacobian_momentum(kots.robot_, kots.state_dict_, ['joint4'],kots.order()-1)@kots.motion_diff(order=kots.order()-1)
    print(kots.state_joint_info('momentum', 'joint4'))
    print(kots.state_joint_info('momentum_diff1', 'joint4'))
    print(kots.state_joint_info('momentum_diff2', 'joint4'))
    print(kots.state_joint_info('momentum_diff3', 'joint4'))
    print(kots.state_joint_info('momentum_diff4', 'joint4'))
    print(jac_joint_momentum)
    print("joint_force")
    print(kots.state_joint_info('force', 'joint4'))
    print(kots.state_joint_info('force_diff1', 'joint4'))
    print(kots.state_joint_info('force_diff2', 'joint4'))
    print(kots.state_joint_info('force_diff3', 'joint4'))

if __name__ == "__main__":
    main()