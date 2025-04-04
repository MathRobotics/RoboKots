import numpy as np

from robokots.robot import *

def main():
    robot = Robot.from_json_file("sample_robot.json")
    robot.print_structure()
  
    robot.set_target_from_file("target_list.json")
    robot.print_targets()

    coord = [1., -1., 1.]
    veloc = [1., 2., 3.]
    accel = [0., 0., 0.]
    
    vec = []
    vec.extend(coord)
    vec.extend(veloc)
    vec.extend(accel)
    
    robot.import_motions(vec)
    
    robot.kinematics()

    print(robot.state_target_link_info('vel'))

    print(robot.link_jacobian_target()@robot.motion("veloc"))
  
if __name__ == "__main__":
    main()