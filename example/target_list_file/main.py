import numpy as np

from robokots.robot import *

def main():
    robot = Robot.from_json_file("sample_robot.json")
    robot.print_structure()
  
    robot.set_target_from_file("target_list.json")
    robot.print_targets()

    coord = [1., -1., 1.]
    veloc = [0., 0., 0.]
    accel = [0., 0., 0.]
    force = [0., 0., 0.]
    
    vec = []
    vec.extend(coord)
    vec.extend(veloc)
    vec.extend(accel)
    vec.extend(force)
    
    robot.import_motions(vec)
    
    robot.kinematics()

    print(robot.link_jacobian_target())
  
if __name__ == "__main__":
    main()