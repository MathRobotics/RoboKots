import numpy as np

from robokots.robot import *

def main():
    robot = Robot.from_json_file("../model/sample_robot.json")
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

    fk_vel = robot.state_target_link_info('vel')
    jacob_vel = robot.link_jacobian_target()@robot.motion("veloc")

    for i in range(len(fk_vel)):
      print(fk_vel[i])
      print(jacob_vel[6*i:6*(i+1)])
  
if __name__ == "__main__":
    main()