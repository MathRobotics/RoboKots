import numpy as np

from robokots.kots import *

def main():
    kots = Kots.from_json_file("../model/sample_robot.json")
    kots.print_structure()
  
    kots.set_target_from_file("target_list.json")
    kots.print_targets()

    coord = [1., -1., 1.]
    veloc = [1., 2., 3.]
    accel = [0., 0., 0.]
    
    vec = []
    vec.extend(coord)
    vec.extend(veloc)
    vec.extend(accel)
    
    kots.import_motions(vec)
    
    kots.kinematics()

    fk_vel = kots.state_target_link_info('vel')
    jacob_vel = kots.link_jacobian_target(1)@kots.motion("veloc")

    for i in range(len(fk_vel)):
      print(fk_vel[i])
      print(jacob_vel[6*i:6*(i+1)])
  
if __name__ == "__main__":
    main()