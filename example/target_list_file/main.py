import numpy as np

from robokots.kots import *

def main():
    kots = Kots.from_json_file("../model/sample_robot.json")
    kots.print_structure()
  
    kots.set_target_from_file("target_list.json")
    kots.print_targets()

    motion = np.random.rand(kots.order()*kots.dof())
    kots.import_motions(motion)
    
    kots.kinematics()

    fk_vel = kots.state_target_link_info('vel')
    jacob_vel = kots.link_jacobian_target(1)@kots.motion("veloc")

    for i in range(len(fk_vel)):
      print(fk_vel[i])
      print(jacob_vel[6*i:6*(i+1)])
  
if __name__ == "__main__":
    main()