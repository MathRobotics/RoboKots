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

    print(kots.link_cmtm_jacobian_target())
  
if __name__ == "__main__":
    main()