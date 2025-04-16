import numpy as np

from robokots.kots import *
import time

def main():
    kots = Kots.from_json_file("../model/sample_robot.json")

    coord = [1., 2., 3.]
    veloc = [4., 5., 6.]
    accel = [0., 0., 0.]
    
    vec = []
    vec.extend(coord)
    vec.extend(veloc)
    vec.extend(accel)    
    kots.import_motions(vec)
    
    kots.kinematics()

    target = ["arm1","arm2","arm3"]
    kots.link_jacobian(target,3)
  
if __name__ == "__main__":
    main()