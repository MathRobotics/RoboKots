import numpy as np

from robokots.kots import *

def main():
  kots = Kots.from_json_file("../model/sample_robot.json")
  kots.print_structure()
  
  link_name_list = ["world","base","arm1","arm2","arm3"]
  links = kots.link_list(link_name_list)
  print(list(l.type for l in links) )

  joint_name_list = ["root","joint1","joint2","joint3"]
  joints = kots.joint_list(joint_name_list)
  print(list(j.type for j in joints) )
  
  coord = [1., -1., 1.]
  veloc = [0., 0., 0.]
  accel = [0., 0., 0.]
  
  vec = []
  vec.extend(coord)
  vec.extend(veloc)
  vec.extend(accel)
  
  kots.import_motions(vec)
  print(kots.motions())
  
  kots.kinematics()
  
  print(kots.state_df())
  
  print(kots.link_jacobian(["arm1","arm2","arm3"]))
  
if __name__ == "__main__":
    main()