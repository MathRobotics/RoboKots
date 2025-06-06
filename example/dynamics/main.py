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
  
  motion = np.random.rand(kots.order()*kots.dof())
  kots.import_motions(motion)
  print(kots.motions())
  
  kots.dynamics()
  
if __name__ == "__main__":
    main()