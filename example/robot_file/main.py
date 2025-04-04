import numpy as np

from robokots.robot import *

def main():
  robot = Robot.from_json_file("sample_robot.json")
  robot.print_structure()
  
  link_name_list = ["world","base","arm1","arm2","arm3"]
  links = robot.link_list(link_name_list)
  print(list(l.type for l in links) )

  joint_name_list = ["root","joint1","joint2","joint3"]
  joints = robot.joint_list(joint_name_list)
  print(list(j.type for j in joints) )
  
  coord = [1., -1., 1.]
  veloc = [0., 0., 0.]
  accel = [0., 0., 0.]
  
  vec = []
  vec.extend(coord)
  vec.extend(veloc)
  vec.extend(accel)
  
  robot.import_motions(vec)
  print(robot.motions())
  
  robot.kinematics()
  
  print(robot.state_df())
  
  print(robot.link_jacobian(["arm1","arm2","arm3"]))

  robot.show_robot()
  
if __name__ == "__main__":
    main()