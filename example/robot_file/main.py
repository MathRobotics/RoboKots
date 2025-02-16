import numpy as np

from robokots.robot import *

def main():
  robot = Robot.from_json_file("sample_robot.json")
  robot.print_structure()
  
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
  print(robot.motions())
  
  robot.kinematics()
  
  print(robot.state_df())

  robot.show_robot()
  
if __name__ == "__main__":
    main()