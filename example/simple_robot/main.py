import numpy as np

import robokots

def main():
  robot = robokots.Robot.init_from_model_file("simple_robot.kts") 
  
  coord = [1., -2., 1.]
  veloc = [0., 0., 0.]
  accel = [0., 0., 0.]
  force = [0., 0., 0.]
  
  vecs = [coord, veloc, accel, force]
  vec = []
  vec.extend(coord)
  vec.extend(veloc)
  vec.extend(accel)
  vec.extend(force)
  
  robot.import_gen_vecs(vecs)
  print(robot.gen_value.df.df)
  
  robot.import_motions(vec)
  print(robot.motions.motions)
  
  robot.update_kinematics()
  
  print(robot.state.df.df)

  robokots.show_robot(robot, robot.state)
  
if __name__ == "__main__":
    main()