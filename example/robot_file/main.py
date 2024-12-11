import numpy as np

from robokots.robot_model import RobotStruct
from robokots.robot_io import *

def main():
  file_path = "sample_robot.json"
  data = load_json(file_path)

  robot = RobotStruct.from_json(data)
  robot.print_structure()
  
if __name__ == "__main__":
    main()