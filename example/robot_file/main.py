import numpy as np

from robokots.robot_model import RobotStruct
from robokots.robot_io import *

def main():
  robot = RobotStruct.from_json_file("sample_robot.json")
  robot.print_structure()
  
if __name__ == "__main__":
    main()