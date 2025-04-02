import numpy as np

from robokots.robot import *

def main():
    robot = Robot.from_json_file("sample_robot.json")
    robot.print_structure()
  
    robot.set_target_from_file("target_list.json")
    robot.print_targets()
  
if __name__ == "__main__":
    main()