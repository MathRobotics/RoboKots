import numpy as np

from robokots.kots import *

order = 5
def main():
  kots = Kots.from_json_file("../model/soft_rod.json", order=order)
  # kots.print_structure()

  kots.set_target_from_file("target_list.json")
  # print("Target Links:", kots.target_.target_names)

  motion = np.random.rand(kots.order()*kots.dof())
  kots.import_motions(motion)

  kots.kinematics()

  jacob = kots.link_jacobian_target(order)
  jacob_num = kots.link_jacobian_target_numerical("cmtm", order)
  # jacob_num = kots.link_jacobian_target_numerical("frame")

  print("jacobian shape: ", jacob.shape)
  print("jacobian_num shape: ", jacob_num.shape)

  # print("jacobian: ", jacob)
  # print("jacobian_num: ", jacob_num)
  
  print(f"norm of jacobian: {np.linalg.norm(jacob - jacob_num)}")

if __name__ == "__main__":
    main()