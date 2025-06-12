import numpy as np

from robokots.kots import *

def main():
  kots = Kots.from_json_file("../model/sample_robot.json", order = 5)
  
  motion = np.random.rand(kots.order()*kots.dof())
  kots.import_motions(motion)
  
  kots.dynamics()
  
if __name__ == "__main__":
    main()