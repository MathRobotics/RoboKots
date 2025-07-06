import numpy as np

from robokots.kots import *

def main():
  kots = Kots.from_json_file("../model/soft_rod.json")
  kots.print_structure()
  
if __name__ == "__main__":
    main()