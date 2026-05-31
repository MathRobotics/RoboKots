import numpy as np
from pathlib import Path

import mathrobo as mr
from robokots.kots import *

EXAMPLE_DIR = Path(__file__).resolve().parent

def main():
    kots = Kots.from_json_file(str(EXAMPLE_DIR.parent / "model" / "sample_robot.json"))
    kots.print_structure()

    kots.set_target_from_file(str(EXAMPLE_DIR / "target_list.json"))
    print(kots.targets())

    kots.set_order(6)

    motion = np.random.rand(kots.order()*kots.dof())
    kots.import_motions(motion)

    kots.kinematics()

    state = StateType('link','end','frame')
    jacob = kots.jacobian(state)
    jacob_num =  kots.jacobian(state, numerical=True)
    print("norm: ", np.linalg.norm(jacob - jacob_num))

    state = StateType('link','end','snap')
    jacob = kots.jacobian(state)
    jacob_num =  kots.jacobian(state, numerical=True)
    print("norm: ", np.linalg.norm(jacob - jacob_num))

    state_list = StateType.create_list('link','end',["vel","acc","jerk","snap"])
    jacob = kots.jacobian(state_list)
    jacob_num = kots.jacobian(state_list, numerical=True)
    print("norm: ", np.linalg.norm(jacob - jacob_num))

    kots.dynamics()

    st_list = StateType.create_list('link','end',["momentum_diff3", "force_diff2"])
    print("norm: ", np.linalg.norm( kots.jacobian(st_list) -  kots.jacobian(st_list, numerical=True) ))

    st_list = StateType.create_list('link','end',["momentum_diff3"], "world")
    print("norm: ", np.linalg.norm( kots.jacobian(st_list) -  kots.jacobian(st_list, numerical=True) ))

    st_list = StateType.create_list('joint','joint4',["momentum_diff3"], "world")
    print("norm: ", np.linalg.norm( kots.jacobian(st_list) -  kots.jacobian(st_list, numerical=True) ))
    
    st_list = StateType.create_list('joint','joint4',["momentum_diff3", "force_diff2", "torque_diff2"])
    print("norm: ", np.linalg.norm( kots.jacobian(st_list) -  kots.jacobian(st_list, numerical=True) ))

    kots.show_robot(save=True)

if __name__ == "__main__":
    main()
