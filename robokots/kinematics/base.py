import numpy as np

from mathrobo import SE3, CMTM

from dataclasses import dataclass

@dataclass
class JointData:
    origin: SE3 # origin frame
    select_mat: np.ndarray # selection matrix
    dof: int = 0 # degree of freedom
    select_indeces: np.ndarray = None # indeces of the selection matrix