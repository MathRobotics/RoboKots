import numpy as np

from mathrobo import SE3, CMTM

from dataclasses import dataclass

@dataclass
class JointData:
    origin: SE3 # origin frame
    select_mat: np.ndarray # selection matrix
    dof: int = 0 # degree of freedom
    select_indeces: np.ndarray = None # indeces of the selection matrix

@dataclass
class SoftLinkData:
    origin_coord: np.ndarray
    select_mat: np.ndarray # selection matrix
    length: float = 0.0 # length of the soft link
    dof: int = 0 # degree of freedom
    select_indeces: np.ndarray = None # indeces of the selection matrix