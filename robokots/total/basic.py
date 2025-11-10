import numpy as np
from mathrobo import Factorial

def total_factorial_mat(num : int, order : int, dim : int = 6) -> np.ndarray:
    '''
    Create a block diagonal matrix where each block is the factorial matrix.
    Args:
        order (int): Order of the CMTM.
        dim (int, optional): Dimension of the space. Defaults to 6.
    '''
    n = dim * order
    mat = np.zeros((num * n, num * n))

    for i in range(num):
        start = i * n
        mat[start:start+n, start:start+n] = Factorial.mat(order, dim)
    return mat

def total_factorial_mat_inv(num : int, order : int, dim : int = 6) -> np.ndarray:
    '''
    Create a block diagonal matrix where each block is the factorial matrix.
    Args:
        order (int): Order of the CMTM.
        dim (int, optional): Dimension of the space. Defaults to 6.
    '''
    n = dim * order
    mat = np.zeros((num * n, num * n))

    for i in range(num):
        start = i * n
        mat[start:start+n, start:start+n] = Factorial.mat_inv(order, dim)
    return mat
