from typing import List
import numpy as np

from .basic.state import keys_order, keys_time_order, data_type_dof, dim_to_dof
from .basic.robot import RobotStruct

def check_valid_str_list(str_list : List[str]):
    if not str_list:
        raise ValueError("str_list is empty")
  
    if type(str_list) is str:
        str_list = [str_list]

    if not all(isinstance(s, str) for s in str_list):
        raise ValueError("str_list must contain only strings")

    return str_list

def check_valid_data_type_list(data_type_list : List[str]):
    if not data_type_list:
        raise ValueError("data_type_list is empty")
  
    if type(data_type_list) is str:
        data_type_list = [data_type_list]

    return data_type_list

def count_time_order(data_type_list : List[str]) -> int:
    max_order = 0

    for data_type in data_type_list:
        if data_type not in keys_time_order:
            raise ValueError(f"Invalid data_type: {data_type}. Must be one of {list(keys_time_order.keys())}.")
        order = keys_time_order[data_type]
        if order > max_order:
            max_order = order

    return max_order

def filter_cmtm_row_data_to_target(cmtm_row_data : np.array, name_list : List[str], data_type_list : List[str], dim = 3) -> np.array:
    idx = []
    base = dim_to_dof(dim)

    
    for data_type in data_type_list:
        if not data_type:
            continue
        order = keys_order[data_type]
        dof   = data_type_dof(data_type, order, dim)
        start = base * (order - 1)
        idx.extend(range(start, start + dof))

    if not idx:
        return cmtm_row_data[:0, :]

    return cmtm_row_data[np.asarray(idx, dtype=int), :]