from mathrobo import SO3, SE3, CMTM

keys_kinematics = \
    ("pos", "rot", "frame", "vel", "acc", "jerk", "snap", "crackle", "pop", "lock", "drop", "shot", "put")

keys_dynamics = \
    ("force", "torque")

keys = keys_kinematics + keys_dynamics

def is_in_keys_kinematics(keys_list):
    for k in keys_list:
        if k not in keys_kinematics:
            return False
    return True

def is_in_keys_dynamics(keys_list):
    for k in keys_list:
        if k not in keys_dynamics:
            return False
    return True

def is_in_keys(keys_list):
    for k in keys_list:
        if k not in keys:
            return False
    return True

def filter_keys_kinematics(keys_list):
    return [k for k in keys_list if k in keys_kinematics]

def filter_keys_dynamics(keys_list):
    return [k for k in keys_list if k in keys_dynamics]

keys_order_kinematics = {
    "pos": 1,
    "rot": 1,
    "frame": 1,
    "vel": 2,
    "acc": 3,
    "jerk": 4,
    "snap": 5,
    "crackle": 6,
    "pop": 7,
    "lock": 8,
    "drop": 9,
    "shot": 10,
    "put": 11
}

keys_dynamics_order = {
    "force": 1,
    "torque": 2
}

keys_order = {**keys_order_kinematics, **keys_dynamics_order}

keys_name = {
    "pos" : "pos",
    "rot" : "rot",
    "frame" : "frame",
    "vel" : "vel",
    "acc" : "acc",
    "jerk" : "acc_diff1",
    "snap" : "acc_diff2",
    "crackle" : "acc_diff3",
    "pop" : "acc_diff4",
    "lock" : "acc_diff5",
    "drop" : "acc_diff6",
    "shot" : "acc_diff7",
    "put" : "acc_diff8"
}

def data_type_to_sub_func(data_type : str):
    if data_type == "pos":
        return None
    elif data_type == "rot":
        return SO3.sub_tan_vec
    elif data_type == "vel":
        return None
    elif data_type == "acc":
        return None
    elif data_type in ["jerk", "snap", "crackle", "pop", "lock", "drop", "shot", "put"]:
        return None
    elif data_type == "frame":
        return SE3.sub_tan_vec
    elif data_type == "cmtm":
        return CMTM.sub_vec
    else:
        raise ValueError(f"Invalid data_type: {data_type}. Must be 'pos', 'rot', 'vel', 'acc', 'jerk', 'frame' or 'cmtm'.")

def data_type_dof(data_type : str, order = None, dim = 3):
    if data_type == "pos" or data_type == "rot":
        return dim
    elif data_type == "vel" or data_type == "acc" or data_type == "jerk"  \
        or data_type == "snap" or data_type == "crackle" or data_type == "pop" \
        or data_type == "lock" or data_type == "drop" or data_type == "shot" or data_type == "put" \
        or data_type == "force":
        return dim * 2
    elif data_type == "frame":
        return dim * 2
    elif data_type == "cmtm":
        if order is None:
            return dim * 2
        else:
            return dim * 2 * order
    elif data_type == "cmtm_so3":
        if order is None:
            return dim * 2
        else:
            return dim * order
    else:
        raise ValueError(f"Invalid data_type: {data_type}. Must be 'pos', 'rot', 'vel', 'acc', 'frame' or 'cmtm'.")

def dim_to_dof(dim : int):
    if dim == 1:
        return 2
    elif dim == 2:
        return 3
    elif dim == 3:
        return 6