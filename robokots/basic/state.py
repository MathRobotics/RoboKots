from mathrobo import SO3, SE3, CMTM

keys_kinematics = \
    ("pos", "rot", "frame", "vel", "acc", "jerk", "snap", "crackle", "pop", "lock", "drop", "shot", "put")

keys_momentum = \
    ("momentum","momentum_diff1", "momentum_diff2", "momentum_diff3")

keys_force = \
    ("force","force_diff1", "force_diff2", "force_diff3")

keys_torque = \
    ("torque",)

keys = keys_kinematics + keys_momentum + keys_force + keys_torque

def is_in_keys_kinematics(keys_list):
    for k in keys_list:
        if k not in keys_kinematics:
            return False
    return True

def is_in_keys_momentum(keys_list):
    for k in keys_list:
        if k not in keys_momentum:
            return False
    return True

def is_in_keys_force(keys_list):
    for k in keys_list:
        if k not in keys_force:
            return False
    return True

def is_in_keys_torque(keys_list):
    for k in keys_list:
        if k not in keys_torque:
            return False
    return True

def is_in_keys(keys_list):
    for k in keys_list:
        if k not in keys:
            return False
    return True

def filter_keys_kinematics(keys_list):
    return [k for k in keys_list if k in keys_kinematics]

def filter_keys_momentum(keys_list):
    return [k for k in keys_list if k in keys_momentum]

def filter_keys_force(keys_list):
    return [k for k in keys_list if k in keys_force]

def filter_keys_torque(keys_list):
    return [k for k in keys_list if k in keys_torque]

keys_time_order = {
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
    "put": 11,
    "force": 3,
    "force_diff1": 4,
    "force_diff2": 5,
    "torque": 3,
    "momentum": 2,
}

keys_order_kinematics = {
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
    "put": 11,
}

keys_order_force = {
    "force": 1,
    "force_diff1": 2,
    "force_diff2": 3,
    "force_diff3": 4,
}

keys_order_momentum = {
    "momentum": 2,
    "momentum_diff1": 3,
    "momentum_diff2": 4,
    "momentum_diff3": 5,
}


keys_order = {**keys_order_kinematics, **keys_order_momentum, **keys_order_force}

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
        or data_type == "force" or data_type == "force_diff1" or data_type == "force_diff2" or data_type == "force_diff3" \
        or data_type == "momentum" or data_type == "momentum_diff1" or data_type == "momentum_diff2" or data_type == "momentum_diff3":
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