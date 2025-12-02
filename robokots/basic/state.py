from mathrobo import SO3, SE3, CMTM

frame_names = ("world","local")

data_owner_types = ("joint","link","total_link","total_joint","total")

class StateType:
    owner_type : str
    owner_name : str
    data_type : str
    frame_name : str

    def __init__(self, owner_type : str = None, owner_name : str = None, data_type : str = None, frame_name : str = None):
        self.owner_type = owner_type
        self.owner_name = owner_name
        self.data_type = data_type

        self.frame_name = frame_name
        self.time_order = keys_time_order.get(data_type, 1)
        self.key_order = keys_order.get(data_type, 1)
        self.is_dynamics = is_in_keys_dynamics([data_type])
        self.alliance = f"{owner_name}_{owner_type}_{keys_name[data_type]}"

    def __repr__(self):
        return f"StateType(\n  owner type: {self.owner_type}\n  owner name: {self.owner_name}\n  data type: {self.data_type}\n  frame name: {self.frame_name}\n  time order: {self.time_order}\n  key order: {self.key_order}\n  is dynamics: {self.is_dynamics}\n  alliance: {self.alliance}\n)"

keys_kinematics = \
    ("pos", "rot", "frame", "vel", "acc", "jerk", "snap", "crackle", "pop", "lock", "drop", "shot", "put")

keys_momentum = \
    ("momentum","momentum_diff1", "momentum_diff2", "momentum_diff3", "momentum_diff4", "momentum_diff5", "momentum_diff6", "momentum_diff7", "momentum_diff8")

keys_force = \
    ("force","force_diff1", "force_diff2", "force_diff3", "force_diff4", "force_diff5", "force_diff6", "force_diff7")

keys_torque = \
    ("torque", "torque_diff1", "torque_diff2", "torque_diff3", "torque_diff4", "torque_diff5", "torque_diff6", "torque_diff7")

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

def is_in_keys_dynamics(keys_list):
    for k in keys_list:
        if k not in keys_momentum and k not in keys_force and k not in keys_torque:
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
    "force_diff3": 6,
    "momentum": 2,
    "momentum_diff1": 3,
    "momentum_diff2": 4,
    "momentum_diff3": 5,
    "torque": 3,
    "torque_diff1": 4,
    "torque_diff2": 5,
    "torque_diff3": 6,
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
    "force_diff4": 5,
}

keys_order_momentum = {
    "momentum": 1,
    "momentum_diff1": 2,
    "momentum_diff2": 3,
    "momentum_diff3": 4,
    "momentum_diff4": 5,
}

keys_order_torque = {
    "torque": 1,
    "torque_diff1": 2,
    "torque_diff2": 3,
    "torque_diff3": 4,
    "torque_diff4": 5,
}

keys_order = {**keys_order_kinematics, **keys_order_momentum, **keys_order_force, **keys_order_torque}

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
    "put" : "acc_diff8",
    "momentum" : "momentum",
    "momentum_diff1" : "momentum_diff1",
    "momentum_diff2" : "momentum_diff2",
    "momentum_diff3" : "momentum_diff3",
    "momentum_diff4" : "momentum_diff4",
    "momentum_diff5" : "momentum_diff5",
    "momentum_diff6" : "momentum_diff6",
    "momentum_diff7" : "momentum_diff7",
    "momentum_diff8" : "momentum_diff8",
    "force" : "force",
    "force_diff1" : "force_diff1",
    "force_diff2" : "force_diff2",
    "force_diff3" : "force_diff3",
    "force_diff4" : "force_diff4",
    "force_diff5" : "force_diff5",
    "force_diff6" : "force_diff6",
    "force_diff7" : "force_diff7",
    "torque" : "torque",
    "torque_diff1" : "torque_diff1",
    "torque_diff2" : "torque_diff2",
    "torque_diff3" : "torque_diff3",
    "torque_diff4" : "torque_diff4",
    "torque_diff5" : "torque_diff5",
    "torque_diff6" : "torque_diff6",
    "torque_diff7" : "torque_diff7",
}

def data_type_to_sub_func(data_type : str):
    if data_type == "rot":
        return SO3.sub_tan_vec
    elif data_type == "frame":
        return SE3.sub_tan_vec
    elif data_type == "cmtm":
        return CMTM.sub_vec
    elif data_type in ["pos", "vel", "acc"] \
        or data_type in ["jerk", "snap", "crackle", "pop", "lock", "drop", "shot", "put"]  \
        or data_type in ["force", "force_diff1", "force_diff2", "force_diff3"] \
        or data_type in ["momentum", "momentum_diff1", "momentum_diff2", "momentum_diff3"] \
        or data_type in ["torque", "torque_diff1", "torque_diff2", "torque_diff3"]:
        return None
    else:
        raise ValueError(f"Invalid data_type: {data_type}. Must be 'pos', 'rot', 'vel', 'acc', 'jerk', 'frame' or 'cmtm'.")

def data_type_dof(data_type : str, order = None, dim = 3):
    if data_type == "pos" or data_type == "rot":
        return dim
    elif data_type in ["vel", "acc"] \
        or data_type in ["jerk", "snap", "crackle", "pop", "lock", "drop", "shot", "put"]  \
        or data_type in ["force", "force_diff1", "force_diff2", "force_diff3"] \
        or data_type in ["momentum", "momentum_diff1", "momentum_diff2", "momentum_diff3"]:
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

def state_type_list_condition(state_type_list : list[StateType]) -> int:
    max_order = 1
    is_dynamics = False
    for st in state_type_list:
        order = keys_order.get(st.data_type, 1)
        if order > max_order:
            max_order = order
        if is_in_keys_dynamics([st.data_type]):
            is_dynamics = True
    return max_order, is_dynamics