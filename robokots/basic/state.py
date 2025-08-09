from mathrobo import SO3, SE3, CMTM

keys = ("pos", "rot", "vel", "acc", "acc_diff", \
                 "jerk", "snap", "crackle", "pop", \
                 "lock", "drop", "shot", "put")

keys_order = {
    "pos": 1,
    "rot": 1,
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

keys_name = {
    "pos" : "pos",
    "rot" : "rot",
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
  elif data_type == "jerk":
    return None
  elif data_type == "frame":
    return SE3.sub_tan_vec
  elif data_type == "cmtm":
    return CMTM.sub_vec
  else:
    raise ValueError(f"Invalid data_type: {data_type}. Must be 'pos', 'rot', 'vel', 'acc', 'frame' or 'cmtm'.")
