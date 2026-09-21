# -*- coding: utf-8 -*-
"""State dictionary conversion, extraction, and derived-value helpers."""

import re
import numpy as np
from mathrobo import CMVector
from mathrobo import SE3, SE3wrench, CMTM, CMTM, SO3, Factorial

from ..core.state.spec import keys, keys_order, keys_time_order, state_dict_key
from ..core.state.batch import StateBatch
from ..core.state.protocol import OutwardDataView


_STATE_MEMO_KEY = "__robokots_state_memo__"


def export_state_dict(robot, state: OutwardDataView | StateBatch) -> dict:
    """Export available state values as an independent dictionary snapshot.

    State readers provide CMTMs and derivative series; key naming, flattening,
    and copying belong here, independently of the computational backend.
    """
    if isinstance(state, StateBatch):
        samples = [export_state_dict(robot, sample) for sample in state.outward_states]
        keys = samples[0].keys()
        if any(sample.keys() != keys for sample in samples[1:]):
            raise ValueError("Batch states must expose identical dictionary keys")
        return {key: np.stack([sample[key] for sample in samples]).reshape(
                    state.batch_shape + np.shape(samples[0][key])) for key in keys}
    result = {}
    owners = (("link", robot.links), ("joint", robot.joints))
    for owner_type, entries in owners:
        for owner in entries:
            try:
                cmtm = state.cmtm(owner_type, owner.name)
            except KeyError:
                continue
            result.update(cmtm_to_state_list(cmtm, owner_type, owner.name))
    for owner_type, entries in owners:
        for owner in entries:
            families = ["momentum", "force"]
            if owner_type == "joint" and owner.dof > 0:
                families.append("torque")
            for family in families:
                try:
                    series = state.quantity_series(owner_type, owner.name, family)
                except KeyError:
                    continue
                order = series.shape[-2]
                if order > 0:
                    result.update(vecs_to_state_dict(
                        series, owner_type, owner.name, family, order
                    ))
    return result


def is_state_payload_key(key: object) -> bool:
    return key != _STATE_MEMO_KEY


def extract_state_keys(state: dict, prefix: str = "") -> list:
    """
    Extracts the suffix part of keys from a dictionary that match known physical quantities.

    Args:
        state (dict): Dictionary with string keys.
        prefix (str): Common prefix to strip from keys (e.g., "link").

    Returns:
        list: Suffixes such as "pos", "rot", "vel", etc.
    """

    result = []

    for k in state.keys():
        # If prefix is given, remove it (e.g., "link_" from "link_pos")
        if prefix and k.startswith(prefix + "_"):
            suffix = k[len(prefix) + 1:]
        else:
            suffix = k

        # Check if the suffix matches any keyword
        if any(kw in suffix for kw in keys):
            result.append(suffix)

    return result

def count_dict_time_order(state: dict) -> int:
    """
    Determines the highest derivative order present in the state's keys.

    Recognized keywords and their corresponding derivative orders:
        - "pos", "rot"      → 1st order (position, orientation)
        - "vel"             → 2nd order (velocity)
        - "acc"             → 3rd order (acceleration)
        - "jerk"            → 4th order (jerk)
        - "snap"            → 5th order (snap)
        - "acc_diffN"       → (3 + N)th order

    Args:
        state (dict): Dictionary containing state variable keys.

    Returns:
        int: The maximum derivative order detected in the keys.
    """


    keys = extract_state_keys(state)
    max_order = 0

    for k in keys:
        # Handle special case like "acc_diff1", "acc_diff2", etc.
        match = re.search(r"acc_diff(\d+)", k)
        if match:
            diff_order = int(match.group(1))
            max_order = max(max_order, keys_time_order["acc"] + diff_order)
            continue  # Skip other checks if acc_diff matched

        # Match against known keywords
        for keyword, order in keys_time_order.items():
            if keyword in k:
                max_order = max(max_order, order)
                break  # Stop checking once a match is found

    return max_order

def cmtm_to_state_list(cmtm : CMTM, owner_type : str, owner_name : str) -> list:
  '''
  Convert CMTM to state data
  Args:
    cmtm (CMTM): CMTM object
    name (str): name of the link
  Returns:
    list: state data
  '''
  state = []

  order = cmtm._n

  mat = cmtm.elem_mat()
  pos = mat[..., :3, 3]
  rot_vec = mat[..., :3, :3].reshape(mat.shape[:-2] + (9,))
  state.append((state_dict_key(owner_type, owner_name, "pos") , np.array(pos, copy=True)))
  state.append((state_dict_key(owner_type, owner_name, "rot") , np.array(rot_vec, copy=True)))
  if order > 1:
    veloc = cmtm.elem_vecs(0)
    state.append((state_dict_key(owner_type, owner_name, "vel") , np.array(veloc, copy=True)))
  if order > 2:
    accel = cmtm.elem_vecs(1)
    state.append((state_dict_key(owner_type, owner_name, "acc") , np.array(accel, copy=True)))
  if order > 3:
    for i in range(order-keys_order["acc"]):
      vec = cmtm.elem_vecs(i+2)
      state.append((state_dict_key(owner_type, owner_name, "acc")+"_diff"+str(i+1) , np.array(vec, copy=True)))

  return state

def vecs_to_state_dict(vec : np.ndarray, owner_type : str, owner_name : str, data_type : str,  order : int) -> list:
    '''
    Convert vector data to state data
    Args:
        vecs (np.ndarray): vector data
        name (str): name of the link or joint
        type_name (str): type of the vector (e.g., "link", "joint")
        vec_dof (int): dimension of the vector
    Returns:
        list: state data
    '''
    state = []
    vec = np.asarray(vec)

    if order <= 0:
        raise ValueError("order must be greater than 0")
    if vec.ndim >= 2 and vec.shape[-2] == order:
        vecs = vec
        vec_dof = vec.shape[-1]
    else:
        vec_dof = vec.shape[-1] // order if vec.ndim > 1 else vec.size // order
        vecs = vec.reshape(vec.shape[:-1] + (order, vec_dof))

    alias_name = state_dict_key(owner_type, owner_name, data_type)

    if vec.size == 0 or vec_dof == 0:
        return [
            (
                f"{alias_name}" if i == 0 else f"{alias_name}_diff{i}", []
            )
            for i in range(order)
        ]

    state = [
        (
            f"{alias_name}" if i == 0 else f"{alias_name}_diff{i}",
            vecs[..., i, :].copy(),
        )
        for i in range(order)
    ]

    return state

def state_dict_to_link_pos(state : dict, name : str) -> np.ndarray:
    '''
    Convert state data to link position
    Args:
        state (dict): state data
        name (str): name of the link
    Returns:
        np.ndarray: position vector
    '''
    pos = np.array(state[state_dict_key("link", name, "pos")])

    return pos

def state_dict_to_links_pos(state : dict, link_names : list) -> np.ndarray:
    '''
    Convert state data to link positions
    Args:
        state (dict): state data
        link_names (list): list of link names
    Returns:
        np.ndarray: array of positions
    '''
    pos_list = []
    for name in link_names:
        pos = state_dict_to_link_pos(state, name)
        pos_list.append(pos)

    return np.array(pos_list)

def state_dict_to_rot(state : dict, owner_name : str, owner_type : str = "link") -> np.ndarray:
    '''
    Convert state data to rotation matrix
    Args:
        state (dict): state data
        name (str): name of the link
    Returns:
        np.ndarray: rotation matrix
    '''
    if hasattr(state, "cmtm"):
        mat = np.asarray(state.cmtm(owner_type, owner_name, 1).elem_mat())
        return mat[..., :3, :3]


    rot_vec = np.array(state[state_dict_key(owner_type, owner_name, "rot")])
    rot = rot_vec.reshape(rot_vec.shape[:-1] + (3,3))

    return rot

def state_dict_to_frame(state : dict, owner_name : str, owner_type : str = "link") -> SE3:
    '''
    Convert state data to SE3
    Args:
        state (dict): state data
        name (str): name of the link
    Returns:
        SE3: SE3 object
    '''
    if hasattr(state, "cmtm"):
        return SE3.set_mat(state.cmtm(owner_type, owner_name, 1).elem_mat())


    pos = np.array(state[state_dict_key(owner_type, owner_name, "pos")])
    rot = state_dict_to_rot(state, owner_name, owner_type)
    mat = SE3(rot, pos)

    return mat

def state_dict_to_frame_wrench(state : dict, owner_name : str, owner_type : str = "link") -> SE3wrench:
    '''
    Convert state data to SE3wrench
    Args:
        state (dict): state data
        name (str): name of the link
    Returns:
        SE3wrench: SE3wrench object
    '''
    if hasattr(state, "cmtm_wrench"):
        mat = state.cmtm_wrench(owner_type, owner_name, 1).elem_mat()
        return SE3wrench(mat[..., :3, :3], mat[..., :3, 3])


    pos = np.array(state[state_dict_key(owner_type, owner_name, "pos")])
    rot = state_dict_to_rot(state, owner_name, owner_type)
    mat = SE3wrench(rot, pos)

    return mat

def __state_dict_to_cmtm_vecs(state : dict, owner_name : str, owner_type : str = "link", order = None) -> np.ndarray:
    '''
    Convert state data to CMTM values
    Args:
        state (dict): state data
        name (str): name of the link
    Returns:
        tuple: (SE3, np.ndarray)
    '''
    if order is None:
        order = count_dict_time_order(state)

    if order < 1:
        raise ValueError("order must be over 1")

    if order > 1:
        first = np.asarray(state[state_dict_key(owner_type, owner_name, "vel")])
        vec = np.zeros(first.shape[:-1] + (order-1, 6), dtype=first.dtype)
        vec[..., 0, :] = first
    else:
        vec = np.zeros((0, 6))
    if order > 2:
        vec[..., 1, :] = np.asarray(state[state_dict_key(owner_type, owner_name, "acc")])
    if order > 3:
        for i in range(order-keys_order["acc"]):
            vec[..., i+2, :] = np.asarray(state[state_dict_key(owner_type, owner_name, "acc")+"_diff"+str(i+1)])

    return vec

def state_dict_to_cmtm(state : dict, owner_name : str, owner_type : str = "link", order = None) -> CMTM:
    '''
    Convert state data to CMTM
    Args:
        state (dict): state data
        name (str): name of the link
    Returns:
        CMTM: CMTM object
    '''
    if hasattr(state, "cmtm"):
        return state.cmtm(owner_type, owner_name, order)

    if order is None:
        order = count_dict_time_order(state)


    mat = state_dict_to_frame(state, owner_name, owner_type)
    vec = __state_dict_to_cmtm_vecs(state, owner_name, owner_type, order)

    cmtm = CMTM[SE3](mat, vec)

    return cmtm

def state_dict_to_cmtm_wrench(state : dict, owner_name : str, owner_type : str = "link", order = None) -> CMTM:
    '''
    Convert state data to CMTM
    Args:
        state (dict): state data
        name (str): name of the link
    Returns:
        CMTM: CMTM object
    '''
    if hasattr(state, "cmtm_wrench"):
        return state.cmtm_wrench(owner_type, owner_name, order)

    if order is None:
        order = count_dict_time_order(state)


    mat = state_dict_to_frame_wrench(state, owner_name, owner_type)
    vec = __state_dict_to_cmtm_vecs(state, owner_name, owner_type, order)

    cmtm = CMTM[SE3wrench](mat, vec)

    return cmtm

def state_dict_to_rel_frame(state : dict, base_name : str, target_name : str, owner_type : str = "link") -> SE3:
    '''
    Convert state data to relative SE3
    Args:
        state (dict): state data
        base_name (str): name of the base link
        target_name (str): name of the target link
    Returns:
        SE3: SE3 object
    '''
    if hasattr(state, "rel_cmtm"):
        return SE3.set_mat(state.rel_cmtm(base_name, target_name, owner_type, 1).elem_mat())


    base_frame = state_dict_to_frame(state, base_name, owner_type)
    target_frame = state_dict_to_frame(state, target_name, owner_type)
    rel_frame = base_frame.inv() @ target_frame

    return rel_frame

def state_dict_to_rel_cmtm(state : dict, base_name : str, target_name : str, owner_type : str = "link", order = None) -> CMTM:
    '''
    Convert state data to relative CMTM
    Args:
        state (dict): state data
        base_name (str): name of the base link
        target_name (str): name of the target link
    Returns:
        CMTM: CMTM object
    '''
    if hasattr(state, "rel_cmtm"):
        return state.rel_cmtm(base_name, target_name, owner_type, order)

    if order is None:
        order = count_dict_time_order(state)


    base_cmtm = state_dict_to_cmtm(state, base_name, owner_type, order)
    target_cmtm = state_dict_to_cmtm(state, target_name, owner_type, order)
    rel_cmtm = base_cmtm.inv() @ target_cmtm

    return rel_cmtm

def state_dict_to_rel_cmtm_wrench(state : dict, base_name : str, target_name : str, owner_type : str = "link", order = None) -> CMTM:
    '''
    Convert state data to relative CMTM
    Args:
        state (dict): state data
        base_name (str): name of the base link
        target_name (str): name of the target link
    Returns:
        CMTM: CMTM object
    '''
    if hasattr(state, "rel_cmtm_wrench"):
        return state.rel_cmtm_wrench(base_name, target_name, owner_type, order)

    if order is None:
        order = count_dict_time_order(state)


    base_cmtm = state_dict_to_cmtm(state, base_name, owner_type, order)
    target_cmtm = state_dict_to_cmtm(state, target_name, owner_type, order)
    rel_cmtm = CMTM.change_elemclass(base_cmtm.inv() @ target_cmtm, SE3wrench)
    return rel_cmtm

def _state_vec_base_key(owner_name: str, owner_type: str, data_type: str) -> str:
    return state_dict_key(owner_type, owner_name, data_type)


def _collect_state_vecs_fast(state: dict, owner_name: str, owner_type: str, data_type: str, order: int | None = None) -> list[np.ndarray] | None:
    base_key = _state_vec_base_key(owner_name, owner_type, data_type)
    if base_key not in state:
        return None

    if order is None:
        keys = [base_key]
        idx = 1
        while True:
            diff_key = f"{base_key}_diff{idx}"
            if diff_key not in state:
                break
            keys.append(diff_key)
            idx += 1
    else:
        keys = [base_key] + [f"{base_key}_diff{i}" for i in range(1, order)]
        if any(key not in state for key in keys):
            return None

    return [np.asarray(state[key]) for key in keys]


def _collect_state_vecs_scan(state: dict, owner_name: str, owner_type: str, data_type: str, order: int | None = None) -> list[np.ndarray]:
    base_key = _state_vec_base_key(owner_name, owner_type, data_type)
    diff_re = re.compile(rf"^{re.escape(base_key)}_diff\d+$")
    vecs = []

    for key, value in state.items():
        if key == base_key or diff_re.match(key):
            vecs.append(np.asarray(value))
            if order is not None and len(vecs) >= order:
                break

    return vecs


def state_dict_to_vecs(state : dict, owner_type : str, owner_name : str, data_type : str) -> np.ndarray:
    '''
    Convert state data to vector
    Args:
        state (dict): state data
        name (str): name of the link or joint
        type_name (str): type of the vector (e.g., "link", "joint")
    Returns:
        np.ndarray: vector
    '''
    if hasattr(state, "cmvec"):
        return state.cmvec(owner_type, owner_name, data_type).vec()

    vecs = _collect_state_vecs_fast(state, owner_name, owner_type, data_type)
    if vecs is None:
        vecs = _collect_state_vecs_scan(state, owner_name, owner_type, data_type)
    if len(vecs) == 0:
        raise ValueError(f"Invalid name: {owner_name}, type_name: {owner_type} or data_type: {data_type}.")

    return np.concatenate(vecs, axis=-1)

def state_dict_to_cmvec(state : dict, owner_name : str, owner_type : str, data_type : str, order : int) -> CMVector:
    '''
    Convert state data to vector
    Args:
        state (dict): state data
        name (str): name of the link or joint
        type_name (str): type of the vector (e.g., "link", "joint")
    Returns:
        CMVector: vector
    '''
    if hasattr(state, "cmvec"):
        cmvec = state.cmvec(owner_type, owner_name, data_type)
        if order > cmvec._n:
            raise ValueError(
                f"Invalid order: requested {order}, but source order is {cmvec._n} for "
                f"{owner_name}_{owner_type}_{data_type}."
            )
        if order == cmvec._n:
            return cmvec
        truncate = getattr(cmvec, "truncate", None)
        if truncate is not None:
            return truncate(order)
        return CMVector(cmvec.vecs()[..., :order, :])

    vecs = _collect_state_vecs_fast(state, owner_name, owner_type, data_type, order=order)
    if vecs is None:
        vecs = _collect_state_vecs_scan(state, owner_name, owner_type, data_type, order=order)
    if len(vecs) == 0:
        raise ValueError(f"Invalid name: {owner_name}, type_name: {owner_type} or data_type: {data_type}.")

    # Backward compatibility: some internal caches keep all orders in the base key
    # as one flattened vector (without *_diffN keys).
    if len(vecs) == 1 and order > 1:
        flat = np.asarray(vecs[0])
        if flat.shape[-1] % order != 0:
            raise ValueError(
                f"Invalid order: requested {order}, but packed vector size is {flat.shape[-1]} for "
                f"{owner_name}_{owner_type}_{data_type}."
            )
        return CMVector(flat.reshape(flat.shape[:-1] + (order, -1)))

    if len(vecs) != order:
        raise ValueError(
            f"Invalid order: requested {order}, but found {len(vecs)} values for "
            f"{owner_name}_{owner_type}_{data_type}."
        )

    return CMVector(np.stack(vecs, axis=-2))

def extract_dict_link_info(state : dict, data_type : str, link_name : str, frame = None, rel_frame = 'dummy'):
    if frame != None:
        if frame == 'world':
            order = keys_order[data_type]
            cmtm = state_dict_to_cmtm(state, link_name, "link", order)
            cmtm_wrench = CMTM.change_elemclass(cmtm, SE3wrench)

    if data_type == "rot":
      return state_dict_to_rot(state, link_name)
    elif data_type == "frame":
        return state_dict_to_frame(state, link_name)
    elif data_type == "cmtm":
        return state_dict_to_cmtm(state, link_name, "link")
    elif "momentum" in data_type:
        if frame == 'world':
            local_momentum = state_dict_to_cmvec(state, link_name, "link", "momentum", order).cm_vec()
            world_momentum = CMVector(( Factorial.mat(order, dim=6) @ cmtm_wrench.mat_adj() @ local_momentum).reshape(-1,6)).vecs()
            return world_momentum[-1]
        else:
            return np.array(state[state_dict_key("link", link_name, data_type)])
    elif "force" in data_type:
        if frame == 'world':
            local_force = state_dict_to_cmvec(state, link_name, "link", "force", order).cm_vec()
            world_force = CMVector((Factorial.mat(order, dim=6) @ cmtm_wrench.mat_adj() @ local_force).reshape(-1,6)).vecs()
            return world_force[-1]
        else:
            return np.array(state[state_dict_key("link", link_name, data_type)])
    else:
        return np.array(state[state_dict_key("link", link_name, data_type)])

def extract_dict_joint_info(state : dict, data_type : str, joint_name : str, frame = "dummy", rel_frame = 'dummy'):
    if frame != 'dummy':
        if frame == 'world':
            order = keys_order[data_type]
            cmtm = state_dict_to_cmtm(state, joint_name, "joint", order)
            cmtm_wrench = CMTM.change_elemclass(cmtm, SE3wrench)

    if data_type == "frame":
        return state_dict_to_frame(state, joint_name)
    elif data_type == "cmtm":
        return state_dict_to_cmtm(state, joint_name, "joint")
    elif "momentum" in data_type:
        if frame == 'world':
            local_momentum = state_dict_to_cmvec(state, joint_name, "joint", "momentum", order).cm_vec()
            world_momentum = CMVector((Factorial.mat(order, dim=6) @ cmtm_wrench.mat_adj() @ local_momentum).reshape(-1,6)).vecs()
            return world_momentum[-1]
        else:
            return np.array(state[state_dict_key("joint", joint_name, data_type)])
    elif "force" in data_type:
        if frame == 'world':
            local_force = state_dict_to_cmvec(state, joint_name, "joint", "force", order).cm_vec()
            world_force = CMVector((Factorial.mat(order, dim=6) @ cmtm_wrench.mat_adj() @ local_force).reshape(-1,6)).vecs()
            return world_force[-1]
        else:
            return np.array(state[state_dict_key("joint", joint_name, data_type)])
    elif "torque" in data_type:
        return np.array(state[state_dict_key("joint", joint_name, data_type)])
    else:
        return np.array(state[state_dict_key("joint", joint_name, data_type)])

def extract_dict_total_link(state : dict, link_name_list : str, data_type : str, order : int) -> CMVector:
    for i, link_name in enumerate(link_name_list):
        vec = extract_dict_link_info(state, link_name, data_type, order)
        if i == 0:
            total_vec = np.zeros((len(link_name_list), vec._len))
    return total_vec.flatten()

def extract_dict_total_link_cmvec(state : dict, link_name_list : str, data_type : str, order : int) -> CMVector:
    for i, link_name in enumerate(link_name_list):
        vec = state_dict_to_cmvec(state, link_name, "link", data_type, order)
        if i == 0:
            total_vec = np.zeros((len(link_name_list), vec._len))
        total_vec[i] = vec.cm_vec()
    return total_vec.flatten()

def extract_dict_total_link_info(data : dict, data_type : str, frame = "dummy", rel_frame = 'dummy'):
    return {k: v for k, v in data.items() if "link_" + data_type in k}

def extract_dict_total_joint_info(data : dict, data_type : str, frame = "dummy", rel_frame = 'dummy'):
    return {k: v for k, v in data.items() if "joint_" + data_type in k}

def extract_dict_info(data : dict, data_type : str, owner_type : str, owner_name : str, frame = "dummy", rel_frame = 'dummy'):
    if isinstance(data_type, list):
        return [extract_dict_info(data, dt, owner_type, owner_name, frame, rel_frame) for dt in data_type]
    else:
        if owner_type == "link":
            return extract_dict_link_info(data, data_type, owner_name, frame, rel_frame)
        elif owner_type == "joint":
            return extract_dict_joint_info(data, data_type, owner_name, frame, rel_frame)
        elif owner_type == "total_link":
            return extract_dict_total_link_info(data, data_type, frame, rel_frame)
        elif owner_type == "total_joint":
            return extract_dict_total_joint_info(data, data_type, frame, rel_frame)
        else:
            raise ValueError(f"Invalid owner_type: {set(owner_type)}")

def sub_state_dict_vec(data0 : dict, data1 : dict, type : str, name : str) -> dict:
    '''
    data0 : dict
        state data 0
    data1 : dict
        state data 1
    type : str
        pos
        rot
        vel
        acc
        frame
        cmtm
    name : str
        link name or joint name
    '''

    if type == "pos":
        return data1[type] - data0[type]
    elif type == "rot":
        return SO3.sub_tan_vec(data0[type], data1[type], "bframe")
    elif type == "vel":
        return data1[type] - data0[type]
    elif type == "acc":
        return data1[type] - data0[type]
    elif type == "frame":
        return SE3.sub_tan_vec(data0[type], data1[type], "bframe")
    elif type == "cmtm":
        return CMTM.sub_tan_vec(data0[type], data1[type], "bframe")
    else:
        raise ValueError(f"Invalid type: {set(type)}")

def print_state_dict(state : dict):
    for k in state.keys():
        print(f"{k}: {state[k]}")
