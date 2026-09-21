"""Read and transform state values through the shared reader protocol.

These helpers use mathrobo value types, but do not construct backend state
containers or depend on outward/API implementations.
"""

from __future__ import annotations

import numpy as np
from mathrobo import CMVector, SE3

from .protocol import OutwardDataView


def state_cmtm(state: OutwardDataView, owner_name, owner_type="link", order=None):
    return state.cmtm(owner_type, owner_name, order)


def state_cmtm_wrench(state: OutwardDataView, owner_name, owner_type="link", order=None):
    return state.cmtm_wrench(owner_type, owner_name, order)


def state_frame(state: OutwardDataView, owner_name, owner_type="link"):
    return SE3.set_mat(state.cmtm(owner_type, owner_name, 1).elem_mat())


def state_rel_frame(state: OutwardDataView, base_name, target_name, owner_type="link"):
    return SE3.set_mat(state.rel_cmtm(base_name, target_name, owner_type, 1).elem_mat())


def state_rel_cmtm(state: OutwardDataView, base_name, target_name, owner_type="link", order=None):
    return state.rel_cmtm(base_name, target_name, owner_type, order)


def state_rel_cmtm_wrench(state: OutwardDataView, base_name, target_name, owner_type="link", order=None):
    return state.rel_cmtm_wrench(base_name, target_name, owner_type, order)


def state_cmvec(state: OutwardDataView, owner_name, owner_type, data_type, order):
    value = state.cmvec(owner_type, owner_name, data_type)
    if order < 1 or order > value._n:
        raise ValueError(f"Invalid order: requested {order}, source order is {value._n}.")
    if order == value._n:
        return value
    if hasattr(value, "truncate"):
        return value.truncate(order)
    return CMVector(value.vecs()[..., :order, :])


def total_link_cmvec(state: OutwardDataView, link_names, data_type, order):
    values = [state_cmvec(state, name, "link", data_type, order).cm_vec()
              for name in link_names]
    return np.concatenate(values, axis=-1) if values else np.zeros(0)


def state_link_positions(state: OutwardDataView, link_names):
    return np.stack([np.asarray(state.cmtm("link", name, 1).elem_mat())[..., :3, 3]
                     for name in link_names], axis=-2)
