"""Read computational state objects without materializing flat dictionaries."""

import numpy as np
from mathrobo import CMVector, CMTM, SE3

from .data import OutwardState


def state_sample(robot, state, index):
    """Select one computational batch sample, without dictionary serialization.

    Used by scalar-only analytic kernels; arrays may share storage with state.
    """
    def cmtms(owner_type, owners):
        result = {}
        for owner in owners:
            value = state.cmtm(owner_type, owner.name)
            result[owner.name] = CMTM[SE3](
                SE3.set_mat(np.asarray(value.elem_mat())[index]),
                np.asarray(value.vecs())[index],
            )
        return result

    sample = OutwardState(state.order, cmtms("link", robot.links),
                          cmtms("joint", robot.joints), gravity=np.array(state.gravity, copy=True))
    for owner_type, owners in (("link", robot.links), ("joint", robot.joints)):
        for owner in owners:
            families = ("momentum", "force", "torque") if owner_type == "joint" and owner.dof else ("momentum", "force")
            for family in families:
                try:
                    series = state.quantity_series(owner_type, owner.name, family)[index]
                except KeyError:
                    continue
                if series.shape[-2]:
                    getattr(sample, f"{owner_type}_{family}")[owner.name] = (
                        series if family == "torque" else CMVector(series))
    return sample


def state_cmtm(state, owner_name, owner_type="link", order=None):
    return state.cmtm(owner_type, owner_name, order)


def state_cmtm_wrench(state, owner_name, owner_type="link", order=None):
    return state.cmtm_wrench(owner_type, owner_name, order)


def state_frame(state, owner_name, owner_type="link"):
    return SE3.set_mat(state.cmtm(owner_type, owner_name, 1).elem_mat())


def state_rel_frame(state, base_name, target_name, owner_type="link"):
    return SE3.set_mat(state.rel_cmtm(base_name, target_name, owner_type, 1).elem_mat())


def state_rel_cmtm(state, base_name, target_name, owner_type="link", order=None):
    return state.rel_cmtm(base_name, target_name, owner_type, order)


def state_rel_cmtm_wrench(state, base_name, target_name, owner_type="link", order=None):
    return state.rel_cmtm_wrench(base_name, target_name, owner_type, order)


def state_cmvec(state, owner_name, owner_type, data_type, order):
    value = state.cmvec(owner_type, owner_name, data_type)
    if order < 1 or order > value._n:
        raise ValueError(f"Invalid order: requested {order}, source order is {value._n}.")
    if order == value._n:
        return value
    if hasattr(value, "truncate"):
        return value.truncate(order)
    return CMVector(value.vecs()[..., :order, :])


def total_link_cmvec(state, link_names, data_type, order):
    values = [state_cmvec(state, name, "link", data_type, order).cm_vec()
              for name in link_names]
    return np.concatenate(values, axis=-1) if values else np.zeros(0)


def state_link_positions(state, link_names):
    return np.stack([np.asarray(state.cmtm("link", name, 1).elem_mat())[..., :3, 3]
                     for name in link_names], axis=-2)
