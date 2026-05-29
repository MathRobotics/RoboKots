from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from mathrobo import CMTM, CMVector, SE3, SE3wrench

from .state_dict import cmtm_to_state_list, vecs_to_state_dict

if TYPE_CHECKING:
    from .robot import RobotStruct


def _truncate_cmtm_order(cmtm: CMTM, order: int) -> CMTM:
    if order < 1:
        raise ValueError(f"Invalid order: {order}. Must be >= 1.")
    if order > cmtm._n:
        raise ValueError(f"Invalid order: {order}. Must be <= source order {cmtm._n}.")
    if order == cmtm._n:
        return cmtm
    return CMTM[SE3](SE3.set_mat(cmtm.elem_mat()), cmtm.vecs()[: order - 1])


@dataclass
class OutwardState:
    order: int
    link_cmtm: dict[str, CMTM]
    joint_cmtm: dict[str, CMTM]
    link_momentum: dict[str, CMVector] = field(default_factory=dict)
    joint_momentum: dict[str, CMVector] = field(default_factory=dict)
    link_force: dict[str, CMVector] = field(default_factory=dict)
    joint_force: dict[str, CMVector] = field(default_factory=dict)
    joint_torque: dict[str, np.ndarray] = field(default_factory=dict)
    _cache: dict[tuple, object] = field(default_factory=dict, init=False, repr=False)

    def cmtm(self, owner_type: str, owner_name: str, order: int | None = None) -> CMTM:
        source = self.link_cmtm if owner_type == "link" else self.joint_cmtm
        cmtm = source[owner_name]
        if order is None:
            order = cmtm._n

        cache_key = ("cmtm", owner_type, owner_name, order)
        if cache_key not in self._cache:
            self._cache[cache_key] = _truncate_cmtm_order(cmtm, order)
        return self._cache[cache_key]

    def cmtm_wrench(self, owner_type: str, owner_name: str, order: int | None = None) -> CMTM:
        cache_key = ("cmtm_wrench", owner_type, owner_name, order)
        if cache_key not in self._cache:
            self._cache[cache_key] = CMTM.change_elemclass(
                self.cmtm(owner_type, owner_name, order),
                SE3wrench,
            )
        return self._cache[cache_key]

    def rel_cmtm(
        self,
        base_name: str,
        target_name: str,
        owner_type: str = "link",
        order: int | None = None,
    ) -> CMTM:
        cache_key = ("rel_cmtm", owner_type, base_name, target_name, order)
        if cache_key not in self._cache:
            base = self.cmtm(owner_type, base_name, order)
            target = self.cmtm(owner_type, target_name, order)
            self._cache[cache_key] = base.inv() @ target
        return self._cache[cache_key]

    def rel_cmtm_wrench(
        self,
        base_name: str,
        target_name: str,
        owner_type: str = "link",
        order: int | None = None,
    ) -> CMTM:
        cache_key = ("rel_cmtm_wrench", owner_type, base_name, target_name, order)
        if cache_key not in self._cache:
            self._cache[cache_key] = CMTM.change_elemclass(
                self.rel_cmtm(base_name, target_name, owner_type, order),
                SE3wrench,
            )
        return self._cache[cache_key]

    def cmvec(self, owner_type: str, owner_name: str, data_type: str) -> CMVector:
        source_by_type = {
            ("link", "momentum"): self.link_momentum,
            ("joint", "momentum"): self.joint_momentum,
            ("link", "force"): self.link_force,
            ("joint", "force"): self.joint_force,
        }
        return source_by_type[(owner_type, data_type)][owner_name]

    def to_state_dict(self, robot: RobotStruct) -> dict:
        state_dict = {}

        for link in robot.links:
            cmtm = self.link_cmtm.get(link.name)
            if cmtm is not None:
                state_dict.update(cmtm_to_state_list(cmtm, "link", link.name))

        for joint in robot.joints:
            cmtm = self.joint_cmtm.get(joint.name)
            if cmtm is not None:
                state_dict.update(cmtm_to_state_list(cmtm, "joint", joint.name))

        for link in robot.links:
            momentum = self.link_momentum.get(link.name)
            if momentum is not None:
                state_dict.update(vecs_to_state_dict(momentum.vecs(), "link", link.name, "momentum", momentum._n))

            force = self.link_force.get(link.name)
            if force is not None:
                state_dict.update(vecs_to_state_dict(force.vecs(), "link", link.name, "force", force._n))

        for joint in robot.joints:
            momentum = self.joint_momentum.get(joint.name)
            if momentum is not None:
                state_dict.update(vecs_to_state_dict(momentum.vecs(), "joint", joint.name, "momentum", momentum._n))

            force = self.joint_force.get(joint.name)
            if force is not None:
                state_dict.update(vecs_to_state_dict(force.vecs(), "joint", joint.name, "force", force._n))

            torque = self.joint_torque.get(joint.name)
            if torque is not None:
                torque_arr = np.asarray(torque)
                torque_order = torque_arr.size // joint.dof if joint.dof > 0 else 0
                if torque_order > 0:
                    state_dict.update(vecs_to_state_dict(torque_arr, "joint", joint.name, "torque", torque_order))

        return state_dict
