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
    return CMTM[SE3](SE3.set_mat(cmtm.elem_mat()), cmtm.vecs()[..., : order - 1, :])


@dataclass
class OutwardState:
    order: int
    link_cmtm: dict[str, CMTM]
    joint_cmtm: dict[str, CMTM]
    gravity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
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
                torque_order = torque_arr.shape[-2] if joint.dof > 0 and torque_arr.ndim >= 2 else torque_arr.size // joint.dof if joint.dof > 0 else 0
                if torque_order > 0:
                    state_dict.update(vecs_to_state_dict(torque_arr, "joint", joint.name, "torque", torque_order))

        return state_dict


@dataclass
class ArrayOutwardState:
    order: int
    link_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    joint_dofs: tuple[int, ...]
    link_mat: np.ndarray
    link_vecs: np.ndarray
    joint_mat: np.ndarray
    joint_vecs: np.ndarray
    link_momentum_array: np.ndarray | None = None
    link_force_array: np.ndarray | None = None
    joint_momentum_array: np.ndarray | None = None
    joint_force_array: np.ndarray | None = None
    joint_torque_array: np.ndarray | None = None
    _cache: dict[tuple, object] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.link_mat = np.asarray(self.link_mat)
        self.link_vecs = np.asarray(self.link_vecs)
        self.joint_mat = np.asarray(self.joint_mat)
        self.joint_vecs = np.asarray(self.joint_vecs)
        if self.link_momentum_array is not None:
            self.link_momentum_array = np.asarray(self.link_momentum_array)
        if self.link_force_array is not None:
            self.link_force_array = np.asarray(self.link_force_array)
        if self.joint_momentum_array is not None:
            self.joint_momentum_array = np.asarray(self.joint_momentum_array)
        if self.joint_force_array is not None:
            self.joint_force_array = np.asarray(self.joint_force_array)
        if self.joint_torque_array is not None:
            self.joint_torque_array = np.asarray(self.joint_torque_array)
        self._link_index = {name: i for i, name in enumerate(self.link_names)}
        self._joint_index = {name: i for i, name in enumerate(self.joint_names)}

    @property
    def link_cmtm(self) -> dict[str, CMTM]:
        return {name: self.cmtm("link", name) for name in self.link_names}

    @property
    def joint_cmtm(self) -> dict[str, CMTM]:
        return {name: self.cmtm("joint", name) for name in self.joint_names}

    @property
    def link_momentum(self) -> dict[str, CMVector]:
        return self._cmvec_dict("link", "momentum")

    @property
    def link_force(self) -> dict[str, CMVector]:
        return self._cmvec_dict("link", "force")

    @property
    def joint_momentum(self) -> dict[str, CMVector]:
        return self._cmvec_dict("joint", "momentum")

    @property
    def joint_force(self) -> dict[str, CMVector]:
        return self._cmvec_dict("joint", "force")

    @property
    def joint_torque(self) -> dict[str, np.ndarray]:
        cache_key = ("joint_torque_dict",)
        if cache_key not in self._cache:
            if self.joint_torque_array is None:
                self._cache[cache_key] = {}
            else:
                self._cache[cache_key] = {
                    name: self.joint_torque_array[..., i, :, :dof]
                    for i, (name, dof) in enumerate(zip(self.joint_names, self.joint_dofs))
                    if dof > 0
                }
        return self._cache[cache_key]

    def cmtm(self, owner_type: str, owner_name: str, order: int | None = None) -> CMTM:
        if order is None:
            order = self.order
        if order < 1:
            raise ValueError(f"Invalid order: {order}. Must be >= 1.")

        cache_key = ("cmtm", owner_type, owner_name, order)
        if cache_key not in self._cache:
            mat, vecs = self._cmtm_arrays(owner_type, owner_name)
            source_order = vecs.shape[-2] + 1
            if order > source_order:
                raise ValueError(f"Invalid order: {order}. Must be <= source order {source_order}.")
            self._cache[cache_key] = CMTM[SE3](
                SE3.set_mat(mat),
                vecs[..., : order - 1, :],
            )
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
            self._cache[cache_key] = (
                self.cmtm(owner_type, base_name, order).inv()
                @ self.cmtm(owner_type, target_name, order)
            )
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
        cache_key = ("cmvec", owner_type, owner_name, data_type)
        if cache_key not in self._cache:
            arrays = {
                ("link", "momentum"): self.link_momentum_array,
                ("link", "force"): self.link_force_array,
                ("joint", "momentum"): self.joint_momentum_array,
                ("joint", "force"): self.joint_force_array,
            }
            array = arrays[(owner_type, data_type)]
            if array is None:
                raise KeyError((owner_type, owner_name, data_type))
            idx = self._index(owner_type, owner_name)
            self._cache[cache_key] = CMVector(array[..., idx, :, :])
        return self._cache[cache_key]

    def to_state_dict(self, robot: RobotStruct) -> dict:
        state_dict = {}

        for link in robot.links:
            state_dict.update(cmtm_to_state_list(self.cmtm("link", link.name), "link", link.name))

        for joint in robot.joints:
            state_dict.update(cmtm_to_state_list(self.cmtm("joint", joint.name), "joint", joint.name))

        for link in robot.links:
            if self.link_momentum_array is not None:
                momentum = self.cmvec("link", link.name, "momentum")
                state_dict.update(vecs_to_state_dict(momentum.vecs(), "link", link.name, "momentum", momentum._n))
            if self.link_force_array is not None:
                force = self.cmvec("link", link.name, "force")
                state_dict.update(vecs_to_state_dict(force.vecs(), "link", link.name, "force", force._n))

        for joint in robot.joints:
            if self.joint_momentum_array is not None:
                momentum = self.cmvec("joint", joint.name, "momentum")
                state_dict.update(vecs_to_state_dict(momentum.vecs(), "joint", joint.name, "momentum", momentum._n))
            if self.joint_force_array is not None:
                force = self.cmvec("joint", joint.name, "force")
                state_dict.update(vecs_to_state_dict(force.vecs(), "joint", joint.name, "force", force._n))

            torque = self.joint_torque.get(joint.name)
            if torque is not None:
                torque_arr = np.asarray(torque)
                torque_order = torque_arr.shape[-2] if joint.dof > 0 and torque_arr.ndim >= 2 else 0
                if torque_order > 0:
                    state_dict.update(vecs_to_state_dict(torque_arr, "joint", joint.name, "torque", torque_order))

        return state_dict

    def _cmtm_arrays(self, owner_type: str, owner_name: str) -> tuple[np.ndarray, np.ndarray]:
        idx = self._index(owner_type, owner_name)
        if owner_type == "link":
            return self.link_mat[..., idx, :, :], self.link_vecs[..., idx, :, :]
        if owner_type == "joint":
            return self.joint_mat[..., idx, :, :], self.joint_vecs[..., idx, :, :]
        raise KeyError(owner_type)

    def _index(self, owner_type: str, owner_name: str) -> int:
        if owner_type == "link":
            return self._link_index[owner_name]
        if owner_type == "joint":
            return self._joint_index[owner_name]
        raise KeyError(owner_type)

    def _cmvec_dict(self, owner_type: str, data_type: str) -> dict[str, CMVector]:
        cache_key = ("cmvec_dict", owner_type, data_type)
        if cache_key not in self._cache:
            arrays = {
                ("link", "momentum"): self.link_momentum_array,
                ("link", "force"): self.link_force_array,
                ("joint", "momentum"): self.joint_momentum_array,
                ("joint", "force"): self.joint_force_array,
            }
            if arrays[(owner_type, data_type)] is None:
                self._cache[cache_key] = {}
                return self._cache[cache_key]
            names = self.link_names if owner_type == "link" else self.joint_names
            self._cache[cache_key] = {
                name: self.cmvec(owner_type, name, data_type)
                for name in names
            }
        return self._cache[cache_key]
