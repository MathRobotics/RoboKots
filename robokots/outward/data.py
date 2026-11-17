from __future__ import annotations

from dataclasses import dataclass, field

from collections.abc import MutableMapping

import numpy as np
from mathrobo import CMTM, CMVector, SE3, SE3wrench




class CMSeriesMapping(MutableMapping):
    """Ordinary-array mapping interface backed exclusively by CMVector values."""
    def __init__(self, values=()):
        self._coefficients = {}
        for name, value in dict(values).items():
            self[name] = value

    def __getitem__(self, name):
        return self._coefficients[name].vecs()

    def __setitem__(self, name, value):
        self._coefficients[name] = value if isinstance(value, CMVector) else CMVector(np.asarray(value))

    def __delitem__(self, name):
        del self._coefficients[name]

    def __iter__(self):
        return iter(self._coefficients)

    def __len__(self):
        return len(self._coefficients)

    def cmvector(self, name):
        return self._coefficients[name]


def _truncate_cmtm_order(cmtm: CMTM, order: int) -> CMTM:
    if order < 1:
        raise ValueError(f"Invalid order: {order}. Must be >= 1.")
    if order > cmtm._n:
        raise ValueError(f"Invalid order: {order}. Must be <= source order {cmtm._n}.")
    if order == cmtm._n:
        return cmtm
    return CMTM[SE3](SE3.set_mat(cmtm.elem_mat()), cmtm.cmvecs().truncate(order - 1))


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

    def __post_init__(self):
        self.joint_torque = CMSeriesMapping(self.joint_torque)

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
        if data_type == "torque" and owner_type == "joint":
            return self.joint_torque.cmvector(owner_name)
        source_by_type = {
            ("link", "momentum"): self.link_momentum,
            ("joint", "momentum"): self.joint_momentum,
            ("link", "force"): self.link_force,
            ("joint", "force"): self.joint_force,
        }
        return source_by_type[(owner_type, data_type)][owner_name]

    def quantity_series(self, owner_type: str, owner_name: str, data_type: str) -> np.ndarray:
        """Read stored derivatives with shape (..., order, dimension).

        Missing quantities raise KeyError. Ordinary arrays are reconstructed on read.
        """
        if data_type == "torque":
            if owner_type != "joint":
                raise KeyError((owner_type, owner_name, data_type))
            return self.joint_torque[owner_name]
        return self.cmvec(owner_type, owner_name, data_type).vecs()


class ArrayOutwardState:
    """Array-backed state. Public array attributes reconstruct detached ordinary values."""
    def __init__(self, order, link_names, joint_names, joint_dofs, link_mat, link_vecs,
                 joint_mat, joint_vecs, link_momentum_array=None, link_force_array=None,
                 joint_momentum_array=None, joint_force_array=None, joint_torque_array=None,
                 *, cm_storage=False):
        self.order = order
        self.link_names, self.joint_names, self.joint_dofs = link_names, joint_names, joint_dofs
        self.link_mat, self.joint_mat = np.asarray(link_mat), np.asarray(joint_mat)
        self._series_cm = {}
        for name, value in {
            "link_vecs": link_vecs,
            "joint_vecs": joint_vecs,
            "link_momentum_array": link_momentum_array,
            "link_force_array": link_force_array,
            "joint_momentum_array": joint_momentum_array,
            "joint_force_array": joint_force_array,
            "joint_torque_array": joint_torque_array,
        }.items():
            if value is not None:
                values = np.asarray(value)
                self._series_cm[name] = CMVector.set_cmvecs(values) if cm_storage else CMVector(values)
        self._cache = {}
        self._link_index = {name: i for i, name in enumerate(link_names)}
        self._joint_index = {name: i for i, name in enumerate(joint_names)}

    @property
    def link_vecs(self):
        value = self._series_cm.get("link_vecs")
        return None if value is None else value.vecs()

    @property
    def joint_vecs(self):
        value = self._series_cm.get("joint_vecs")
        return None if value is None else value.vecs()

    @property
    def link_momentum_array(self):
        value = self._series_cm.get("link_momentum_array")
        return None if value is None else value.vecs()

    @property
    def link_force_array(self):
        value = self._series_cm.get("link_force_array")
        return None if value is None else value.vecs()

    @property
    def joint_momentum_array(self):
        value = self._series_cm.get("joint_momentum_array")
        return None if value is None else value.vecs()

    @property
    def joint_force_array(self):
        value = self._series_cm.get("joint_force_array")
        return None if value is None else value.vecs()

    @property
    def joint_torque_array(self):
        value = self._series_cm.get("joint_torque_array")
        return None if value is None else value.vecs()

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
    def joint_torque(self):
        return {name: self.cmvec("joint", name, "torque").vecs()
                for name, dof in zip(self.joint_names, self.joint_dofs)
                if dof > 0 and "joint_torque_array" in self._series_cm}

    def cmtm(self, owner_type: str, owner_name: str, order: int | None = None) -> CMTM:
        if order is None:
            order = self.order
        if order < 1:
            raise ValueError(f"Invalid order: {order}. Must be >= 1.")

        cache_key = ("cmtm", owner_type, owner_name, order)
        if cache_key not in self._cache:
            mat, vecs = self._cmtm_arrays(owner_type, owner_name)
            source_order = vecs._n + 1
            if order > source_order:
                raise ValueError(f"Invalid order: {order}. Must be <= source order {source_order}.")
            self._cache[cache_key] = CMTM[SE3](
                SE3.set_mat(mat),
                vecs.truncate(order - 1),
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

    def cmvec(self, owner_type, owner_name, data_type):
        cache_key = ("cmvec", owner_type, owner_name, data_type)
        if cache_key not in self._cache:
            name = f"{owner_type}_{data_type}_array"
            if name not in self._series_cm:
                raise KeyError((owner_type, owner_name, data_type))
            idx = self._index(owner_type, owner_name)
            coefficients = self._series_cm[name].cm_vecs()[..., idx, :, :]
            if data_type == "torque":
                coefficients = coefficients[..., :self.joint_dofs[idx]]
            self._cache[cache_key] = CMVector.set_cmvecs(coefficients)
        return self._cache[cache_key]

    def quantity_series(self, owner_type: str, owner_name: str, data_type: str) -> np.ndarray:
        """Read stored derivatives with shape (..., order, dimension).

        Missing quantities raise KeyError. Ordinary arrays are reconstructed on read.
        """
        if data_type == "torque":
            if owner_type != "joint" or "joint_torque_array" not in self._series_cm:
                raise KeyError((owner_type, owner_name, data_type))
            idx = self._index(owner_type, owner_name)
            return self.cmvec("joint", owner_name, "torque").vecs()
        return self.cmvec(owner_type, owner_name, data_type).vecs()

    def _cmtm_arrays(self, owner_type: str, owner_name: str) -> tuple[np.ndarray, np.ndarray]:
        idx = self._index(owner_type, owner_name)
        if owner_type == "link":
            return self.link_mat[..., idx, :, :], CMVector.set_cmvecs(self._series_cm["link_vecs"].cm_vecs()[..., idx, :, :])
        if owner_type == "joint":
            return self.joint_mat[..., idx, :, :], CMVector.set_cmvecs(self._series_cm["joint_vecs"].cm_vecs()[..., idx, :, :])
        raise KeyError(owner_type)

    def _index(self, owner_type: str, owner_name: str) -> int:
        if owner_type == "link":
            return self._link_index[owner_name]
        if owner_type == "joint":
            return self._joint_index[owner_name]
        raise KeyError(owner_type)

    def _cmvec_dict(self, owner_type, data_type):
        cache_key = ("cmvec_dict", owner_type, data_type)
        if cache_key not in self._cache:
            names = self.link_names if owner_type == "link" else self.joint_names
            self._cache[cache_key] = ({name: self.cmvec(owner_type, name, data_type) for name in names}
                if f"{owner_type}_{data_type}_array" in self._series_cm else {})
        return self._cache[cache_key]


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
                CMVector.set_cmvecs(np.asarray(value.cmvecs().cm_vecs())[index]),
            )
        return result

    sample = OutwardState(state.order, cmtms("link", robot.links),
                          cmtms("joint", robot.joints), gravity=np.array(state.gravity, copy=True))
    for owner_type, owners in (("link", robot.links), ("joint", robot.joints)):
        for owner in owners:
            families = ("momentum", "force", "torque") if owner_type == "joint" and owner.dof else ("momentum", "force")
            for family in families:
                try:
                    coefficients = state.cmvec(owner_type, owner.name, family).cm_vecs()[index]
                except KeyError:
                    continue
                if coefficients.shape[-2]:
                    getattr(sample, f"{owner_type}_{family}")[owner.name] = (
                        CMVector.set_cmvecs(coefficients))
    return sample
