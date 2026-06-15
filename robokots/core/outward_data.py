from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .state import StateType


@runtime_checkable
class OutwardDataView(Protocol):
    """Read-only outward-state interface shared by backend adapters.

    Backends should implement this as a view over their native storage. Mutable
    compute/update methods are intentionally not part of this protocol so a JAX
    implementation can stay immutable and pytree-friendly.
    """

    order: int

    def state_value(self, state_type: StateType) -> Any: ...

    def cmtm(self, owner_type: str, owner_name: str, order: int | None = None) -> Any: ...

    def cmtm_wrench(self, owner_type: str, owner_name: str, order: int | None = None) -> Any: ...

    def rel_cmtm(
        self,
        base_name: str,
        target_name: str,
        owner_type: str = "link",
        order: int | None = None,
    ) -> Any: ...

    def rel_cmtm_wrench(
        self,
        base_name: str,
        target_name: str,
        owner_type: str = "link",
        order: int | None = None,
    ) -> Any: ...

    def cmvec(self, owner_type: str, owner_name: str, data_type: str) -> Any: ...
