from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Iterable
import numpy as np

Array = np.ndarray


class PackLike(Protocol):
    revision: int
    def get(self) -> Array: ...


@dataclass(frozen=True)
class OwnerKey:
    owner_type: str
    owner_name: str


@dataclass(frozen=True)
class StateKey:
    k: int
    owner: OwnerKey
    dtype: str
    field: str
    frame: Optional[str] = None
    rel_frame: Optional[str] = None


@dataclass
class StateCache:
    """
    Cache for expensive state computations.

    build_state should ideally accept:
      build_state(x_all, time=time_grid, required=required_keys) -> dict[StateKey, Any]

    Backward compatible:
      build_state(x_all) -> dict
      build_state(x_all, time=time_grid) -> dict
    """

    build_state: Callable[..., dict]

    # Latest cached state mapping
    state: dict[StateKey, Any] = field(default_factory=dict)

    _rev_last: int = -1
    _time_rev_last: int = -1

    # Optional memo for *derived* heavy queries (same key space is ideal)
    _memo: dict[StateKey, Any] = field(default_factory=dict)

    _req_sig_last: int = 0

    def invalidate(self) -> None:
        self._rev_last = -1
        self._time_rev_last = -1
        self._memo.clear()
        self.state.clear()

    def _required_sig(self, required: Optional[Iterable[StateKey]]) -> int:
        if required is None:
            return 0
        return hash(frozenset(required))

    def update_if_needed(self, pack: PackLike, time: Any = None, required: Optional[Iterable[StateKey]] = None) -> None:
        rev = int(getattr(pack, "revision", 0))
        time_rev = int(getattr(time, "revision", 0)) if time is not None else 0
        req_sig = self._required_sig(required)

        if rev == self._rev_last and time_rev == self._time_rev_last and req_sig == self._req_sig_last:
            return

        x_all = np.asarray(pack.get(), dtype=float).reshape(-1)

        try:
            st = self.build_state(x_all, time=time, required=required)
        except TypeError:
            try:
                st = self.build_state(x_all, time=time)
            except TypeError:
                st = self.build_state(x_all)

        if not isinstance(st, dict):
            raise TypeError("StateCache.build_state must return a dict.")

        self.state = st  # Replace the state map atomically for safety.
        self._rev_last = rev
        self._time_rev_last = time_rev
        self._req_sig_last = req_sig
        self._memo.clear()

    def get(self, key: StateKey) -> Any:
        if key in self._memo:
            return self._memo[key]
        if key not in self.state:
            raise KeyError(f"StateCache: missing key: {key}")
        return self.state[key]

    def set_memo(self, key: StateKey, value: Any) -> None:
        self._memo[key] = value
