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

    build_state must accept:
      build_state(x_all, time=time_grid, required=required_keys) -> state object

    Builders return computational state objects. Flat dictionaries belong to
    the export layer and are not used for state lookup here.
    """

    build_state: Callable[..., Any]

    # Latest computational state, absent until the first successful update.
    state: Any = None

    _rev_last: int = -1
    _time_rev_last: int = -1

    # Optional memo for *derived* heavy queries (same key space is ideal)
    _memo: dict[StateKey, Any] = field(default_factory=dict)

    _req_sig_last: int = 0

    def invalidate(self) -> None:
        self._rev_last = -1
        self._time_rev_last = -1
        self._memo.clear()
        self.state = None

    def _required_sig(self, required: Optional[Iterable[StateKey]]) -> int:
        if required is None:
            return 0
        return hash(frozenset(required))

    def is_fresh(self, revision: int, time: Any = None, required: Optional[Iterable[StateKey]] = None) -> bool:
        """Return True when cached state matches the requested revision/signature."""
        time_rev = int(getattr(time, "revision", 0)) if time is not None else 0
        req_sig = self._required_sig(required)
        return (
            int(revision) == self._rev_last
            and time_rev == self._time_rev_last
            and req_sig == self._req_sig_last
        )

    def update_if_needed(self, pack: PackLike, time: Any = None, required: Optional[Iterable[StateKey]] = None) -> None:
        rev = int(getattr(pack, "revision", 0))
        time_rev = int(getattr(time, "revision", 0)) if time is not None else 0
        req_sig = self._required_sig(required)

        if rev == self._rev_last and time_rev == self._time_rev_last and req_sig == self._req_sig_last:
            return

        x_all = np.asarray(pack.get(), dtype=float).reshape(-1)

        st = self.build_state(x_all, time=time, required=required)

        self.state = st
        self._rev_last = rev
        self._time_rev_last = time_rev
        self._req_sig_last = req_sig
        self._memo.clear()

    def get(self, key: StateKey) -> Any:
        if key in self._memo:
            return self._memo[key]
        raise KeyError(f"StateCache: missing derived value: {key}")

    def set_memo(self, key: StateKey, value: Any) -> None:
        self._memo[key] = value
