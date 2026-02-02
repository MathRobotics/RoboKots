from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any, Optional, Protocol
import numpy as np

Array = np.ndarray

from .state_dict import count_dict_time_order


class PackLike(Protocol):
    revision: int
    def get(self) -> Array: ...


@dataclass
class StateCache:
    """
    Cache for expensive robot state computations (state_dict).

    Update is triggered when:
      - pack.revision changes, OR
      - (optional) time.revision changes (for trajectory/time-grid problems).
    """

    # Backward compatible:
    # - allow build_state(x_all) OR build_state(x_all, time=time)
    build_state: Callable[..., dict]

    state: dict = field(default_factory=dict)

    _rev_last: int = -1
    _time_rev_last: int = -1 

    _time_order: Optional[int] = None
    _memo: dict[tuple, Any] = field(default_factory=dict)

    def invalidate(self) -> None:
        self._rev_last = -1
        self._time_rev_last = -1
        self._time_order = None
        self._memo.clear()

    def update_if_needed(self, pack: PackLike, time: Any = None) -> None:
        """
        Update cache only if pack.revision (or time.revision) changed.

        - pack must have .revision and .get()
        - time is optional (e.g., TimeGrid). If provided, and has .revision,
          we also use it as part of the cache key.
        """
        rev = int(getattr(pack, "revision", 0))
        time_rev = int(getattr(time, "revision", 0)) if time is not None else 0

        if rev == self._rev_last and time_rev == self._time_rev_last:
            return

        x_all = np.asarray(pack.get(), dtype=float).reshape(-1)

        # backward compatible call
        try:
            st = self.build_state(x_all, time=time)
        except TypeError:
            st = self.build_state(x_all)

        if not isinstance(st, dict):
            raise TypeError("StateCache.build_state must return a dict (state_dict).")

        self.state = st
        self._rev_last = rev
        self._time_rev_last = time_rev
        self._time_order = None
        self._memo.clear()

    def time_order(self) -> int:
        if self._time_order is None:
            self._time_order = int(count_dict_time_order(self.state))
        return self._time_order
