from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any, Optional
import numpy as np

Array = np.ndarray

from .state_dict import (
    count_dict_time_order,
)

@dataclass
class StateCache:
    """
    Cache for expensive robot state computations (state_dict).

    - build_state(x_all) must return a dict compatible with your utilities:
        e.g., keys like "{link}_link_pos", "{link}_link_rot", "{joint}_joint_torque", etc.

    - Update is triggered only when VariablePack.revision changes.
    """

    build_state: Callable[[Array], dict]  # x_all -> state_dict

    # latest cached state_dict
    state: dict = field(default_factory=dict)

    # revision bookkeeping
    _rev_last: int = -1

    # optional cached properties derived from state_dict
    _time_order: Optional[int] = None

    # optional memo for heavy derived queries:
    # (owner_type, owner_name, data_type, frame, rel_frame) -> np.ndarray/SE3/etc.
    _memo: dict[tuple, Any] = field(default_factory=dict)

    def invalidate(self) -> None:
        """Force recomputation on next update."""
        self._rev_last = -1
        self._time_order = None
        self._memo.clear()

    def update_if_needed(self, pack: Any) -> None:
        """
        Update cache only if pack.revision changed.
        pack is your VariablePack (must have .revision and .get()).
        """
        rev = int(getattr(pack, "revision", 0))
        if rev == self._rev_last:
            return

        x_all = np.asarray(pack.get(), dtype=float).reshape(-1)

        self.state = self.build_state(x_all)
        if not isinstance(self.state, dict):
            raise TypeError("StateCache.build_state must return a dict (state_dict).")

        self._rev_last = rev
        self._time_order = None
        self._memo.clear()

    # -----------------------------
    # Light helpers
    # -----------------------------
    def time_order(self) -> int:
        """Compute and cache max derivative order present in state_dict keys."""
        if self._time_order is None:
            self._time_order = int(count_dict_time_order(self.state))
        return self._time_order
