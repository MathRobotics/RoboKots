from __future__ import annotations
from typing import Protocol, Iterable, Optional, Any
import numpy as np
from robokots.core.state_cache import StateKey

Array = np.ndarray

class Backend(Protocol):
    def build_state(
        self,
        x_all: Array,
        *,
        time: Any = None,
        required: Optional[Iterable[StateKey]] = None,
    ) -> dict[StateKey, Any]:
        ...
