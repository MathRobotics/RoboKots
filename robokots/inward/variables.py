from dataclasses import dataclass
from typing import List
import numpy as np

from ..outward.interface import Variable

Array = np.ndarray

@dataclass
class VariablePack:
    vars: List["Variable"]

    def dim(self) -> int:
        return int(sum(v.x.size for v in self.vars))

    def get(self) -> Array:
        return np.concatenate([v.x.reshape(-1) for v in self.vars], axis=0)

    def set(self, x: Array) -> None:
        x = np.asarray(x).reshape(-1)
        off = 0
        for v in self.vars:
            n = v.x.size
            v.x = x[off:off+n].reshape(v.x.shape)
            off += n

    def apply_dx(self, dx: Array) -> None:
        self.set(self.get() + np.asarray(dx).reshape(-1))
