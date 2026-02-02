from dataclasses import dataclass
import numpy as np

@dataclass
class TimeGrid:
    N: int
    dt: float
    revision: int = 0

    def t(self, k: int) -> float:
        return float(k) * float(self.dt)

    def ks(self) -> range:
        return range(self.N + 1)

    def update(self, *, N: int | None = None, dt: float | None = None) -> None:
        changed = False
        if N is not None and int(N) != int(self.N):
            self.N = int(N); changed = True
        if dt is not None and float(dt) != float(self.dt):
            self.dt = float(dt); changed = True
        if changed:
            self.revision += 1
