# outward/linearized.py (or inward/)
from dataclasses import dataclass
import numpy as np

Array = np.ndarray

@dataclass
class Linearized:
    r: Array  # (M,)
    J: Array  # (M,N)

    def cost(self) -> float:
        return float(self.r @ self.r)

    def normal_equations(self, damping: float = 0.0):
        # A dx = b  (A is NxN, b is N)
        A = self.J.T @ self.J
        if damping > 0:
            A = A + damping * np.eye(A.shape[0])
        b = -(self.J.T @ self.r)
        return A, b
