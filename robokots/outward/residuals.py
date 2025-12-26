from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .interface import Quantity

Array = np.ndarray

@dataclass
class VectorSquaredSumResidual:
    """
    Minimize ||v(x)||^2 by returning residual r=v and Jacobian J=dv/dx.
    """
    name: str
    quantity: "Quantity"   # value()->(m,), jacobian()->(m,N)

    def evaluate(self) -> Tuple[Array, Array]:
        r = self.quantity.value().reshape(-1)      # (m,)
        J = self.quantity.jacobian()               # (m,N)
        return r, J
