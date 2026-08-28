"""Fixed-base inward-dynamics methods for the public ``Kots`` facade."""
from __future__ import annotations


class InwardDynamicsMixin:
  def inverse_dynamics(
      self, q, v, a, gravity=(0.0, 0.0, -9.81), backend: str = "rust",
  ):
    """Compute RNEA joint torques, including world-frame gravity."""
    self._fast_backend(backend)
    q, v, a = self._fast_qva(q, v, a)
    gravity = self._validate_gravity(gravity)
    rust_robot = self._rust_inverse_dynamics_robot()
    if q.ndim == 1:
      return rust_robot.rnea(q, v, a, gravity)
    return rust_robot.rnea_batch(q, v, a, gravity)

  def forward_dynamics(
      self, q, v, tau, gravity=(0.0, 0.0, -9.81),
      external_wrenches=None, backend: str = "rust",
  ):
    """Solve fixed-base forward dynamics for generalized acceleration.

    ``rust`` uses ABA. ``reference``/``numpy`` construct ``M(q)`` with RNEA
    and remain a correctness oracle. This method does not mutate StateCache.
    """
    if external_wrenches is not None:
      raise NotImplementedError("external_wrenches are not implemented for forward dynamics yet")
    if backend not in ("reference", "numpy", "rust"):
      raise ValueError("Unsupported forward dynamics backend: use 'reference', 'numpy', or 'rust'.")
    q, v, tau = self._fast_qva(q, v, tau)
    gravity = self._validate_gravity(gravity)
    if backend == "rust":
      robot = self._rust_inverse_dynamics_robot()
      if q.ndim == 1:
        return robot.aba(q, v, tau, gravity)
      return robot.aba_batch(q, v, tau, gravity)
    from ..inward.dynamics import forward_dynamics_reference

    def rnea(q_value, v_value, a_value, gravity_value):
      return self.inverse_dynamics(q_value, v_value, a_value, gravity=gravity_value, backend="rust")

    return forward_dynamics_reference(q, v, tau, gravity, rnea)

  def create_inward_cache(self):
    """Create reusable fixed-base inward-dynamics cache for this robot."""
    from ..inward import InwardCache
    return InwardCache(self)
