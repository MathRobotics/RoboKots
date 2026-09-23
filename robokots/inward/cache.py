"""Reusable cache for fixed-base inward dynamics.

This is intentionally separate from :class:`StateCache`.  It owns numerical
workspace and the input dependency stamps of an inward solver; it does not
publish ABA scratch values as semantic robot state.  The public phases are
stable even if the internal mass-solve factor changes from ABA to another
implementation later.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..kots import Kots


class InwardCache:
    """Reusable fixed-base inward-dynamics workspace for one robot.

    ``prepare`` fixes ``q``, ``v``, and gravity.  ``forward_dynamics`` can
    then be called repeatedly with different efforts.  Preparation stores an
    ABA factorization and bias acceleration for Rust, or a NumPy mass matrix
    and bias effort. NumPy solves the cached matrix for each new effort.
    """

    def __init__(self, kots: "Kots", *, backend=None) -> None:
        self._kots = kots
        self._backend = kots._resolve_dynamics_backend(backend)
        self._aba_data = kots._create_rust_aba_data() if self._backend == "rust" else None
        self._mass = self._bias = None
        self._q: np.ndarray | None = None
        self._v: np.ndarray | None = None
        self._gravity: np.ndarray | None = None
        self._tau: np.ndarray | None = None
        self._qdd: np.ndarray | None = None

    @property
    def is_prepared(self) -> bool:
        return self._q is not None

    def invalidate(self) -> None:
        """Forget prepared inputs and the memoized final solution."""
        self._q = self._v = self._gravity = self._tau = self._qdd = None
        self._mass = self._bias = None

    def prepare(self, q, v, gravity=(0.0, 0.0, -9.81)) -> "InwardCache":
        q, v, _ = self._kots._fast_qva(q, v, q)
        if q.ndim != 1:
            raise ValueError("InwardCache currently supports a single state; use Kots.forward_dynamics for batches.")
        gravity = self._kots._validate_gravity(gravity)
        if (
            self._q is None
            or not np.array_equal(q, self._q)
            or not np.array_equal(v, self._v)
            or not np.array_equal(gravity, self._gravity)
        ):
            # Publish the new cache keys only after successful preparation.
            self.invalidate()
            if self._aba_data is not None:
                self._aba_data.prepare(q, v, gravity)
            else:
                from .dynamics.forward_dynamics import mass_and_bias_reference
                self._mass, self._bias = mass_and_bias_reference(
                    q, v, gravity,
                    lambda q, v, a, g: self._kots.inverse_dynamics(q, v, a, gravity=g, backend="numpy"),
                )
            self._q = q.copy()
            self._v = v.copy()
            self._gravity = gravity.copy()
            self._tau = self._qdd = None
        return self

    def forward_dynamics(self, tau) -> np.ndarray:
        """Solve for one effort vector after :meth:`prepare`."""
        if not self.is_prepared:
            raise RuntimeError("call prepare(q, v, gravity) before forward_dynamics(tau)")
        tau = np.ascontiguousarray(np.asarray(tau, dtype=float))
        if tau.shape != self._q.shape:
            raise ValueError(f"tau shape must match prepared q shape {self._q.shape}.")
        if self._tau is not None and np.array_equal(tau, self._tau):
            return self._qdd.copy()
        if not np.all(np.isfinite(tau)):
            raise ValueError("tau must contain only finite values")
        qdd = (np.asarray(self._aba_data.solve(tau)) if self._aba_data is not None
               else np.linalg.solve(self._mass, tau - self._bias))
        self._tau = tau.copy()
        self._qdd = qdd.copy()
        return qdd

    def forward_dynamics_many(self, tau) -> np.ndarray:
        """Solve several effort vectors for the prepared ``q/v/gravity``."""
        if not self.is_prepared:
            raise RuntimeError("call prepare(q, v, gravity) before forward_dynamics_many(tau)")
        tau = np.ascontiguousarray(np.asarray(tau, dtype=float))
        if tau.ndim != 2 or tau.shape[1] != self._q.shape[0]:
            raise ValueError(f"tau shape must be (rhs, {self._q.shape[0]}).")
        if not np.all(np.isfinite(tau)):
            raise ValueError("tau must contain only finite values")
        if self._aba_data is None:
            return np.linalg.solve(self._mass, (tau - self._bias).T).T
        return np.asarray(self._aba_data.compute_many(self._q, self._v, tau, self._gravity))
