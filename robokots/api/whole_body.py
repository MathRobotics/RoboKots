"""Whole-body physical quantities derived from the current motion."""
from __future__ import annotations

import numpy as np

from .. import outward as outward_api
from ..core.motion import RobotMotions
from ..core import batch_shape as batch_shapes
from ..core.state.spec import StateType


class WholeBodyMixin:
  """Center-of-mass and whole-body angular-momentum convenience APIs.

  These quantities are evaluated on demand.  They are deliberately not stored
  as regular outward states because they describe the complete robot, rather
  than a link or joint.
  """

  def _total_mass(self) -> float:
    mass = float(sum(link.mass for link in self.robot_.links))
    if not np.isfinite(mass) or mass <= 0.0:
      raise ValueError("center of mass requires a finite, positive total link mass")
    return mass

  def _center_of_mass_from_motion(self, motion: np.ndarray) -> np.ndarray:
    state = outward_api.build_kinematics_outward_state(self.robot_, motion, order=1)
    total_mass = self._total_mass()
    result = None
    for link in self.robot_.links:
      if link.mass == 0.0:
        continue
      mat = np.asarray(state.link_cmtm[link.name].elem_mat())
      offset = (mat[..., :3, :3] @ np.asarray(link.cog)[..., None])[..., 0]
      point = mat[..., :3, 3] + offset
      term = link.mass * point
      result = term if result is None else result + term
    if result is None:
      # _total_mass() already rejects this, but keep the invariant explicit.
      raise ValueError("center of mass requires at least one positive-mass link")
    return result / total_mass

  def center_of_mass(self) -> np.ndarray:
    """Return the whole-body center of mass in world coordinates."""
    return self._center_of_mass_from_motion(self.motion(order=1))

  def _project_motion_order(self, motion: np.ndarray, input_order: int, output_order: int) -> np.ndarray:
    if input_order < output_order:
      raise ValueError("input motion order must be at least output motion order")
    motion = np.asarray(motion)
    projected = np.zeros(motion.shape[:-1] + (self.robot_.dof * output_order,), dtype=motion.dtype)
    for owner in self.robot_.motion_owners():
      src = RobotMotions.owner_vec_index(owner.dof, owner.dof_index, input_order, output_order)
      dst = RobotMotions.owner_vec_index(owner.dof, owner.dof_index, output_order)
      projected[..., dst] = motion[..., src]
    return projected

  def _numerical_motion_jacobian(self, func, order: int, eps: float) -> np.ndarray:
    self._ensure_not_batched(func.__name__)
    if eps <= 0.0 or not np.isfinite(eps):
      raise ValueError("eps must be finite and positive")
    motion = np.asarray(self.motion(order=order), dtype=float).copy()
    value = np.asarray(func(motion), dtype=float)
    jacobian = np.empty(value.shape + (self.robot_.dof * order,), dtype=float)
    for dof_index in range(self.robot_.dof):
      for derivative in range(order):
        # owner-major layout places each derivative block immediately after the
        # coordinate block for its owner, not after all robot coordinates.
        owner = next(owner for owner in self.robot_.motion_owners()
                     if owner.dof_index <= dof_index < owner.dof_index + owner.dof)
        local = dof_index - owner.dof_index
        owner_indices = RobotMotions.owner_vec_index(owner.dof, owner.dof_index, order)
        index = owner_indices.start + derivative * owner.dof + local
        plus = motion.copy()
        minus = motion.copy()
        plus[index] += eps; minus[index] -= eps
        column = (np.asarray(func(plus)) - np.asarray(func(minus))) / (2.0 * eps)
        jacobian[..., index] = column
    return jacobian

  def center_of_mass_jacobian(self, eps: float = 1e-7) -> np.ndarray:
    """Return ``d center_of_mass / dq`` with shape ``(3, dof)``.

    The center of mass depends only on coordinates, so velocity and higher
    motion derivatives are not included in the result.
    """
    return self._numerical_motion_jacobian(self._center_of_mass_from_motion, order=1, eps=eps)

  def _angular_momentum_from_motion(self, motion: np.ndarray, about: str) -> np.ndarray:
    if about not in ("world", "com"):
      raise ValueError("about must be 'world' or 'com'")
    state = outward_api.build_dynamics_outward_state(self.robot_, motion, dynamics_order=0)
    total = None
    for link in self.robot_.links:
      momentum = np.asarray(outward_api.get_value(
        self.robot_, state, StateType("link", link.name, "momentum", "world")
      ))
      total = momentum if total is None else total + momentum
    if total is None:
      return np.zeros(3)
    angular = total[..., :3]
    if about == "world":
      return angular
    coords = self._project_motion_order(motion, input_order=2, output_order=1)
    return angular - np.cross(self._center_of_mass_from_motion(coords), total[..., 3:])

  def angular_momentum(self, about: str = "world") -> np.ndarray:
    """Return whole-body angular momentum about the world origin or CoM."""
    return self._angular_momentum_from_motion(self.motion(order=2), about)

  def angular_momentum_jacobian(self, about: str = "world", eps: float = 1e-7) -> np.ndarray:
    """Return angular-momentum Jacobian wrt ``(q, qdot)``.

    The column layout is RoboKots' normal owner-major motion layout and the
    result has shape ``(3, 2 * dof)``.
    """
    return self._numerical_motion_jacobian(
      lambda motion: self._angular_momentum_from_motion(motion, about), order=2, eps=eps
    )

  def kinetic_energy_state(self, *, backend=None):
    """Return total kinetic energy from the current joint coordinates and velocities.

    Energy depends only on ``motion(2)`` (`q`, `qdot`).  The non-batched
    result is a Python float; batched motions return an array with the motion
    batch shape.
    """
    if self._resolve_dynamics_backend(backend) == "numpy":
      return self._numpy_kinetic_energy()
    motion = np.asarray(self.motion(2), dtype=float)
    batch_shape = motion.shape[:-1] if batch_shapes.is_batched_feature_array(motion) else ()
    robot = self._rust_compiled_robot()
    if not batch_shape:
      return float(np.asarray(robot.kinetic_energy(np.ascontiguousarray(motion)))[0])
    flat_motion = np.ascontiguousarray(motion.reshape((-1, motion.shape[-1])))
    return np.asarray(robot.kinetic_energy_batch(flat_motion)).reshape(batch_shape)

  def kinetic_energy_jacobian_mul(self, rhs : np.ndarray, *, backend=None):
    """Apply the kinetic-energy Jacobian to q/qdot directions.

    ``rhs`` follows :meth:`jacobian_mul` conventions with input dimension
    ``2 * dof``.  The scalar energy output retains its row dimension of one.
    """
    if self._resolve_dynamics_backend(backend) == "numpy":
      return self._numpy_kinetic_energy_apply(rhs, transpose=False)
    motion = np.asarray(self.motion(2), dtype=float)
    batch_shape = motion.shape[:-1] if batch_shapes.is_batched_feature_array(motion) else ()
    input_dim = self._rust_model_info().dof * 2
    rhs, rhs_is_matrix = batch_shapes.broadcast_feature_rhs(rhs, batch_shape, input_dim, name="rhs")
    tangent = rhs if rhs_is_matrix else rhs[..., None]
    robot = self._rust_compiled_robot()
    if not batch_shape:
      out = np.asarray(robot.kinetic_energy_jacobian_mul_rhs(
        np.ascontiguousarray(motion), np.ascontiguousarray(tangent),
      ))
    else:
      flat_motion = np.ascontiguousarray(motion.reshape((-1, motion.shape[-1])))
      out = np.asarray(robot.kinetic_energy_jacobian_mul_rhs_batch(
        flat_motion, np.ascontiguousarray(tangent),
      )).reshape(batch_shape + (1, tangent.shape[-1]))
    return out if rhs_is_matrix else out[..., 0]

  def kinetic_energy_jacobian_transpose_mul(self, rhs : np.ndarray, *, backend=None):
    """Apply the kinetic-energy VJP to scalar output cotangents.

    ``rhs`` has one output row, with optional final RHS-column axis.  The
    returned gradient is ordered ``[q0, qdot0, q1, qdot1, ...]``.
    """
    if self._resolve_dynamics_backend(backend) == "numpy":
      return self._numpy_kinetic_energy_apply(rhs, transpose=True)
    motion = np.asarray(self.motion(2), dtype=float)
    batch_shape = motion.shape[:-1] if batch_shapes.is_batched_feature_array(motion) else ()
    rhs, rhs_is_matrix = batch_shapes.broadcast_feature_rhs(rhs, batch_shape, 1, name="rhs")
    cotangent = rhs if rhs_is_matrix else rhs[..., None]
    robot = self._rust_compiled_robot()
    if not batch_shape:
      out = np.asarray(robot.kinetic_energy_jacobian_transpose_mul_rhs(
        np.ascontiguousarray(motion), np.ascontiguousarray(cotangent),
      ))
    else:
      flat_motion = np.ascontiguousarray(motion.reshape((-1, motion.shape[-1])))
      out = np.asarray(robot.kinetic_energy_jacobian_transpose_mul_rhs_batch(
        flat_motion, np.ascontiguousarray(cotangent),
      )).reshape(batch_shape + (self._rust_model_info().dof * 2, cotangent.shape[-1]))
    return out if rhs_is_matrix else out[..., 0]

  def _numpy_energy_terms(self):
    from ..core.kernels.inertia import spatial_inertia
    if self.dim_ != 3 or any(link.dof for link in self.robot_.links):
      raise NotImplementedError("NumPy kinetic energy requires 3D rigid links")
    motion = np.asarray(self.motion(2), dtype=float)
    state = outward_api.build_kinematics_outward_state(self.robot_, motion, order=2)
    selections, momenta = [], []
    energy = np.zeros(motion.shape[:-1], dtype=float)
    for link in self.robot_.links[1:]:
      selection = StateType("link", link.name, "vel", "local")
      velocity = np.asarray(outward_api.get_value(self.robot_, state, selection))
      momentum = velocity @ spatial_inertia(link.mass, link.inertia, link.cog).T
      energy += .5 * np.sum(velocity * momentum, axis=-1)
      selections.append(selection)
      momenta.append(momentum)
    weights = np.concatenate(momenta, axis=-1) if momenta else np.zeros(motion.shape[:-1] + (0,))
    return state, selections, weights, energy

  def _numpy_kinetic_energy(self):
    energy = self._numpy_energy_terms()[3]
    return float(energy) if energy.ndim == 0 else energy

  def _numpy_kinetic_energy_apply(self, rhs, *, transpose):
    state, selections, weights, energy = self._numpy_energy_terms()
    batch_shape = energy.shape
    input_dim = self.robot_.dof * 2
    rhs, matrix = batch_shapes.broadcast_feature_rhs(
      rhs, batch_shape, 1 if transpose else input_dim, name="rhs")
    rhs = rhs.reshape(batch_shape + rhs.shape[-2:]) if matrix else rhs.reshape(batch_shape + (rhs.shape[-1],))
    if not selections:
      shape = batch_shape + ((input_dim,) if transpose else (1,))
      return np.zeros(shape + ((rhs.shape[-1],) if matrix else ()))
    if transpose:
      # One reverse product yields dE/dmotion; scalar output weights share it.
      gradient = outward_api.outward_jacobian_transpose_matvec(
        self.robot_, state, selections, weights, max_time_order=2, dim=self.dim_)
      return gradient[..., :, None] * rhs if matrix else gradient * rhs
    if matrix:
      velocity_tangent = outward_api.outward_jacobian_matmul_rhs(
        self.robot_, state, selections, rhs, max_time_order=2, dim=self.dim_)
      return np.sum(weights[..., :, None] * velocity_tangent, axis=-2, keepdims=True)
    velocity_tangent = outward_api.outward_jacobian_matvec(
      self.robot_, state, selections, rhs, max_time_order=2, dim=self.dim_)
    return np.sum(weights * velocity_tangent, axis=-1, keepdims=True)
