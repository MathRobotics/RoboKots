"""Generic Jacobian, JVP, and VJP dispatch for the ``Kots`` facade."""
from __future__ import annotations

import numpy as np

from .. import outward as outward_api
from ..core import batch as batch_api
from ..core.state import StateType, data_type_dof, dim_to_dof, keys_force, keys_kinematics, keys_momentum, keys_torque
from ..core.state_tensor import JacobianTensor


class DerivativesMixin:
  def _jacobian_numerical(self, state_type_list, max_order : int, list_output : bool = False):
    if not self.motions_.is_batched():
      jacobs = [
        outward_api.jacobian_numerical(
          self.robot_, self.motions_, st, max_order, gravity=self.gravity_
        )
        for st in state_type_list
      ]
      return jacobs if list_output else np.vstack(jacobs)

    flat_motion, batch_shape = batch_api.flatten_feature_batch(self.motion(max_order))
    sample_results = [
      [
        outward_api.jacobian_numerical(
          self.robot_, self._sample_motions(x, max_order), st, max_order, gravity=self.gravity_
        )
        for st in state_type_list
      ]
      for x in flat_motion
    ]
    return batch_api.stack_results(
      sample_results,
      batch_shape,
      list_output,
      len(state_type_list),
      combine=np.vstack,
    )

  def _has_gravity_force_output(self, state_type_list) -> bool:
    """Whether an analytic force/torque Jacobian must include gravity."""
    return np.any(self.gravity_) and any(
      st.data_type in keys_force or st.data_type in keys_torque
      for st in state_type_list
    )

  def _jacobian_from_state(self, state, state_type_list, max_order : int, list_output : bool = False):
    fast = self._joint_motion_torque_jacobian(state, state_type_list, max_order, list_output=list_output)
    if fast is not None:
      return fast
    fast = self._rust_torque_jacobian(state_type_list, max_order, list_output=list_output)
    if fast is not None:
      return fast
    fast = self._rust_cmtm_torque_jacobian(state_type_list, max_order, list_output=list_output)
    if fast is not None:
      return fast
    fast = self._rust_link_local_jacobian(state_type_list, max_order, list_output=list_output)
    if fast is not None:
      return fast

    if not isinstance(state, list):
      if self.batch_shape_:
        try:
          return outward_api.outward_jacobian(self.robot_, state, state_type_list, max_time_order=max_order, dim = self.dim_, list_output = list_output)
        except (AttributeError, IndexError, TypeError, ValueError):
          # Fall back when the cached state cannot be consumed as a batched
          # outward state. Unexpected runtime errors should still surface.
          pass
        is_dynamics = any(st.is_dynamics for st in state_type_list)
        _, build_state = self._state_builder(max_order, is_dynamics=is_dynamics)
        flat_motion, batch_shape = batch_api.flatten_feature_batch(self.motion(max_order))
        states = [build_state(x) for x in flat_motion]
        sample_results = [
          outward_api.outward_jacobian(self.robot_, st, state_type_list, max_time_order=max_order, dim = self.dim_, list_output = list_output)
          for st in states
        ]
        return batch_api.stack_results(
          sample_results,
          batch_shape,
          list_output,
          len(state_type_list),
        )
      return outward_api.outward_jacobian(self.robot_, state, state_type_list, dim = self.dim_, list_output = list_output)

    sample_results = [
      outward_api.outward_jacobian(self.robot_, st, state_type_list, max_time_order=max_order, dim = self.dim_, list_output = list_output)
      for st in state
    ]
    return batch_api.stack_results(
      sample_results,
      self.batch_shape_,
      list_output,
      len(state_type_list),
    )

  def _jacobian_matvec_numerical(self, state_type_list, max_order : int, vec, list_output : bool = False):
    if not self.motions_.is_batched():
      results = [
        outward_api.jacobian_numerical(
          self.robot_, self.motions_, st, max_order, gravity=self.gravity_
        ) @ vec
        for st in state_type_list
      ]
      return results if list_output else np.concatenate(results)

    flat_motion, batch_shape = batch_api.flatten_feature_batch(self.motion(max_order))
    sample_results = []
    for x, v in zip(flat_motion, vec):
      sample_motions = self._sample_motions(x, max_order)
      parts = [
        outward_api.jacobian_numerical(
          self.robot_, sample_motions, st, max_order, gravity=self.gravity_
        ) @ v
        for st in state_type_list
      ]
      sample_results.append(parts if list_output else np.concatenate(parts))
    return batch_api.stack_results(
      sample_results,
      batch_shape,
      list_output,
      len(state_type_list),
    )

  def _jacobian_matvec_from_state(self, state, state_type_list, max_order : int, vec, batch_shape : tuple, list_output : bool = False):
    if any(st.owner_type in ("link", "joint") and st.data_type in keys_force and st.frame_name == "world" for st in state_type_list):
      jacob = self._jacobian_from_state(state, state_type_list, max_order, False)
      vec_part = vec.reshape(batch_shape + (vec.shape[-1],)) if batch_shape else vec
      applied = (jacob @ vec_part[..., None])[..., 0]
      if not list_output:
        return applied
      sizes = [self._jacobian_output_dim([st]) for st in state_type_list]
      offsets = np.cumsum([0] + sizes)
      return [applied[..., offsets[i]:offsets[i + 1]] for i in range(len(sizes))]
    fast = self._joint_motion_torque_jacobian_apply(state, state_type_list, max_order, vec, batch_shape, rhs_is_matrix=False, list_output=list_output)
    if fast is not None:
      return fast
    fast = self._rust_torque_jacobian_apply(state_type_list, max_order, vec, batch_shape, rhs_is_matrix=False, list_output=list_output)
    if fast is not None:
      return fast
    fast = self._rust_cmtm_torque_jacobian_apply(state_type_list, max_order, vec, batch_shape, rhs_is_matrix=False, list_output=list_output)
    if fast is not None:
      return fast
    fast = self._rust_link_local_jacobian_apply(state_type_list, max_order, vec, batch_shape, rhs_is_matrix=False, list_output=list_output)
    if fast is not None:
      return fast

    if not isinstance(state, list):
      if batch_shape:
        try:
          direct_vec = vec.reshape(batch_shape + (vec.shape[-1],))
          return outward_api.outward_jacobian_matvec(self.robot_, state, state_type_list, direct_vec, max_time_order=max_order, dim = self.dim_, list_output = list_output)
        except (AttributeError, IndexError, TypeError, ValueError):
          # Fall back when the cached state cannot be consumed as a batched
          # outward state. Unexpected runtime errors should still surface.
          pass
        is_dynamics = any(st.is_dynamics for st in state_type_list)
        _, build_state = self._state_builder(max_order, is_dynamics=is_dynamics)
        flat_motion, _ = batch_api.flatten_feature_batch(self.motion(max_order))
        states = [build_state(x) for x in flat_motion]
        sample_results = [
          outward_api.outward_jacobian_matvec(self.robot_, st, state_type_list, v, max_time_order=max_order, dim = self.dim_, list_output = list_output)
          for st, v in zip(states, vec)
        ]
        return batch_api.stack_results(
          sample_results,
          batch_shape,
          list_output,
          len(state_type_list),
        )
      return outward_api.outward_jacobian_matvec(self.robot_, state, state_type_list, vec, dim = self.dim_, list_output = list_output)

    sample_results = [
      outward_api.outward_jacobian_matvec(self.robot_, st, state_type_list, v, max_time_order=max_order, dim = self.dim_, list_output = list_output)
      for st, v in zip(state, vec)
    ]
    return batch_api.stack_results(
      sample_results,
      batch_shape,
      list_output,
      len(state_type_list),
    )

  def _jacobian_matmul_rhs_from_state(self, state, state_type_list, max_order : int, rhs, batch_shape : tuple, list_output : bool = False):
    if any(st.owner_type in ("link", "joint") and st.data_type in keys_force and st.frame_name == "world" for st in state_type_list):
      jacob = self._jacobian_from_state(state, state_type_list, max_order, False)
      rhs_part = rhs.reshape(batch_shape + rhs.shape[-2:]) if batch_shape else rhs
      applied = jacob @ rhs_part
      if not list_output:
        return applied
      sizes = [self._jacobian_output_dim([st]) for st in state_type_list]
      offsets = np.cumsum([0] + sizes)
      return [applied[..., offsets[i]:offsets[i + 1], :] for i in range(len(sizes))]
    fast = self._joint_motion_torque_jacobian_apply(state, state_type_list, max_order, rhs, batch_shape, rhs_is_matrix=True, list_output=list_output)
    if fast is not None:
      return fast
    fast = self._rust_torque_jacobian_apply(state_type_list, max_order, rhs, batch_shape, rhs_is_matrix=True, list_output=list_output)
    if fast is not None:
      return fast
    fast = self._rust_cmtm_torque_jacobian_apply(state_type_list, max_order, rhs, batch_shape, rhs_is_matrix=True, list_output=list_output)
    if fast is not None:
      return fast
    fast = self._rust_link_local_jacobian_apply(state_type_list, max_order, rhs, batch_shape, rhs_is_matrix=True, list_output=list_output)
    if fast is not None:
      return fast

    if not isinstance(state, list):
      if batch_shape:
        try:
          direct_rhs = rhs.reshape(batch_shape + rhs.shape[-2:])
          return outward_api.outward_jacobian_matmul_rhs(self.robot_, state, state_type_list, direct_rhs, max_time_order=max_order, dim = self.dim_, list_output = list_output)
        except (AttributeError, IndexError, TypeError, ValueError):
          # Fall back when the cached state cannot be consumed as a batched
          # outward state. Unexpected runtime errors should still surface.
          pass
        is_dynamics = any(st.is_dynamics for st in state_type_list)
        _, build_state = self._state_builder(max_order, is_dynamics=is_dynamics)
        flat_motion, _ = batch_api.flatten_feature_batch(self.motion(max_order))
        states = [build_state(x) for x in flat_motion]
        sample_results = [
          outward_api.outward_jacobian_matmul_rhs(self.robot_, st, state_type_list, r, max_time_order=max_order, dim = self.dim_, list_output = list_output)
          for st, r in zip(states, rhs)
        ]
        return batch_api.stack_results(
          sample_results,
          batch_shape,
          list_output,
          len(state_type_list),
        )
      return outward_api.outward_jacobian_matmul_rhs(self.robot_, state, state_type_list, rhs, dim = self.dim_, list_output = list_output)

    sample_results = [
      outward_api.outward_jacobian_matmul_rhs(self.robot_, st, state_type_list, r, max_time_order=max_order, dim = self.dim_, list_output = list_output)
      for st, r in zip(state, rhs)
    ]
    return batch_api.stack_results(
      sample_results,
      batch_shape,
      list_output,
      len(state_type_list),
    )

  def _stack_mul_columns(self, column_results, list_output : bool, item_count : int):
    if list_output:
      return [
        np.stack([column[i] for column in column_results], axis=-1)
        for i in range(item_count)
      ]
    return np.stack(column_results, axis=-1)

  def _jacobian_mul_numerical(self, state_type_list, max_order : int, rhs, rhs_is_matrix : bool, list_output : bool = False):
    if not rhs_is_matrix:
      return self._jacobian_matvec_numerical(state_type_list, max_order, rhs, list_output)

    rhs_count = rhs.shape[-1]
    column_results = [
      self._jacobian_matvec_numerical(state_type_list, max_order, rhs[..., i], list_output)
      for i in range(rhs_count)
    ]
    return self._stack_mul_columns(column_results, list_output, len(state_type_list))

  def _jacobian_mul_from_state(self, state, state_type_list, max_order : int, rhs, batch_shape : tuple, rhs_is_matrix : bool, list_output : bool = False):
    if not rhs_is_matrix:
      return self._jacobian_matvec_from_state(state, state_type_list, max_order, rhs, batch_shape, list_output)

    if not isinstance(state, list):
      return self._jacobian_matmul_rhs_from_state(state, state_type_list, max_order, rhs, batch_shape, list_output)

    rhs_count = rhs.shape[-1]
    column_results = [
      self._jacobian_matvec_from_state(state, state_type_list, max_order, rhs[..., i], batch_shape, list_output)
      for i in range(rhs_count)
    ]
    return self._stack_mul_columns(column_results, list_output, len(state_type_list))

  def _jacobian_output_dim(self, state_type_list) -> int:
    dim_dof = dim_to_dof(self.dim_)
    output_dim = 0
    for st in state_type_list:
      if self._is_total_body_kinetic_energy(st):
        output_dim += 1
      elif self._is_joint_motion_state(st):
        joint = self.robot_.joint(st.owner_name)
        if joint is None:
          raise ValueError(f"Invalid joint name: {st.owner_name}")
        output_dim += joint.dof
      elif st.data_type in keys_kinematics:
        output_dim += data_type_dof(st.data_type, dim=self.dim_)
      elif st.data_type in keys_momentum or st.data_type in keys_force:
        output_dim += dim_dof
      elif st.data_type in keys_torque:
        if st.owner_type != "joint":
          raise ValueError("torque can be specified only for joint owner type")
        joint = self.robot_.joint(st.owner_name)
        if joint is None:
          raise ValueError(f"Invalid joint name: {st.owner_name}")
        output_dim += joint.dof
      else:
        raise ValueError(f"Unsupported data_type: {st.data_type}")
    return output_dim

  def _jacobian_transpose_matvec_numerical(self, state_type_list, max_order : int, vec):
    if not self.motions_.is_batched():
      jacob = self._jacobian_numerical(state_type_list, max_order)
      return jacob.T @ vec

    flat_motion, batch_shape = batch_api.flatten_feature_batch(self.motion(max_order))
    sample_results = []
    for x, v in zip(flat_motion, vec):
      sample_motions = self._sample_motions(x, max_order)
      parts = [
        outward_api.jacobian_numerical(
          self.robot_, sample_motions, st, max_order, gravity=self.gravity_
        )
        for st in state_type_list
      ]
      jacob = np.vstack(parts)
      sample_results.append(jacob.T @ v)
    return batch_api.stack_sample_results(sample_results, batch_shape)

  def _jacobian_transpose_matvec_from_state(self, state, state_type_list, max_order : int, vec, batch_shape : tuple):
    world_vjp = getattr(self, "_rust_cmtm_world_link_dynamics_jacobian_transpose_apply", None)
    if world_vjp is not None:
      fast = world_vjp(state_type_list, max_order, vec, batch_shape, rhs_is_matrix=False)
      if fast is not None:
        return fast
    world_joint_vjp = getattr(self, "_rust_cmtm_world_joint_dynamics_jacobian_transpose_apply", None)
    if world_joint_vjp is not None:
      fast = world_joint_vjp(state_type_list, max_order, vec, batch_shape, rhs_is_matrix=False)
      if fast is not None:
        return fast
    if any(st.owner_type in ("link", "joint") and st.data_type in keys_force and st.frame_name == "world" for st in state_type_list):
      jacob = self._jacobian_from_state(state, state_type_list, max_order, False)
      vec_part = vec.reshape(batch_shape + (vec.shape[-1],)) if batch_shape else vec
      return (np.swapaxes(jacob, -1, -2) @ vec_part[..., None])[..., 0]
    # Try analytic backend kernels before considering the gravity-aware
    # outward fallback.  In particular, the RNEA Rust VJP includes gravity;
    # routing batched force/torque outputs above this point used to make the
    # fast path unreachable.
    # Higher-order torque series has a separate CMTM VJP.  This needs to be
    # attempted before the mixed joint-state helper, which otherwise routes
    # torque_diff1 and above to Python outward reverse mode.
    cmtm_vjp = getattr(self, "_rust_cmtm_world_link_dynamics_jacobian_transpose_apply", None)
    if cmtm_vjp is not None:
      fast = cmtm_vjp(state_type_list, max_order, vec, batch_shape, rhs_is_matrix=False)
      if fast is not None:
        return fast
    cmtm_vjp = getattr(self, "_rust_cmtm_jacobian_transpose_apply", None)
    if cmtm_vjp is not None:
      fast = cmtm_vjp(state_type_list, max_order, vec, batch_shape, rhs_is_matrix=False)
      if fast is not None:
        return fast
    cmtm_vjp = getattr(self, "_rust_cmtm_outward_dynamics_jacobian_transpose_apply", None)
    if cmtm_vjp is not None:
      fast = cmtm_vjp(state_type_list, max_order, vec, batch_shape, rhs_is_matrix=False)
      if fast is not None:
        return fast
    cmtm_vjp = getattr(self, "_rust_cmtm_torque_jacobian_transpose_apply", None)
    if cmtm_vjp is not None:
      fast = cmtm_vjp(state_type_list, max_order, vec, batch_shape, rhs_is_matrix=False)
      if fast is not None:
        return fast
    fast = self._joint_motion_torque_jacobian_transpose_apply(state, state_type_list, max_order, vec, batch_shape, rhs_is_matrix=False)
    if fast is not None:
      return fast
    fast = self._rust_torque_jacobian_transpose_apply(state_type_list, max_order, vec, batch_shape, rhs_is_matrix=False)
    if fast is not None:
      return fast
    fast = self._rust_link_local_jacobian_transpose_apply(state_type_list, max_order, vec, batch_shape, rhs_is_matrix=False)
    if fast is not None:
      return fast

    if self._has_gravity_force_output(state_type_list) and batch_shape and not isinstance(state, list):
      is_dynamics = any(st.is_dynamics for st in state_type_list)
      _, build_state = self._state_builder(
        max_order, is_dynamics=is_dynamics, gravity=self.gravity_
      )
      flat_motion, _ = batch_api.flatten_feature_batch(self.motion(max_order))
      flat_vec = np.asarray(vec).reshape((-1, vec.shape[-1]))
      sample_results = [
        outward_api.outward_jacobian_transpose_matvec(
          self.robot_, build_state(x), state_type_list, v,
          max_time_order=max_order, dim=self.dim_,
        )
        for x, v in zip(flat_motion, flat_vec)
      ]
      return batch_api.stack_sample_results(sample_results, batch_shape)

    if not isinstance(state, list):
      if batch_shape:
        try:
          direct_vec = vec.reshape(batch_shape + (vec.shape[-1],))
          return outward_api.outward_jacobian_transpose_matvec(
            self.robot_,
            state,
            state_type_list,
            direct_vec,
            max_time_order=max_order,
            dim=self.dim_,
          )
        except (AttributeError, IndexError, TypeError, ValueError):
          # Fall back when the cached state cannot be consumed as a batched
          # outward state. Unexpected runtime errors should still surface.
          pass
        is_dynamics = any(st.is_dynamics for st in state_type_list)
        _, build_state = self._state_builder(max_order, is_dynamics=is_dynamics)
        flat_motion, _ = batch_api.flatten_feature_batch(self.motion(max_order))
        states = [build_state(x) for x in flat_motion]
        sample_results = [
          outward_api.outward_jacobian_transpose_matvec(self.robot_, st, state_type_list, v, max_time_order=max_order, dim=self.dim_)
          for st, v in zip(states, vec)
        ]
        return batch_api.stack_sample_results(sample_results, batch_shape)
      return outward_api.outward_jacobian_transpose_matvec(self.robot_, state, state_type_list, vec, dim=self.dim_)

    sample_results = [
      outward_api.outward_jacobian_transpose_matvec(self.robot_, st, state_type_list, v, max_time_order=max_order, dim=self.dim_)
      for st, v in zip(state, vec)
    ]
    return batch_api.stack_sample_results(sample_results, batch_shape)

  def _jacobian_transpose_mul_numerical(self, state_type_list, max_order : int, rhs, rhs_is_matrix : bool):
    if not rhs_is_matrix:
      return self._jacobian_transpose_matvec_numerical(state_type_list, max_order, rhs)

    rhs_count = rhs.shape[-1]
    column_results = [
      self._jacobian_transpose_matvec_numerical(state_type_list, max_order, rhs[..., i])
      for i in range(rhs_count)
    ]
    return np.stack(column_results, axis=-1)

  def _jacobian_transpose_mul_from_state(self, state, state_type_list, max_order : int, rhs, batch_shape : tuple, rhs_is_matrix : bool):
    if not rhs_is_matrix:
      return self._jacobian_transpose_matvec_from_state(state, state_type_list, max_order, rhs, batch_shape)

    world_vjp = getattr(self, "_rust_cmtm_world_link_dynamics_jacobian_transpose_apply", None)
    if world_vjp is not None:
      fast = world_vjp(state_type_list, max_order, rhs, batch_shape, rhs_is_matrix=True)
      if fast is not None:
        return fast
    world_joint_vjp = getattr(self, "_rust_cmtm_world_joint_dynamics_jacobian_transpose_apply", None)
    if world_joint_vjp is not None:
      fast = world_joint_vjp(state_type_list, max_order, rhs, batch_shape, rhs_is_matrix=True)
      if fast is not None:
        return fast

    if any(st.owner_type in ("link", "joint") and st.data_type in keys_force and st.frame_name == "world" for st in state_type_list):
      jacob = self._jacobian_from_state(state, state_type_list, max_order, False)
      rhs_part = rhs.reshape(batch_shape + rhs.shape[-2:]) if batch_shape else rhs
      return np.swapaxes(jacob, -1, -2) @ rhs_part

    cmtm_vjp = getattr(self, "_rust_cmtm_world_link_dynamics_jacobian_transpose_apply", None)
    if cmtm_vjp is not None:
      fast = cmtm_vjp(state_type_list, max_order, rhs, batch_shape, rhs_is_matrix=True)
      if fast is not None:
        return fast
    cmtm_vjp = getattr(self, "_rust_cmtm_jacobian_transpose_apply", None)
    if cmtm_vjp is not None:
      fast = cmtm_vjp(state_type_list, max_order, rhs, batch_shape, rhs_is_matrix=True)
      if fast is not None:
        return fast
    cmtm_vjp = getattr(self, "_rust_cmtm_outward_dynamics_jacobian_transpose_apply", None)
    if cmtm_vjp is not None:
      fast = cmtm_vjp(state_type_list, max_order, rhs, batch_shape, rhs_is_matrix=True)
      if fast is not None:
        return fast
    cmtm_vjp = getattr(self, "_rust_cmtm_torque_jacobian_transpose_apply", None)
    if cmtm_vjp is not None:
      fast = cmtm_vjp(state_type_list, max_order, rhs, batch_shape, rhs_is_matrix=True)
      if fast is not None:
        return fast
    fast = self._joint_motion_torque_jacobian_transpose_apply(state, state_type_list, max_order, rhs, batch_shape, rhs_is_matrix=True)
    if fast is not None:
      return fast
    fast = self._rust_torque_jacobian_transpose_apply(state_type_list, max_order, rhs, batch_shape, rhs_is_matrix=True)
    if fast is not None:
      return fast
    fast = self._rust_link_local_jacobian_transpose_apply(state_type_list, max_order, rhs, batch_shape, rhs_is_matrix=True)
    if fast is not None:
      return fast

    if self._has_gravity_force_output(state_type_list) and batch_shape:
      flat_rhs = np.asarray(rhs).reshape((-1,) + rhs.shape[-2:])
      if isinstance(state, list):
        states = state
      else:
        is_dynamics = any(st.is_dynamics for st in state_type_list)
        _, build_state = self._state_builder(
          max_order, is_dynamics=is_dynamics, gravity=self.gravity_
        )
        flat_motion, _ = batch_api.flatten_feature_batch(self.motion(max_order))
        states = [build_state(x) for x in flat_motion]
      sample_results = [
        np.moveaxis(
          outward_api.outward_jacobian_transpose_matvec(
            self.robot_, st, state_type_list, np.moveaxis(r, -1, 0),
            max_time_order=max_order, dim=self.dim_,
          ),
          0,
          -1,
        )
        for st, r in zip(states, flat_rhs)
      ]
      return batch_api.stack_sample_results(sample_results, batch_shape)

    if (
      not batch_shape
      and not isinstance(state, list)
    ):
      block_result = self._jacobian_transpose_matvec_from_state(
        state,
        state_type_list,
        max_order,
        np.swapaxes(rhs, -1, -2),
        batch_shape,
      )
      return np.swapaxes(block_result, -1, -2)

    if batch_shape and not isinstance(state, list):
      jacob = self._jacobian_from_state(state, state_type_list, max_order)
      rhs_batch = rhs.reshape(batch_shape + rhs.shape[-2:])
      return np.swapaxes(jacob, -1, -2) @ rhs_batch

    rhs_count = rhs.shape[-1]
    column_results = [
      self._jacobian_transpose_matvec_from_state(state, state_type_list, max_order, rhs[..., i], batch_shape)
      for i in range(rhs_count)
    ]
    return np.stack(column_results, axis=-1)

  def jacobian(self, state_type, numerical : bool = False, list_output : bool = False):
    state_type_list = self._state_type_list(state_type)
    if any(self._is_total_body_kinetic_energy(st) for st in state_type_list) and not all(self._is_total_body_kinetic_energy(st) for st in state_type_list):
      max_order = StateType.max_time_order(state_type_list)
      parts = []
      for st in state_type_list:
        part = self.jacobian(st, numerical=numerical)
        source_order = StateType.max_time_order([st])
        parts.append(self._embed_motion_order_jacobian(np.asarray(part), source_order, max_order))
      return parts if list_output else np.concatenate(parts, axis=-2)
    if state_type_list and all(self._is_total_body_kinetic_energy(st) for st in state_type_list):
      if numerical:
        raise NotImplementedError("numerical Jacobian is not implemented for total_body kinetic_energy")
      batch_shape = self.batch_shape_ if self.batch_shape_ else self.motions_.batch_shape()
      ones = np.ones(batch_shape + (1,)) if batch_shape else np.ones(1)
      jacobian = self.kinetic_energy_jacobian_transpose_mul(ones)[..., None, :]
      parts = [jacobian for _ in state_type_list]
      return parts if list_output else np.concatenate(parts, axis=-2)
    max_order = StateType.max_time_order(state_type_list)
    if numerical:
      return self._jacobian_numerical(state_type_list, max_order, list_output)

    state = self.outward_state_ if self.outward_state_ is not None else self.state_dict_
    return self._jacobian_from_state(state, state_type_list, max_order, list_output)

  def jacobian_mul(self, state_type, rhs : np.ndarray, numerical : bool = False, list_output : bool = False):
    """
    Compute J @ rhs for the Jacobian of ``state_type``.

    ``rhs`` may be a vector with shape ``(motion_dim,)`` or
    ``batch_shape + (motion_dim,)``, or a matrix with shape
    ``(motion_dim, rhs_dim)`` or ``batch_shape + (motion_dim, rhs_dim)``.
    """
    state_type_list = self._state_type_list(state_type)
    max_order = StateType.max_time_order(state_type_list)
    input_dim = self.robot_.dof * max_order
    batch_shape = self.batch_shape_ if self.batch_shape_ else self.motions_.batch_shape()
    rhs, rhs_is_matrix = batch_api.broadcast_feature_rhs(rhs, batch_shape, input_dim, name="rhs")

    if any(self._is_total_body_kinetic_energy(st) for st in state_type_list) and not all(self._is_total_body_kinetic_energy(st) for st in state_type_list):
      parts = []
      for st in state_type_list:
        source_order = StateType.max_time_order([st])
        part_rhs = self._select_motion_order_rhs(rhs, source_order, max_order, rhs_is_matrix)
        parts.append(self.jacobian_mul(st, part_rhs, numerical=numerical))
      return parts if list_output else np.concatenate(parts, axis=-2 if rhs_is_matrix else -1)

    if state_type_list and all(self._is_total_body_kinetic_energy(st) for st in state_type_list):
      if numerical:
        raise NotImplementedError("numerical Jacobian is not implemented for total_body kinetic_energy")
      value = self.kinetic_energy_jacobian_mul(rhs)
      parts = [value for _ in state_type_list]
      if list_output:
        return parts
      return np.concatenate(parts, axis=-2 if rhs_is_matrix else -1)

    if numerical:
      return self._jacobian_mul_numerical(state_type_list, max_order, rhs, rhs_is_matrix, list_output)

    state = self.outward_state_ if self.outward_state_ is not None else self.state_dict_
    return self._jacobian_mul_from_state(state, state_type_list, max_order, rhs, batch_shape, rhs_is_matrix, list_output)

  def jacobian_transpose_mul(self, state_type, rhs : np.ndarray, numerical : bool = False):
    """
    Compute J.T @ rhs for the Jacobian of ``state_type``.

    ``rhs`` may be a vector with shape ``(total_state_dim,)`` or
    ``batch_shape + (total_state_dim,)``, or a matrix with shape
    ``(total_state_dim, rhs_dim)`` or ``batch_shape + (total_state_dim, rhs_dim)``.
    """
    state_type_list = self._state_type_list(state_type)
    max_order = StateType.max_time_order(state_type_list)
    output_dim = self._jacobian_output_dim(state_type_list)
    batch_shape = self.batch_shape_ if self.batch_shape_ else self.motions_.batch_shape()
    rhs, rhs_is_matrix = batch_api.broadcast_feature_rhs(rhs, batch_shape, output_dim, name="rhs")

    if any(self._is_total_body_kinetic_energy(st) for st in state_type_list) and not all(self._is_total_body_kinetic_energy(st) for st in state_type_list):
      out = np.zeros(rhs.shape[:-(2 if rhs_is_matrix else 1)] + (self.robot_.dof * max_order,) + ((rhs.shape[-1],) if rhs_is_matrix else ()), dtype=rhs.dtype)
      row = 0
      for st in state_type_list:
        width = self._jacobian_output_dim([st])
        part_rhs = rhs[..., row:row + width, :] if rhs_is_matrix else rhs[..., row:row + width]
        source_order = StateType.max_time_order([st])
        part = self.jacobian_transpose_mul(st, part_rhs, numerical=numerical)
        out += self._embed_motion_order_rhs(np.asarray(part), source_order, max_order, rhs_is_matrix)
        row += width
      return out

    if state_type_list and all(self._is_total_body_kinetic_energy(st) for st in state_type_list):
      if numerical:
        raise NotImplementedError("numerical Jacobian is not implemented for total_body kinetic_energy")
      energy_rhs = np.sum(rhs, axis=-2 if rhs_is_matrix else -1, keepdims=True)
      return self.kinetic_energy_jacobian_transpose_mul(energy_rhs)

    if numerical:
      return self._jacobian_transpose_mul_numerical(state_type_list, max_order, rhs, rhs_is_matrix)

    state = self.outward_state_ if self.outward_state_ is not None else self.state_dict_
    return self._jacobian_transpose_mul_from_state(state, state_type_list, max_order, rhs, batch_shape, rhs_is_matrix)

  def jacobian_transpose_mul_many(self, state_rhs_pairs, numerical : bool = False):
    """Fuse VJPs for several state references into one reverse pass.

    ``state_rhs_pairs`` is a non-empty sequence of ``(state_type, rhs)``
    pairs.  Each ``state_type`` accepts the same ``StateType`` or list form as
    :meth:`jacobian_transpose_mul`; its RHS uses that state's own output-row
    count.  All pairs must use either vectors or matrices.  Matrix RHS inputs
    must have a common final column count.  The returned VJP is the sum over
    pairs, retaining that common RHS-column axis.

    This is intended for callers which naturally group cotangents by a state
    reference (for example ``torque`` and ``torque_diff1``).  The state rows
    are concatenated before dispatch, so compatible CMTM dynamics states use
    one Rust reverse recurrence rather than one recurrence per group.
    """
    if not isinstance(state_rhs_pairs, (list, tuple)) or not state_rhs_pairs:
      raise ValueError("state_rhs_pairs must be a non-empty sequence of (state_type, rhs) pairs")

    batch_shape = self.batch_shape_ if self.batch_shape_ else self.motions_.batch_shape()
    state_type_list = []
    rhs_parts = []
    energy_rhs_parts = []
    rhs_is_matrix = None
    rhs_cols = None

    for index, pair in enumerate(state_rhs_pairs):
      if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise ValueError(f"state_rhs_pairs[{index}] must be a (state_type, rhs) pair")
      pair_states = self._state_type_list(pair[0])
      if not pair_states:
        raise ValueError(f"state_rhs_pairs[{index}] must contain at least one StateType")
      pair_output_dim = self._jacobian_output_dim(pair_states)
      pair_rhs, pair_is_matrix = batch_api.broadcast_feature_rhs(
        pair[1], batch_shape, pair_output_dim, name=f"state_rhs_pairs[{index}][1]",
      )
      if rhs_is_matrix is None:
        rhs_is_matrix = pair_is_matrix
        rhs_cols = pair_rhs.shape[-1] if pair_is_matrix else None
      elif rhs_is_matrix != pair_is_matrix:
        raise ValueError("all state_rhs_pairs RHS values must all be vectors or all be matrices")
      elif pair_is_matrix and pair_rhs.shape[-1] != rhs_cols:
        raise ValueError("all matrix RHS values must have the same number of columns")
      row = 0
      for st in pair_states:
        width = self._jacobian_output_dim([st])
        part = pair_rhs[..., row:row + width, :] if pair_is_matrix else pair_rhs[..., row:row + width]
        if self._is_total_body_kinetic_energy(st):
          energy_rhs_parts.append(part)
        else:
          state_type_list.append(st)
          rhs_parts.append(part)
        row += width

    max_order = StateType.max_time_order([
      *state_type_list,
      *([StateType("total_body", "total_body", "kinetic_energy")] if energy_rhs_parts else []),
    ])
    if numerical:
      # Kinetic energy has no numerical implementation.  Preserve the public
      # single-VJP contract instead of silently routing it through outward.
      if energy_rhs_parts:
        raise NotImplementedError("numerical Jacobian is not implemented for total_body kinetic_energy")
      fused_rhs = np.concatenate(rhs_parts, axis=-2 if rhs_is_matrix else -1)
      return self._jacobian_transpose_mul_numerical(state_type_list, max_order, fused_rhs, rhs_is_matrix)

    energy_rhs = None
    if energy_rhs_parts:
      energy_rhs = np.sum(np.stack(energy_rhs_parts, axis=0), axis=0)

    if not state_type_list:
      energy_vjp = self.kinetic_energy_jacobian_transpose_mul(energy_rhs)
      return self._embed_motion_order_rhs(np.asarray(energy_vjp), 2, max_order, rhs_is_matrix)

    fused_rhs = np.concatenate(rhs_parts, axis=-2 if rhs_is_matrix else -1)
    state = self.outward_state_ if self.outward_state_ is not None else self.state_dict_
    if energy_rhs is not None:
      fast = self._rust_cmtm_torque_energy_jacobian_transpose_apply(
        state_type_list, fused_rhs, energy_rhs, max_order, batch_shape, rhs_is_matrix,
      )
      if fast is not None:
        return fast

    dynamics_vjp = self._jacobian_transpose_mul_from_state(
      state, state_type_list, max_order, fused_rhs, batch_shape, rhs_is_matrix,
    )
    if energy_rhs is None:
      return dynamics_vjp
    energy_vjp = self.kinetic_energy_jacobian_transpose_mul(energy_rhs)
    return dynamics_vjp + self._embed_motion_order_rhs(
      np.asarray(energy_vjp), 2, max_order, rhs_is_matrix,
    )

  def squared_power_torque_vjp_terms(self, torque_state, power_rhs, torque_value=None):
    """Return the torque request and direct velocity VJP for ``(tau.T qdot)^2``.

    ``torque_request`` can be appended directly to
    :meth:`jacobian_transpose_mul_many`; ``motion_vjp_order2`` is the direct
    derivative through ``qdot`` and must be added by the caller after padding
    it to its optimization motion order.  Keeping that direct selector out of
    the dynamics request preserves the single Rust CMTM reverse pass for all
    torque-series losses.
    """
    states = self._state_type_list(torque_state)
    if len(states) != self.robot_.dof or any(st.data_type != "torque" for st in states):
      raise ValueError("squared_power_torque_vjp_terms requires a total_joint torque StateType")
    batch_shape = self.batch_shape_ if self.batch_shape_ else self.motions_.batch_shape()
    power_rhs, rhs_is_matrix = batch_api.broadcast_feature_rhs(
      power_rhs, batch_shape, 1, name="power_rhs",
    )
    tau = np.asarray(self.state_info(torque_state) if torque_value is None else torque_value, dtype=float)
    if tau.shape != batch_shape + (self.robot_.dof,):
      raise ValueError(f"torque_value must have shape {batch_shape + (self.robot_.dof,)}, got {tau.shape}")
    motion = np.asarray(self.motion(2), dtype=float).reshape(batch_shape + (self.robot_.dof, 2))
    velocity = motion[..., :, 1]
    power = np.sum(tau * velocity, axis=-1)
    if rhs_is_matrix:
      scale = 2.0 * power[..., None] * power_rhs[..., 0, :]
      torque_rhs = velocity[..., :, None] * scale[..., None, :]
      velocity_rhs = tau[..., :, None] * scale[..., None, :]
      motion_vjp = np.zeros(batch_shape + (self.robot_.dof * 2, scale.shape[-1]), dtype=float)
      motion_vjp.reshape(batch_shape + (self.robot_.dof, 2, scale.shape[-1]))[..., :, 1, :] = velocity_rhs
    else:
      scale = 2.0 * power * power_rhs[..., 0]
      torque_rhs = velocity * scale[..., None]
      velocity_rhs = tau * scale[..., None]
      motion_vjp = np.zeros(batch_shape + (self.robot_.dof * 2,), dtype=float)
      motion_vjp.reshape(batch_shape + (self.robot_.dof, 2))[..., :, 1] = velocity_rhs
    return {
      "torque_request": (torque_state, torque_rhs),
      "motion_vjp_order2": motion_vjp,
      "power": power,
    }

  def jacobian_tensor(self, state_type, numerical : bool = False) -> JacobianTensor:
    state_type_list = self._state_type_list(state_type)
    return JacobianTensor.from_array(self.jacobian(state_type_list, numerical=numerical), state_type_list)

  def jacobian_target(self, numerical : bool = False, list_output : bool = False):
    if self.target_ is None:
      raise ValueError("target is not set")

    return self.jacobian(self.target_._targets, numerical=numerical, list_output=list_output)

  def jacobian_target_mul(self, rhs : np.ndarray, numerical : bool = False, list_output : bool = False):
    if self.target_ is None:
      raise ValueError("target is not set")

    return self.jacobian_mul(self.target_._targets, rhs, numerical=numerical, list_output=list_output)

  def jacobian_target_transpose_mul(self, rhs : np.ndarray, numerical : bool = False):
    if self.target_ is None:
      raise ValueError("target is not set")

    return self.jacobian_transpose_mul(self.target_._targets, rhs, numerical=numerical)

  def jacobian_target_tensor(self, numerical : bool = False) -> JacobianTensor:
    if self.target_ is None:
      raise ValueError("target is not set")
    return JacobianTensor.from_array(self.jacobian_target(numerical=numerical), self.target_._targets)
