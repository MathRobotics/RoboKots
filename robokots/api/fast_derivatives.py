"""Specialized NumPy derivative paths for joint motion and torque states."""
from __future__ import annotations

import numpy as np

from .. import outward as outward_api
from ..core.state import StateType, keys_joint_motion, keys_torque


class FastDerivativesMixin:
  def _is_joint_motion_state(self, st : StateType) -> bool:
    return st.owner_type == "joint" and st.data_type in keys_joint_motion

  def _is_joint_torque_state(self, st : StateType) -> bool:
    return st.owner_type == "joint" and st.data_type in keys_torque and st.frame_name is None

  def _joint_motion_torque_supported(self, state_type_list) -> bool:
    has_motion = False
    for st in state_type_list:
      if self._is_joint_motion_state(st):
        has_motion = True
      elif not self._is_joint_torque_state(st):
        return False
    return has_motion

  def _joint_state_dof(self, st : StateType) -> int:
    joint = self.robot_.joint(st.owner_name)
    if joint is None or joint.dof <= 0:
      raise ValueError(f"Invalid active joint state: {st.owner_name}")
    return joint.dof

  def _joint_motion_col_slice(self, st : StateType, max_order : int) -> slice:
    joint = self.robot_.joint(st.owner_name)
    if joint is None or joint.dof <= 0:
      raise ValueError(f"Invalid active joint motion state: {st.owner_name}")
    motion_index = self._joint_motion_index(st.data_type)
    if motion_index is None or motion_index >= max_order:
      raise ValueError(f"{st.data_type} is not available for order={max_order}")
    start = joint.dof_index * max_order + motion_index * joint.dof
    return slice(start, start + joint.dof)

  def _joint_motion_selector_jacobian(self, st : StateType, max_order : int, batch_shape : tuple):
    joint = self.robot_.joint(st.owner_name)
    col_slice = self._joint_motion_col_slice(st, max_order)
    out = np.zeros(tuple(batch_shape) + (joint.dof, self.robot_.dof * max_order), dtype=float)
    diag = np.arange(joint.dof)
    out[..., diag, col_slice.start + diag] = 1.0
    return out

  def _joint_torque_jacobian_parts(self, state, states : list[StateType], max_order : int):
    if not states:
      return []
    fast = self._rust_torque_jacobian(states, max_order, list_output=True)
    if fast is not None:
      return fast
    return outward_api.outward_jacobian(
      self.robot_,
      state,
      states,
      max_time_order=max_order,
      dim=self.dim_,
      list_output=True,
    )

  def _joint_motion_torque_jacobian(self, state, state_type_list, max_order : int, list_output : bool = False):
    if not self._joint_motion_torque_supported(state_type_list):
      return None
    batch_shape = self.batch_shape_
    parts = [None] * len(state_type_list)
    torque_indices = []
    torque_states = []
    for i, st in enumerate(state_type_list):
      if self._is_joint_motion_state(st):
        parts[i] = self._joint_motion_selector_jacobian(st, max_order, batch_shape)
      else:
        torque_indices.append(i)
        torque_states.append(st)
    for i, part in zip(torque_indices, self._joint_torque_jacobian_parts(state, torque_states, max_order)):
      parts[i] = part
    if list_output:
      return parts
    return np.concatenate(parts, axis=-2)

  def _joint_torque_jacobian_apply_parts(self, state, states : list[StateType], max_order : int, rhs, batch_shape : tuple, rhs_is_matrix : bool):
    if not states:
      return []
    fast = self._rust_torque_jacobian_apply(states, max_order, rhs, batch_shape, rhs_is_matrix=rhs_is_matrix, list_output=True)
    if fast is not None:
      return fast
    if rhs_is_matrix:
      direct_rhs = rhs.reshape(batch_shape + rhs.shape[-2:]) if batch_shape else rhs
      return outward_api.outward_jacobian_matmul_rhs(
        self.robot_,
        state,
        states,
        direct_rhs,
        max_time_order=max_order,
        dim=self.dim_,
        list_output=True,
      )
    direct_vec = rhs.reshape(batch_shape + (rhs.shape[-1],)) if batch_shape else rhs
    return outward_api.outward_jacobian_matvec(
      self.robot_,
      state,
      states,
      direct_vec,
      max_time_order=max_order,
      dim=self.dim_,
      list_output=True,
    )

  def _joint_motion_torque_jacobian_apply(self, state, state_type_list, max_order : int, rhs, batch_shape : tuple, rhs_is_matrix : bool, list_output : bool = False):
    if not self._joint_motion_torque_supported(state_type_list):
      return None
    rhs = np.asarray(rhs)
    if rhs_is_matrix:
      rhs_matrix = rhs.reshape(batch_shape + rhs.shape[-2:]) if batch_shape else rhs
    else:
      rhs_vec = rhs.reshape(batch_shape + (rhs.shape[-1],)) if batch_shape else rhs
    parts = [None] * len(state_type_list)
    torque_indices = []
    torque_states = []
    for i, st in enumerate(state_type_list):
      if self._is_joint_motion_state(st):
        col_slice = self._joint_motion_col_slice(st, max_order)
        parts[i] = rhs_matrix[..., col_slice, :] if rhs_is_matrix else rhs_vec[..., col_slice]
      else:
        torque_indices.append(i)
        torque_states.append(st)
    for i, part in zip(torque_indices, self._joint_torque_jacobian_apply_parts(state, torque_states, max_order, rhs, batch_shape, rhs_is_matrix)):
      parts[i] = part
    if list_output:
      return parts
    return np.concatenate(parts, axis=-2 if rhs_is_matrix else -1)

  def _joint_torque_jacobian_transpose_apply_parts(self, state, states : list[StateType], rhs_parts : list[np.ndarray], max_order : int, batch_shape : tuple, rhs_is_matrix : bool):
    if not states:
      return None
    if len(states) == 1:
      fast = self._rust_torque_jacobian_transpose_apply(states, max_order, rhs_parts[0], batch_shape, rhs_is_matrix=rhs_is_matrix)
      if fast is not None:
        return fast

    if rhs_is_matrix:
      rhs_data = [
        part.reshape(batch_shape + part.shape[-2:]) if batch_shape else part
        for part in rhs_parts
      ]
      combined = np.concatenate(rhs_data, axis=-2)
      if not batch_shape:
        packed = outward_api.outward_jacobian_transpose_matvec(
          self.robot_,
          state,
          states,
          np.moveaxis(combined, -1, 0),
          max_time_order=max_order,
          dim=self.dim_,
        )
        return np.moveaxis(packed, 0, -1)
      cols = [
        outward_api.outward_jacobian_transpose_matvec(
          self.robot_,
          state,
          states,
          combined[..., i],
          max_time_order=max_order,
          dim=self.dim_,
        )
        for i in range(combined.shape[-1])
      ]
      return np.stack(cols, axis=-1)

    rhs_data = [
      part.reshape(batch_shape + (part.shape[-1],)) if batch_shape else part
      for part in rhs_parts
    ]
    combined = np.concatenate(rhs_data, axis=-1)
    return outward_api.outward_jacobian_transpose_matvec(
      self.robot_,
      state,
      states,
      combined,
      max_time_order=max_order,
      dim=self.dim_,
    )

  def _joint_motion_torque_jacobian_transpose_apply(self, state, state_type_list, max_order : int, rhs, batch_shape : tuple, rhs_is_matrix : bool):
    if not self._joint_motion_torque_supported(state_type_list):
      return None
    rhs = np.asarray(rhs)
    if rhs_is_matrix:
      rhs_data = rhs.reshape(batch_shape + rhs.shape[-2:]) if batch_shape else rhs
      result = np.zeros(tuple(batch_shape) + (self.robot_.dof * max_order, rhs_data.shape[-1]), dtype=rhs_data.dtype)
    else:
      rhs_data = rhs.reshape(batch_shape + (rhs.shape[-1],)) if batch_shape else rhs
      result = np.zeros(tuple(batch_shape) + (self.robot_.dof * max_order,), dtype=rhs_data.dtype)
    row_start = 0
    torque_states = []
    torque_rhs_parts = []
    for st in state_type_list:
      rows = self._joint_state_dof(st)
      rhs_part = rhs_data[..., row_start:row_start + rows, :] if rhs_is_matrix else rhs_data[..., row_start:row_start + rows]
      if self._is_joint_motion_state(st):
        col_slice = self._joint_motion_col_slice(st, max_order)
        if rhs_is_matrix:
          result[..., col_slice, :] += rhs_part
        else:
          result[..., col_slice] += rhs_part
      else:
        torque_states.append(st)
        torque_rhs_parts.append(rhs_part)
      row_start += rows
    if torque_states:
      result += self._joint_torque_jacobian_transpose_apply_parts(
        state,
        torque_states,
        torque_rhs_parts,
        max_order,
        batch_shape,
        rhs_is_matrix,
      )
    if not rhs_is_matrix:
      return result
    return result
