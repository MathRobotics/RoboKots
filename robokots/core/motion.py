#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.12.13 Created by T.Ishigaki

import numpy as np
import re
import math
from dataclasses import dataclass

from .axis_tensor import AxisTensor


@dataclass(frozen=True)
class MotionLayoutOwner:
  dof: int
  dof_index: int


@dataclass(frozen=True)
class MotionTensor:
  """Axis-aware adapter for motion data in computational (..., dof, order) form."""

  tensor: AxisTensor
  owner_layout: tuple[MotionLayoutOwner, ...]

  @staticmethod
  def from_flat_owner_major(data, owner_layout, order: int) -> "MotionTensor":
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 0:
      raise ValueError("motion data must have at least one dimension")
    owner_layout = RobotMotions._normalize_owner_layout_from_owners(owner_layout)
    dof = sum(owner.dof for owner in owner_layout)
    expected = dof * order
    if arr.shape[-1] != expected:
      raise ValueError(f"flat owner-major motion last dimension must be {expected}, got {arr.shape[-1]}")

    batch_shape = arr.shape[:-1]
    dof_order = np.zeros(batch_shape + (dof, order), dtype=arr.dtype)
    for owner in owner_layout:
      src = RobotMotions.owner_vec_index(owner.dof, owner.dof_index, order)
      block = arr[..., src].reshape(batch_shape + (order, owner.dof))
      dof_order[..., owner.dof_index:owner.dof_index+owner.dof, :] = np.swapaxes(block, -1, -2)

    axes = MotionTensor._batch_axes(batch_shape) + ("dof", "order")
    return MotionTensor(AxisTensor(dof_order, axes), owner_layout)

  @staticmethod
  def from_dof_order(data, owner_layout) -> "MotionTensor":
    arr = np.asarray(data, dtype=float)
    if arr.ndim < 2:
      raise ValueError("dof-order motion must have shape (..., dof, order)")
    owner_layout = RobotMotions._normalize_owner_layout_from_owners(owner_layout)
    dof = sum(owner.dof for owner in owner_layout)
    if arr.shape[-2] != dof:
      raise ValueError(f"dof axis must have length {dof}, got {arr.shape[-2]}")
    axes = MotionTensor._batch_axes(arr.shape[:-2]) + ("dof", "order")
    return MotionTensor(AxisTensor(arr, axes), owner_layout)

  @staticmethod
  def _batch_axes(batch_shape: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(f"batch{i}" for i in range(len(batch_shape)))

  @property
  def batch_shape(self) -> tuple[int, ...]:
    return self.tensor.shape[:-2]

  @property
  def dof(self) -> int:
    return self.tensor.shape[-2]

  @property
  def order(self) -> int:
    return self.tensor.shape[-1]

  def as_dof_order(self, order: int | None = None) -> AxisTensor:
    if order is None:
      order = self.order
    self._validate_order(order)
    axes = self.tensor.axes
    return AxisTensor(self.tensor.data[..., :, :order], axes)

  def as_order_dof(self, order: int | None = None) -> AxisTensor:
    tensor = self.as_dof_order(order)
    return tensor.to_axes(*(tensor.axes[:-2] + ("order", "dof")))

  def owner_block(self, owner, order: int | None = None, cm: bool = False) -> AxisTensor:
    if order is None:
      order = self.order
    self._validate_order(order)
    block = self.tensor.data[..., owner.dof_index:owner.dof_index+owner.dof, :order]
    block = np.swapaxes(block, -1, -2)
    if cm:
      block = self._scale_cm_block(block)
    axes = self.tensor.axes[:-2] + ("order", "owner_dof")
    return AxisTensor(block, axes)

  def to_flat_owner_major(self, order: int | None = None, cm: bool = False) -> AxisTensor:
    if order is None:
      order = self.order
    self._validate_order(order)
    parts = [
      self.owner_block(owner, order=order, cm=cm).data.reshape(self.batch_shape + (owner.dof * order,))
      for owner in self.owner_layout
    ]
    data = np.concatenate(parts, axis=-1) if parts else np.zeros(self.batch_shape + (0,))
    axes = self.tensor.axes[:-2] + ("motion",)
    return AxisTensor(data, axes)

  def derivative(self, tail = None) -> "MotionTensor":
    tail = self._normalize_tail(tail)
    data = np.concatenate([self.tensor.data[..., :, 1:], tail[..., :, None]], axis=-1)
    return MotionTensor(AxisTensor(data, self.tensor.axes), self.owner_layout)

  def cm_scaled(self) -> "MotionTensor":
    order_dof = np.swapaxes(self.tensor.data, -1, -2)
    dof_order = np.swapaxes(self._scale_cm_block(order_dof), -1, -2)
    return MotionTensor(AxisTensor(dof_order, self.tensor.axes), self.owner_layout)

  def _normalize_tail(self, tail):
    if tail is None:
      return np.zeros(self.batch_shape + (self.dof,), dtype=self.tensor.data.dtype)
    tail = np.asarray(tail, dtype=self.tensor.data.dtype)
    if tail.shape == (self.dof,):
      return np.broadcast_to(tail, self.batch_shape + (self.dof,)).copy()
    expected = self.batch_shape + (self.dof,)
    if tail.shape != expected:
      raise ValueError(f"tail must have shape {(self.dof,)} or {expected}, got {tail.shape}")
    return tail

  def _validate_order(self, order: int):
    if order < 1:
      raise ValueError("order must be greater than 0")
    if order > self.order:
      raise ValueError(f"order must be <= motion tensor order ({self.order}), got {order}")

  @staticmethod
  def _scale_cm_block(block: np.ndarray) -> np.ndarray:
    factors = np.array([math.factorial(i) for i in range(block.shape[-2])], dtype=block.dtype)
    return block / factors.reshape((1,) * (block.ndim - 2) + (block.shape[-2], 1))


class RobotMotions:
  """Legacy flat owner-major motion storage.

  This class owns the mutable flat backend layout ``(..., dof * order)``.
  Axis-aware conversions and block reshaping live in ``MotionTensor``.
  """

  ALLOWED_ALIASES = frozenset(["coord", "veloc", "accel"])

  ACCEL_DIFF_PATTERN = re.compile(r"^accel_diff\d+$")

  def __init__(self, robot_dof : int, aliases_ = None, owner_dofs = None, owner_layout = None):
    if aliases_ is None:
      aliases_ = ["coord", "veloc", "accel"]

    self.aliases = self._validate_aliases(aliases_)
    self.dof = robot_dof
    self.owner_layout = self._normalize_owner_layout(robot_dof, owner_dofs, owner_layout)
    self.owner_dofs = [owner.dof for owner in self.owner_layout]
    self.owner_dof_indices = [owner.dof_index for owner in self.owner_layout]
    self.motion_num = len(self.aliases) 
    self.motions = np.zeros(self.dof * self.motion_num)
    self._revision = 0

  def revision(self) -> int:
    return self._revision
  
  def increment_revision(self) -> None:
    self._revision += 1

  def set_aliases(self, aliases_ = ["coord", "veloc", "accel"]):
    self.aliases = self._validate_aliases(aliases_)
    old_motion_num = self.motion_num
    self.motion_num = len(self.aliases)
    batch_shape = self.motions.shape[:-1]
    new_motions = np.zeros(batch_shape + (self.dof * self.motion_num,), dtype=self.motions.dtype)
    copy_order = min(old_motion_num, self.motion_num)
    for owner_dof, owner_dof_index in zip(self.owner_dofs, self.owner_dof_indices):
      old_src = self.owner_vec_index(owner_dof, owner_dof_index, old_motion_num, copy_order)
      new_dst = self.owner_vec_index(owner_dof, owner_dof_index, self.motion_num, copy_order)
      new_motions[..., new_dst] = self.motions[..., old_src]
    self.motions = new_motions
    
  def set_motion(self, vecs):
    motions = np.asarray(vecs, dtype=float)
    if motions.ndim == 0:
      raise ValueError("motions must have at least one dimension")
    expected = self.dof * self.motion_num
    if motions.shape[-1] != expected:
      raise ValueError(f"motions last dimension must be {expected}, got {motions.shape[-1]}")
    self.motions = motions
    
  def motion_index(self, name : str) -> int:
    if name not in self.aliases:
      raise ValueError(f"Invalid alias: {name}")
    for i in range(len(self.aliases)):
      if name == self.aliases[i]:
        return i
  
  def gen_values(self, name : str):
    m_index = self.motion_index(name)
    values = []
    for owner_dof, owner_dof_index in zip(self.owner_dofs, self.owner_dof_indices):
      offset = owner_dof_index * self.motion_num + m_index * owner_dof
      values.append(self.motions[..., offset : offset + owner_dof])
    return np.concatenate(values, axis=-1) if values else np.zeros(self.batch_shape() + (0,))

  def coord(self):
    return self.gen_values("coord")

  def veloc(self):
    return self.gen_values("veloc")
    
  def accel(self):
    return self.gen_values("accel")
  
  def gen_value(self, dof : int, dof_index : int, name : str):
    order = len(self.aliases)
    m_index = self.motion_index(name)
    offset = dof_index * order + m_index * dof
    return self.motions[..., offset : offset + dof]
  
  def joint_coord(self, dof : int, dof_index : int):
    return self.gen_value(dof, dof_index, "coord")
  
  def joint_veloc(self, dof : int, dof_index : int):
    return self.gen_value(dof, dof_index, "veloc")
  
  def joint_accel(self, dof : int, dof_index : int):
    return self.gen_value(dof, dof_index, "accel")

  def link_coord(self, dof : int, dof_index : int):
    return self.gen_value(dof, dof_index, "coord")
  
  def link_veloc(self, dof : int, dof_index : int):
    return self.gen_value(dof, dof_index, "veloc")
  
  def link_accel(self, dof : int, dof_index : int):
    return self.gen_value(dof, dof_index, "accel")

  def joint_motions(self, joint_dof : int, joint_dof_index : int, order = None):
    if order is None:
      order = self.motion_num
    self._validate_order(order)
    return self.owner_block(self._owner(joint_dof, joint_dof_index), order=order)
  
  def joint_motions_cm(self, joint_dof : int, joint_dof_index : int, order = None):
    if order is None:
      order = self.motion_num
    self._validate_order(order)
    return self.owner_block(self._owner(joint_dof, joint_dof_index), order=order, cm=True)

  def link_motions(self, link_dof : int, link_dof_index : int, order = None):
    if order is None:
      order = self.motion_num
    self._validate_order(order)
    return self.owner_block(self._owner(link_dof, link_dof_index), order=order)
  
  def link_motions_cm(self, link_dof : int, link_dof_index : int, order = None):
    if order is None:
      order = self.motion_num
    self._validate_order(order)
    return self.owner_block(self._owner(link_dof, link_dof_index), order=order, cm=True)

  def motion_tensor(self) -> MotionTensor:
    return MotionTensor.from_flat_owner_major(self.motions, self.owner_layout, self.motion_num)

  def owner_block(self, owner, order : int = None, cm : bool = False):
    if order is None:
      order = self.motion_num
    self._validate_order(order)
    block = self._owner_block_flat(owner, order)
    if cm:
      block = MotionTensor._scale_cm_block(block)
    return block

  def owner_derivative_block(self, owner, order : int = None, tail = None, cm : bool = False):
    if order is None:
      order = self.motion_num
    self._validate_order(order)
    return self.motion_tensor().derivative(tail).owner_block(owner, order=order, cm=cm).data

  def to_dof_order(self, order : int = None):
    if order is None:
      order = self.motion_num
    self._validate_order(order)
    return self.motion_tensor().as_dof_order(order).data

  def set_dof_order(self, data):
    tensor = MotionTensor.from_dof_order(data, self.owner_layout)
    if tensor.order != self.motion_num:
      raise ValueError(f"dof-order motion order must be {self.motion_num}, got {tensor.order}")
    self.motions = tensor.to_flat_owner_major(self.motion_num).data

  def to_vector(self, order : int = None, cm : bool = False):
    if order is None:
      order = self.motion_num
    self._validate_order(order)
    if not cm and order == self.motion_num:
      return self.motions.copy()
    return self._owner_blocks_to_vector(
      [self.owner_block(owner, order=order, cm=cm) for owner in self.owner_layout],
      order,
    )

  def to_derivative_vector(self, order : int = None, tail = None, cm : bool = False):
    if order is None:
      order = self.motion_num
    self._validate_order(order)
    tail = self._normalize_tail(tail)
    blocks = []
    for owner in self.owner_layout:
      block = self._owner_block_flat(owner, self.motion_num)
      owner_tail = tail[..., owner.dof_index:owner.dof_index+owner.dof][..., None, :]
      derivative_block = np.concatenate([block[..., 1:, :], owner_tail], axis=-2)[..., :order, :]
      if cm:
        derivative_block = MotionTensor._scale_cm_block(derivative_block)
      blocks.append(derivative_block)
    return self._owner_blocks_to_vector(blocks, order)

  def batch_shape(self):
    return self.motions.shape[:-1]

  def is_batched(self):
    return len(self.batch_shape()) > 0

  def _validate_order(self, order : int):
    if order < 1:
      raise ValueError("order must be greater than 0")
    if order > self.motion_num:
      raise ValueError(f"order must be <= motion_num ({self.motion_num}), got {order}")

  @classmethod
  def _validate_aliases(cls, aliases):
    aliases = list(aliases)
    invalid = {a for a in aliases if a not in cls.ALLOWED_ALIASES and not cls.ACCEL_DIFF_PATTERN.match(a)}
    if invalid:
      raise ValueError(f"Invalid alias: {invalid}")
    return aliases

  @staticmethod
  def _owner(dof : int, dof_index : int) -> MotionLayoutOwner:
    return MotionLayoutOwner(int(dof), int(dof_index))

  def _owner_block_flat(self, owner, order : int):
    owner = self._owner(owner.dof, owner.dof_index)
    src = self.owner_vec_index(owner.dof, owner.dof_index, self.motion_num, order)
    return self.motions[..., src].reshape(self.batch_shape() + (order, owner.dof))

  def _owner_blocks_to_vector(self, blocks, order : int):
    parts = [
      block.reshape(self.batch_shape() + (owner.dof * order,))
      for block, owner in zip(blocks, self.owner_layout)
    ]
    return np.concatenate(parts, axis=-1) if parts else np.zeros(self.batch_shape() + (0,))

  def _normalize_tail(self, tail):
    if tail is None:
      return np.zeros(self.batch_shape() + (self.dof,), dtype=self.motions.dtype)
    tail = np.asarray(tail, dtype=self.motions.dtype)
    if tail.shape == (self.dof,):
      return np.broadcast_to(tail, self.batch_shape() + (self.dof,)).copy()
    expected = self.batch_shape() + (self.dof,)
    if tail.shape != expected:
      raise ValueError(f"tail must have shape {(self.dof,)} or {expected}, got {tail.shape}")
    return tail

  @staticmethod
  def _normalize_owner_layout(robot_dof : int, owner_dofs, owner_layout):
    if owner_layout is not None:
      layout = tuple(owner_layout)
      normalized = tuple(
        MotionLayoutOwner(dof=int(owner.dof), dof_index=int(owner.dof_index))
        for owner in layout
      )
    else:
      if owner_dofs is None:
        owner_dofs = [1] * robot_dof
      normalized_list = []
      offset = 0
      for owner_dof in owner_dofs:
        owner = MotionLayoutOwner(dof=int(owner_dof), dof_index=int(offset))
        normalized_list.append(owner)
        offset += int(owner_dof)
      normalized = tuple(normalized_list)

    if sum(owner.dof for owner in normalized) != robot_dof:
      raise ValueError(f"motion owner dofs must sum to robot_dof ({robot_dof}), got {sum(owner.dof for owner in normalized)}")

    expected = 0
    for owner in sorted(normalized, key=lambda item: item.dof_index):
      if owner.dof_index != expected:
        raise ValueError("motion owner layout must be contiguous and sorted by dof_index")
      expected += owner.dof

    return tuple(sorted(normalized, key=lambda item: item.dof_index))

  @staticmethod
  def _normalize_owner_layout_from_owners(owner_layout):
    normalized = tuple(
      MotionLayoutOwner(dof=int(owner.dof), dof_index=int(owner.dof_index))
      for owner in owner_layout
    )
    expected = 0
    for owner in sorted(normalized, key=lambda item: item.dof_index):
      if owner.dof_index != expected:
        raise ValueError("motion owner layout must be contiguous and sorted by dof_index")
      expected += owner.dof
    return tuple(sorted(normalized, key=lambda item: item.dof_index))
  
  @staticmethod
  def owner_vec_index(owner_dof, owner_dof_index, vec_order, out_put_order = None):
    if out_put_order is None:
      out_put_order = vec_order
    return slice(owner_dof_index * vec_order, owner_dof_index * vec_order + owner_dof * out_put_order)
