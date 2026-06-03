from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .axis_tensor import AxisTensor


def _batch_axes(batch_shape: tuple[int, ...]) -> tuple[str, ...]:
  return tuple(f"batch{i}" for i in range(len(batch_shape)))


@dataclass(frozen=True)
class StateTensor:
  tensor: AxisTensor
  state_types: tuple[Any, ...]

  @staticmethod
  def from_array(data, state_types = ()) -> "StateTensor":
    arr = np.asarray(data, dtype=float)
    if arr.ndim < 1:
      raise ValueError("state tensor data must have at least one dimension")
    axes = _batch_axes(arr.shape[:-1]) + ("state",)
    return StateTensor(AxisTensor(arr, axes), tuple(state_types))

  @property
  def data(self):
    return self.tensor.data

  @property
  def axes(self):
    return self.tensor.axes

  @property
  def shape(self):
    return self.tensor.shape

  @property
  def batch_shape(self):
    return self.tensor.shape[:-1]

  @property
  def state_dim(self):
    return self.tensor.shape[-1]


@dataclass(frozen=True)
class JacobianTensor:
  tensor: AxisTensor
  state_types: tuple[Any, ...]

  @staticmethod
  def from_array(data, state_types = ()) -> "JacobianTensor":
    arr = np.asarray(data, dtype=float)
    if arr.ndim < 2:
      raise ValueError("jacobian tensor data must have shape (..., state, motion)")
    axes = _batch_axes(arr.shape[:-2]) + ("state", "motion")
    return JacobianTensor(AxisTensor(arr, axes), tuple(state_types))

  @property
  def data(self):
    return self.tensor.data

  @property
  def axes(self):
    return self.tensor.axes

  @property
  def shape(self):
    return self.tensor.shape

  @property
  def batch_shape(self):
    return self.tensor.shape[:-2]

  @property
  def state_dim(self):
    return self.tensor.shape[-2]

  @property
  def motion_dim(self):
    return self.tensor.shape[-1]
