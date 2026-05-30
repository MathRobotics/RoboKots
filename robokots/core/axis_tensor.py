from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PhysicalLayout:
  backend: str = "numpy"
  contiguous: bool = False
  memory_order: str | None = None


@dataclass(frozen=True)
class LayoutPolicy:
  axes: tuple[str, ...] | None = None
  contiguous: bool = False
  backend: str | None = None
  memory_order: str | None = "C"


@dataclass(frozen=True)
class AlgorithmSpec:
  required_axes: tuple[str, ...]
  layout_policy: LayoutPolicy | None = None


@dataclass(frozen=True)
class AxisTensor:
  data: Any
  axes: tuple[str, ...]
  layout: PhysicalLayout = PhysicalLayout()

  def __post_init__(self):
    data = np.asarray(self.data)
    axes = tuple(self.axes)
    if data.ndim != len(axes):
      raise ValueError(f"axes length ({len(axes)}) must match data.ndim ({data.ndim})")
    if len(set(axes)) != len(axes):
      raise ValueError(f"axes must be unique, got {axes}")
    object.__setattr__(self, "data", data)
    object.__setattr__(self, "axes", axes)
    if self.layout.contiguous != self.is_contiguous():
      object.__setattr__(
        self,
        "layout",
        PhysicalLayout(
          backend=self.layout.backend,
          contiguous=self.is_contiguous(),
          memory_order=self.layout.memory_order,
        ),
      )

  @property
  def shape(self) -> tuple[int, ...]:
    return self.data.shape

  def axis_index(self, axis: str) -> int:
    try:
      return self.axes.index(axis)
    except ValueError as e:
      raise ValueError(f"axis {axis!r} is not present in {self.axes}") from e

  def to_axes(self, *axes: str) -> "AxisTensor":
    axes = tuple(axes)
    if axes == self.axes:
      return self
    if set(axes) != set(self.axes) or len(axes) != len(self.axes):
      raise ValueError(f"axes must be a permutation of {self.axes}, got {axes}")
    perm = tuple(self.axis_index(axis) for axis in axes)
    return AxisTensor(
      np.transpose(self.data, perm),
      axes,
      PhysicalLayout(
        backend=self.layout.backend,
        contiguous=False,
        memory_order=None,
      ),
    )

  def materialize(self, memory_order: str | None = None) -> "AxisTensor":
    order = memory_order or self.layout.memory_order or "C"
    if order == "F":
      data = np.asfortranarray(self.data)
    elif order == "C":
      data = np.ascontiguousarray(self.data)
    else:
      data = np.array(self.data, copy=True)
    return AxisTensor(
      data,
      self.axes,
      PhysicalLayout(
        backend=self.layout.backend,
        contiguous=True,
        memory_order=order,
      ),
    )

  def to_layout(self, policy: LayoutPolicy) -> "AxisTensor":
    tensor = self
    if policy.axes is not None:
      tensor = tensor.to_axes(*policy.axes)
    backend = policy.backend or tensor.layout.backend
    if policy.contiguous:
      tensor = tensor.materialize(policy.memory_order)
    if tensor.layout.backend == backend:
      return tensor
    return AxisTensor(
      tensor.data,
      tensor.axes,
      PhysicalLayout(
        backend=backend,
        contiguous=tensor.layout.contiguous,
        memory_order=tensor.layout.memory_order,
      ),
    )

  def prepare_for(self, spec: AlgorithmSpec) -> "AxisTensor":
    policy = spec.layout_policy or LayoutPolicy(axes=spec.required_axes)
    if policy.axes is None:
      policy = LayoutPolicy(
        axes=spec.required_axes,
        contiguous=policy.contiguous,
        backend=policy.backend,
        memory_order=policy.memory_order,
      )
    return self.to_layout(policy)

  def is_contiguous(self) -> bool:
    return bool(self.data.flags.c_contiguous or self.data.flags.f_contiguous)
