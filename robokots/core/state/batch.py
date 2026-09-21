"""Backend-independent collection of scalar states with a batch shape."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from math import prod

from .. import batch_shape as batch_shapes


@dataclass(frozen=True)
class StateBatch:
  outward_states: list
  batch_shape: tuple[int, ...]

  def __post_init__(self):
    shape = batch_shapes.validate_batch_shape(self.batch_shape, require_batch=True)
    if len(self.outward_states) != prod(shape):
      raise ValueError(f"StateBatch requires {prod(shape)} states for shape {shape}, got {len(self.outward_states)}")
    object.__setattr__(self, "batch_shape", shape)
    object.__setattr__(self, "outward_states", list(self.outward_states))

  @classmethod
  def from_states(cls, states: Sequence, batch_shape: tuple[int, ...]) -> "StateBatch":
    return cls(outward_states=list(states), batch_shape=batch_shape)
