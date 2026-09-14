from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from . import batch_shape as batch_shapes


@dataclass(frozen=True)
class StateBatch:
  outward_states: list
  batch_shape: tuple[int, ...]

  @classmethod
  def from_states(cls, states: Sequence, batch_shape: tuple[int, ...]) -> "StateBatch":
    if not batch_shape:
      raise ValueError("StateBatch requires a non-empty batch_shape")
    return cls(outward_states=list(states), batch_shape=batch_shape)

  def state_info(self, robot, state_type, get_value: Callable):
    states = self._states_for_read()
    values = [get_value(robot, state, state_type) for state in states]
    return batch_shapes.stack_batch_values(values, self.batch_shape)

  def state_info_list(self, robot, state_type_list, get_value: Callable, list_output: bool = False):
    states = self._states_for_read()
    values = []
    for state in states:
      state_list = [get_value(robot, state, st) for st in state_type_list]
      if list_output:
        values.append(state_list)
      else:
        values.append(np.concatenate([np.asarray(v).reshape(-1) for v in state_list]))

    if list_output:
      return [
        batch_shapes.stack_batch_values([sample[i] for sample in values], self.batch_shape)
        for i in range(len(state_type_list))
      ]
    return batch_shapes.stack_sample_results(values, self.batch_shape)

  def _states_for_read(self):
    return self.outward_states
