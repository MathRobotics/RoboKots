from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from . import batch as batch_api


@dataclass(frozen=True)
class StateBatch:
  state_dicts: list
  batch_shape: tuple[int, ...]
  outward_states: list | None = None

  @classmethod
  def from_states(cls, states: Sequence, batch_shape: tuple[int, ...], robot) -> "StateBatch":
    if not batch_shape:
      raise ValueError("StateBatch requires a non-empty batch_shape")
    outward_states = list(states) if states and hasattr(states[0], "to_state_dict") else None
    state_dicts = [
      st.to_state_dict(robot) if hasattr(st, "to_state_dict") else st
      for st in states
    ]
    return cls(state_dicts=state_dicts, batch_shape=batch_shape, outward_states=outward_states)

  def state_info(self, robot, state_type, get_value: Callable):
    values = [get_value(robot, state, state_type) for state in self.state_dicts]
    return batch_api.stack_batch_values(values, self.batch_shape)

  def state_info_list(self, robot, state_type_list, get_value: Callable, list_output: bool = False):
    values = []
    for state in self.state_dicts:
      state_list = [get_value(robot, state, st) for st in state_type_list]
      if list_output:
        values.append(state_list)
      else:
        values.append(np.concatenate([np.asarray(v).reshape(-1) for v in state_list]))

    if list_output:
      return [
        batch_api.stack_batch_values([sample[i] for sample in values], self.batch_shape)
        for i in range(len(state_type_list))
      ]
    return batch_api.stack_sample_results(values, self.batch_shape)
