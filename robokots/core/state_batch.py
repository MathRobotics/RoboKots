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
  def from_states(cls, states: Sequence, batch_shape: tuple[int, ...], robot, materialize_dict: bool = True) -> "StateBatch":
    if not batch_shape:
      raise ValueError("StateBatch requires a non-empty batch_shape")
    outward_states = list(states) if states and hasattr(states[0], "to_state_dict") else None
    if materialize_dict or outward_states is None:
      state_dicts = [
        st.to_state_dict(robot) if hasattr(st, "to_state_dict") else st
        for st in states
      ]
    else:
      state_dicts = []
    return cls(state_dicts=state_dicts, batch_shape=batch_shape, outward_states=outward_states)

  def state_info(self, robot, state_type, get_value: Callable):
    states = self._states_for_read()
    values = [get_value(robot, state, state_type) for state in states]
    return batch_api.stack_batch_values(values, self.batch_shape)

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
        batch_api.stack_batch_values([sample[i] for sample in values], self.batch_shape)
        for i in range(len(state_type_list))
      ]
    return batch_api.stack_sample_results(values, self.batch_shape)

  def _states_for_read(self):
    if self.outward_states is not None and self.outward_states and hasattr(self.outward_states[0], "cmtm"):
      return self.outward_states
    return self.state_dicts
