"""Model input and lazy Python representation for a native-first Kots instance."""
from __future__ import annotations

import copy
import json

from ..core.robot import RobotModelInfo, RobotStruct


class ModelInputMixin:
  def _init_model(self, robot):
    self._python_robot_ = None if isinstance(robot, RobotModelInfo) else robot
    self._native_model_info_ = robot if isinstance(robot, RobotModelInfo) else None
    self._model_source_ = None
    self._input_backend_ = None
    self._computed_backend_ = None

  @property
  def robot_(self):
    """Full Python model; materialized on first explicit/model-based use."""
    if self._python_robot_ is None:
      source = self._model_source_
      if source is None:
        raise RuntimeError("No source is available for the Python model")
      data = json.loads(source) if isinstance(source, str) else source
      self._python_robot_ = RobotStruct.from_dict(data, lib=self.lib_)
    return self._python_robot_

  @robot_.setter
  def robot_(self, robot):
    self._python_robot_ = robot

  @property
  def _model_metadata(self):
    # Preserve the existing Python-model semantics for ordinary instances.
    return self._native_model_info_ if self._input_backend_ == "rust" else self.robot_

  @classmethod
  def _from_rust_input(cls, source, order, dim, lib):
    if dim != 3 or lib != "numpy":
      raise ValueError("backend='rust' model input requires dim=3 and lib='numpy'")
    from .._rust import RustCompiledRobot
    # Keep a private snapshot for a later explicit NumPy/JAX/model operation.
    # Dict input crosses PyO3 directly; it is not serialized to JSON.
    source = source if isinstance(source, str) else copy.deepcopy(source)
    compiled = (RustCompiledRobot.from_json(source) if isinstance(source, str)
                else RustCompiledRobot.from_input_data(source))
    info = RobotModelInfo.from_native(compiled)
    result = cls(info, order, dim, lib)
    result._model_source_ = source
    result._input_backend_ = "rust"
    result._rust_compiled_robot_ = compiled
    result._rust_inverse_dynamics_robot_ = compiled
    result._rust_model_info_cache_ = (compiled, info)
    return result

  @staticmethod
  def _check_input_backend(backend):
    if backend not in (None, "numpy", "rust"):
      raise ValueError("model input backend must be None, 'numpy', or 'rust'")

  def _resolve_dynamics_backend(self, backend=None):
    """Select on-demand calculations without silently crossing a NumPy boundary."""
    if backend is None:
      backend = self._computed_backend_ or self._input_backend_ or "rust"
    if backend not in ("numpy", "rust", "reference"):
      raise ValueError("backend must be 'numpy', 'rust', or 'reference'")
    return "numpy" if backend == "reference" else backend
