from __future__ import annotations

from typing import Any

from ...core.robot import RobotStruct


def _rust_compiled_robot(robot: RobotStruct):
  try:
    from robokots._rust import RustCompiledRobot
  except ImportError as exc:
    raise ImportError(
      "RoboKots Rust backend is not built. Run "
      "`uvx maturin develop --release --manifest-path robokots/_rust/Cargo.toml` "
      "or install a package that includes the Rust extension."
    ) from exc

  _ensure_supported_robot(robot)
  return RustCompiledRobot.from_model_data(_model_data_from_robot(robot))


def _rust_inverse_dynamics_robot(robot: RobotStruct):
  """Compile the fixed/revolute/prismatic subset used by the RNEA API."""
  try:
    from robokots._rust import RustCompiledRobot
  except ImportError as exc:
    raise ImportError(
      "RoboKots Rust backend is not built. Run "
      "`uvx maturin develop --release --manifest-path robokots/_rust/Cargo.toml` "
      "or install a package that includes the Rust extension."
    ) from exc

  unsupported_links = [link.name for link in robot.links if link.dof != 0 or link.type != "rigid"]
  if unsupported_links:
    raise NotImplementedError(
      "Rust inverse dynamics currently supports rigid links only; unsupported links: "
      + ", ".join(unsupported_links)
    )
  unsupported_joints = [
    joint.name for joint in robot.joints if joint.type not in ("fixed", "revolute", "prismatic")
  ]
  if unsupported_joints:
    raise NotImplementedError(
      "Rust inverse dynamics supports fixed/revolute/prismatic joints only; unsupported joints: "
      + ", ".join(unsupported_joints)
    )
  return RustCompiledRobot.from_model_data(_model_data_from_robot(robot), True)


def _ensure_supported_robot(robot: RobotStruct) -> None:
  unsupported_links = [link.name for link in robot.links if link.dof != 0 or link.type != "rigid"]
  if unsupported_links:
    raise NotImplementedError(
      "Rust backend currently supports rigid links only; unsupported links: "
      + ", ".join(unsupported_links)
    )

  unsupported_joints = [
    joint.name
    for joint in robot.joints
    if joint.type not in ("fixed", "revolute")
  ]
  if unsupported_joints:
    raise NotImplementedError(
      "Rust backend currently supports fixed/revolute joints only. "
      "Use the Python backend for prismatic, spherical, or floating joints; unsupported joints: "
      + ", ".join(unsupported_joints)
    )


def _to_list(value: Any):
  if hasattr(value, "tolist"):
    return value.tolist()
  if isinstance(value, tuple):
    return [_to_list(item) for item in value]
  if isinstance(value, list):
    return [_to_list(item) for item in value]
  if isinstance(value, dict):
    return {key: _to_list(item) for key, item in value.items()}
  return value


def _model_data_from_robot(robot: RobotStruct) -> dict:
  data = robot.to_dict()
  for link in data["links"]:
    link["cog"] = _to_list(link.get("cog", [0.0, 0.0, 0.0]))
    link["inertia"] = _to_list(link.get("inertia"))
  for joint in data["joints"]:
    joint["axis"] = _to_list(joint.get("axis", [0.0, 0.0, 1.0]))
    origin = joint.get("origin", {})
    origin["position"] = _to_list(origin.get("position", [0.0, 0.0, 0.0]))
    origin["orientation"] = _to_list(origin.get("orientation", [1.0, 0.0, 0.0, 0.0]))
    joint["origin"] = origin
  return data
