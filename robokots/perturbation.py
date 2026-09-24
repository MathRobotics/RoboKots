"""Reproducible, non-mutating perturbations of robot model parameters."""
from __future__ import annotations

import copy
import json
import math
import numbers
from dataclasses import dataclass, field, fields
from pathlib import Path
import tomllib
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .kots import Kots
from .core.robot import RobotStruct
from ._perturbation_rules import NoiseSpec, ParameterPerturbation, apply_rules, length_scale


def _validate_std(name: str, value: float) -> None:
  if isinstance(value, bool) or not isinstance(value, numbers.Real):
    raise TypeError(f"{name} must be a finite, non-negative number.")
  if not math.isfinite(value) or value < 0.0:
    raise ValueError(f"{name} must be a finite, non-negative number.")


def _normalise_names(name: str, values: Sequence[str] | None) -> tuple[str, ...] | None:
  if values is None:
    return None
  if isinstance(values, str):
    raise TypeError(f"{name} must be a sequence of link names, not a string.")
  result = tuple(values)
  if any(not isinstance(value, str) or not value for value in result):
    raise ValueError(f"{name} must contain non-empty strings only.")
  if len(set(result)) != len(result):
    raise ValueError(f"{name} must not contain duplicates.")
  return result


@dataclass(frozen=True, kw_only=True)
class PerturbationSpec:
  """Distribution and scope for :func:`apply_perturbation`.

  ``mass_relative_std`` and ``link_length_relative_std`` are standard
  deviations in log space.  Thus their multipliers are always positive and
  are equal to one when the corresponding standard deviation is zero.

  ``link_length_relative_std`` scales outgoing joint distances for rigid
  links and the length field for soft links. CoG and inertia are not scaled
  with length. ``link_translation_std`` offsets the INCOMING joint origin.
  Explicit ``rules`` run afterwards, in order, with their own target scopes.
  """

  seed: int | None = None
  mass_relative_std: float = 0.0
  link_length_relative_std: float = 0.0
  link_translation_std: float = 0.0
  link_cog_translation_std: float = 0.0
  scale_inertia_with_mass: bool = True
  link_names: tuple[str, ...] | None = None
  rules: tuple[ParameterPerturbation, ...] = ()

  @classmethod
  def from_dict(cls, data: dict) -> PerturbationSpec:
    """Load a configuration dictionary, rejecting misspelled/unknown keys.

    Rules are dictionaries with a nested ``noise`` dictionary. Validation
    errors include the offending configuration path. Input is not modified.
    """
    def table(value, model, path):
      if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a table")
      unknown = set(value) - {f.name for f in fields(model) if f.init}
      if unknown:
        raise ValueError(f"{path}: unknown key(s): {', '.join(sorted(map(str, unknown)))}")
      for name in ("names", "link_names", "groups"):
        if name in value and not isinstance(value[name], (list, tuple)):
          raise ValueError(f"{path}.{name}: expected an array")
      if "groups" in value and any(not isinstance(g, (list, tuple)) for g in value["groups"]):
        raise ValueError(f"{path}.groups: expected an array of arrays")
      return copy.deepcopy(value)

    def construct(model, values, path):
      try:
        return model(**values)
      except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc

    values = table(data, cls, "spec")
    raw_rules = values.pop("rules", [])
    if not isinstance(raw_rules, (list, tuple)):
      raise ValueError("spec.rules: expected an array of tables")
    rules = []
    for index, value in enumerate(raw_rules):
      path = f"spec.rules[{index}]"
      rule = table(value, ParameterPerturbation, path)
      noise = table(rule.get("noise"), NoiseSpec, f"{path}.noise")
      rule["noise"] = construct(NoiseSpec, noise, f"{path}.noise")
      rules.append(construct(ParameterPerturbation, rule, path))
    values["rules"] = tuple(rules)
    return construct(cls, values, "spec")

  @classmethod
  def from_toml(cls, text: str) -> PerturbationSpec:
    """Parse TOML text with spec fields at the document root."""
    return cls.from_dict(tomllib.loads(text))

  @classmethod
  def from_toml_file(cls, path: str | Path) -> PerturbationSpec:
    """Read a UTF-8 TOML configuration file (str or pathlib.Path)."""
    with open(path, "rb") as stream:
      return cls.from_dict(tomllib.load(stream))

  def __post_init__(self) -> None:
    if self.seed is not None:
      if isinstance(self.seed, bool) or not isinstance(self.seed, numbers.Integral):
        raise TypeError("seed must be an integer or None.")
      if self.seed < 0:
        raise ValueError("seed must be non-negative or None.")
    for name in (
        "mass_relative_std",
        "link_length_relative_std",
        "link_translation_std",
        "link_cog_translation_std",
    ):
      _validate_std(name, getattr(self, name))
    if not isinstance(self.scale_inertia_with_mass, bool):
      raise TypeError("scale_inertia_with_mass must be a bool.")
    object.__setattr__(self, "link_names", _normalise_names("link_names", self.link_names))
    rules = tuple(self.rules)
    if any(not isinstance(rule, ParameterPerturbation) for rule in rules):
      raise TypeError("rules must contain ParameterPerturbation instances")
    object.__setattr__(self, "rules", rules)


@dataclass(frozen=True)
class PerturbationReport:
  """Realized parameter changes, suitable for recording an experiment."""

  seed: int | None
  mass_scales: Mapping[str, float]
  length_scales: Mapping[str, float]
  cog_translations: Mapping[str, tuple[float, float, float]]
  joint_origin_translations: Mapping[str, tuple[float, float, float]]
  rule_changes: tuple = ()
  model_data: dict = field(default_factory=dict)

  def to_dict(self):
    """JSON-serializable snapshot, including the realized model for replay."""
    return {"seed": int(self.seed) if self.seed is not None else None,
            "mass_scales": dict(self.mass_scales), "length_scales": dict(self.length_scales),
            "cog_translations": dict(self.cog_translations),
            "joint_origin_translations": dict(self.joint_origin_translations),
            "rule_changes": copy.deepcopy(list(self.rule_changes)),
            "model_data": copy.deepcopy(self.model_data)}


def _report_mapping(values: dict[str, float] | dict[str, tuple[float, float, float]]):
  return MappingProxyType(dict(values))


def apply_perturbation(
    nominal: Kots,
    spec: PerturbationSpec,
    *,
    return_report: bool = False,
) -> Kots | tuple[Kots, PerturbationReport]:
  """Create a perturbed copy of ``nominal`` without changing it.

  Masses are multiplied by log-normal samples.  By default their inertia
  tensors receive the same multiplier, preserving a valid inertia tensor and
  the nominal radius of gyration.  CoG and joint-origin translations use
  independent Cartesian Gaussian samples in metres.
  Returns a fresh model with no motions, targets or computed state copied.
  Model parameters are sampled once and remain fixed across all evaluations.
  """
  if not isinstance(nominal, Kots):
    raise TypeError("nominal must be a Kots instance.")
  if not isinstance(spec, PerturbationSpec):
    raise TypeError("spec must be a PerturbationSpec instance.")

  # to_dict() is the canonical model interchange format and creates a clean
  # boundary from any cached state or native workspace held by nominal.
  robot = nominal._python_robot_
  if robot is None:
    source = nominal._model_source_
    data = json.loads(source) if isinstance(source, str) else source
    robot = RobotStruct.from_dict(data)
  model_data = copy.deepcopy(robot.to_dict())
  links_by_name = {link["name"]: link for link in model_data["links"]}
  nominal_masses = {name: link["mass"] for name, link in links_by_name.items()}
  if spec.link_names is None:
    selected_names = tuple(name for name in links_by_name if name != "world")
  else:
    unknown = sorted(set(spec.link_names).difference(links_by_name))
    if unknown:
      raise ValueError("Unknown link name(s): " + ", ".join(unknown))
    selected_names = spec.link_names

  incoming_joint = {
      joint["child_link_id"]: joint
      for joint in model_data["joints"]
  }
  rng = np.random.Generator(np.random.PCG64(spec.seed))
  mass_scales: dict[str, float] = {}
  length_scales: dict[str, float] = {}
  cog_translations: dict[str, tuple[float, float, float]] = {}
  joint_origin_translations: dict[str, tuple[float, float, float]] = {}

  for name in selected_names:
    link = links_by_name[name]
    # Zero-mass links (including the generated URDF world link) are left
    # massless, rather than accidentally becoming physical bodies.
    if link.get("mass", 0.0) > 0.0 and spec.mass_relative_std:
      scale = float(NoiseSpec(distribution="lognormal", mode="scale", std=spec.mass_relative_std).sample(rng, None))
      link["mass"] *= scale
      if spec.scale_inertia_with_mass:
        link["inertia"] = {key: value * scale for key, value in link["inertia"].items()}
      mass_scales[name] = scale

    has_length = (link["length"] > 0 if link["type"] == "soft" else
                  any(j["parent_link_id"] == link["id"] and
                      np.linalg.norm(j["origin"]["position"]) > 0 for j in model_data["joints"]))
    if has_length and spec.link_length_relative_std:
      scale = float(NoiseSpec(distribution="lognormal", mode="scale", std=spec.link_length_relative_std).sample(rng, None))
      length_scale(model_data, link, scale, "scale")
      length_scales[name] = scale

    if spec.link_cog_translation_std:
      delta = rng.normal(scale=spec.link_cog_translation_std, size=3)
      link["cog"] = [float(value + offset) for value, offset in zip(link["cog"], delta)]
      cog_translations[name] = tuple(float(value) for value in delta)

    if spec.link_translation_std:
      joint = incoming_joint.get(link["id"])
      if joint is not None:
        delta = rng.normal(scale=spec.link_translation_std, size=3)
        origin = joint.setdefault("origin", {})
        position = origin.setdefault("position", [0.0, 0.0, 0.0])
        origin["position"] = [float(value + offset) for value, offset in zip(position, delta)]
        joint_origin_translations[joint["name"]] = tuple(float(value) for value in delta)

  changes = apply_rules(model_data, spec.rules, rng, spec.scale_inertia_with_mass)
  for link in model_data["links"]:
    values = [link["mass"], link["length"], *link["cog"], *link["inertia"].values()]
    if not np.all(np.isfinite(values)) or link["mass"] < 0 or link["length"] < 0:
      raise ValueError(f"Non-finite or negative model parameter for {link['name']}")
    if nominal_masses[link["name"]] > 0 and link["mass"] == 0:
      raise ValueError("Mass underflow")
  backend = nominal._input_backend_
  perturbed = Kots.from_json_data(
      model_data,
      order=nominal.order_,
      dim=nominal.dim_,
      lib=nominal.lib_,
      backend=backend,
  )
  if not return_report:
    return perturbed
  report = PerturbationReport(
      seed=spec.seed,
      mass_scales=_report_mapping(mass_scales),
      length_scales=_report_mapping(length_scales),
      cog_translations=_report_mapping(cog_translations),
      joint_origin_translations=_report_mapping(joint_origin_translations),
      rule_changes=tuple(changes),
      model_data=copy.deepcopy(model_data),
  )
  return perturbed, report
