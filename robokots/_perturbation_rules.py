"""Distribution specifications and model-dictionary perturbation operations."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numbers

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True, kw_only=True)
class NoiseSpec:
  """Explicit distribution of an additive offset or multiplicative factor.

  Normal: ``mean + std * N(0, 1)``. Uniform: ``U(low, high)``.
  Lognormal: ``exp(mean + std * N(0, 1))`` (mean/std in log space).
  Loguniform: ``exp(U(log(low), log(high)))`` (bounds are factors).
  Scale normals default to mean 1; all other normal means default to 0.
  Invalid sampled positive parameters raise; samples are never clipped.
  """
  distribution: str = "normal"
  mode: str = "additive"
  mean: float | None = None
  std: float | None = None
  low: float | None = None
  high: float | None = None

  def __post_init__(self):
    if self.distribution not in {"normal", "uniform", "lognormal", "loguniform"}:
      raise ValueError("Unknown noise distribution")
    if self.mode not in {"additive", "scale"}:
      raise ValueError("mode must be additive or scale")
    if self.distribution.startswith("log") and self.mode != "scale":
      raise ValueError("Log distributions require mode='scale'")
    for name in ("mean", "std", "low", "high"):
      value = getattr(self, name)
      if value is not None and (isinstance(value, bool) or not isinstance(value, numbers.Real)
                                or not np.isfinite(value)):
        raise ValueError(f"{name} must be finite and real")
    if self.distribution in {"normal", "lognormal"}:
      if self.std is None or self.std < 0 or self.low is not None or self.high is not None:
        raise ValueError("Normal distributions require std >= 0 and no bounds")
      if self.mean is None:
        object.__setattr__(self, "mean", 1.0 if self.mode == "scale" and self.distribution == "normal" else 0.0)
    else:
      if self.low is None or self.high is None or self.low > self.high or self.mean is not None or self.std is not None:
        raise ValueError("Uniform distributions require low <= high and no mean/std")
      if self.mode == "scale" and self.low <= 0:
        raise ValueError("Scale bounds must be positive")

  def sample(self, rng, size):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
      if self.distribution == "normal":
        value = rng.normal(self.mean, self.std, size)
      elif self.distribution == "lognormal":
        value = np.exp(rng.normal(self.mean, self.std, size))
      elif self.distribution == "uniform":
        value = rng.uniform(self.low, self.high, size)
      else:
        value = np.exp(rng.uniform(math.log(self.low), math.log(self.high), size))
    if not np.all(np.isfinite(value)) or (self.mode == "scale" and np.any(value <= 0)):
      raise ValueError("Noise sample must be finite (and positive for scale); reduce the noise range")
    return value


LINK_PARAMETERS = {"mass", "link_length", "cog_translation", "inertia"}
JOINT_PARAMETERS = {"joint_translation", "joint_rotation", "joint_offset"}


@dataclass(frozen=True, kw_only=True)
class ParameterPerturbation:
  """One operation on named links or joints, optionally sharing draws.

  ``groups`` contains disjoint tuples of target names. Each group shares one
  draw; remaining targets draw independently. ``names=None`` selects eligible
  owners except world and joints directly attached to world.
  """
  parameter: str
  noise: NoiseSpec
  names: tuple[str, ...] | None = None
  groups: tuple[tuple[str, ...], ...] = ()

  def __post_init__(self):
    if self.parameter not in LINK_PARAMETERS | JOINT_PARAMETERS:
      raise ValueError(f"Unsupported parameter: {self.parameter}")
    if not isinstance(self.noise, NoiseSpec):
      raise TypeError("noise must be a NoiseSpec")
    required = "scale" if self.parameter == "inertia" else "additive"
    if self.parameter not in {"mass", "link_length"} and self.noise.mode != required:
      raise ValueError(f"{self.parameter} requires mode={required!r}")
    def names_tuple(values):
      if isinstance(values, str):
        raise TypeError("names/groups require sequences, not strings")
      values = tuple(values)
      if any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
        raise ValueError("Names must be nonempty, unique strings")
      return values
    if self.names is not None:
      object.__setattr__(self, "names", names_tuple(self.names))
    groups = tuple(names_tuple(group) for group in self.groups)
    flat = [name for group in groups for name in group]
    if any(not group for group in groups) or len(flat) != len(set(flat)):
      raise ValueError("Groups must be nonempty and disjoint")
    if self.names is not None and not set(flat) <= set(self.names):
      raise ValueError("Group members must belong to names")
    object.__setattr__(self, "groups", groups)


def _positive(value, label):
  if not np.isfinite(value) or value <= 0:
    raise ValueError(f"{label} must remain finite and positive")
  return float(value)


def length_scale(data, link, sample, mode):
  """Rigid lengths belong to the PARENT link of outgoing joint origins."""
  if link["type"] == "soft":
    old = link["length"]
    _positive(old, "Nominal soft length")
    new = _positive(old * sample if mode == "scale" else old + sample, "Length")
    link["length"] = new
    return new / old
  joints = [j for j in data["joints"] if j["parent_link_id"] == link["id"]
            and np.linalg.norm(j["origin"]["position"]) > 0]
  if not joints:
    raise ValueError(f"Link {link['name']!r} has no nonzero outgoing joint distance; specify a joint translation instead")
  for joint in joints:
    p = np.asarray(joint["origin"]["position"], dtype=float)
    old = np.linalg.norm(p)
    new = _positive(old * sample if mode == "scale" else old + sample, "Joint distance")
    joint["origin"]["position"] = (p * (new / old)).tolist()
  # For additive changes a branched link can have different distance ratios.
  return float(sample) if mode == "scale" else None


def _inertia(link, factors):
  d = link["inertia"]
  inertia = np.array([[d["ixx"], d["ixy"], d["ixz"]],
                      [d["ixy"], d["iyy"], d["iyz"]],
                      [d["ixz"], d["iyz"], d["izz"]]])
  _positive(link["mass"], "Mass for inertia perturbation")
  # A positive second-moment matrix enforces inertia triangle inequalities.
  second = np.trace(inertia) / 2 * np.eye(3) - inertia
  eigenvalues, axes = np.linalg.eigh(second)
  tolerance = 1e-12 * max(np.linalg.norm(inertia), np.finfo(float).tiny)
  if eigenvalues.min() < -tolerance:
    raise ValueError(f"Link {link['name']!r} has physically inconsistent inertia")
  second = (axes * (np.maximum(eigenvalues, 0) * factors)) @ axes.T
  inertia = np.trace(second) * np.eye(3) - second
  link["inertia"] = {key: float(inertia[i, j]) for key, i, j in
                     [("ixx", 0, 0), ("ixy", 0, 1), ("ixz", 0, 2),
                      ("iyy", 1, 1), ("iyz", 1, 2), ("izz", 2, 2)]}


def apply_rules(data, rules, rng, scale_inertia_with_mass):
  records = []
  world_ids = {l["id"] for l in data["links"] if l["name"] == "world"}
  for rule in rules:
    parameter = rule.parameter
    owners = data["links"] if parameter in LINK_PARAMETERS else data["joints"]
    lookup = {owner["name"]: owner for owner in owners}
    def eligible(owner):
      if parameter in LINK_PARAMETERS:
        if owner["name"] == "world":
          return False
        if parameter in {"mass", "inertia"}:
          return owner["mass"] > 0
        if parameter == "link_length":
          return (owner["length"] > 0 if owner["type"] == "soft" else
                  any(j["parent_link_id"] == owner["id"] and
                      np.linalg.norm(j["origin"]["position"]) > 0 for j in data["joints"]))
        return True
      return owner["parent_link_id"] not in world_ids and (
          parameter != "joint_offset" or owner["type"] in {"revolute", "prismatic"})
    names = rule.names if rule.names is not None else tuple(o["name"] for o in owners if eligible(o))
    if set(names) - lookup.keys():
      raise ValueError(f"Unknown {parameter} targets: {sorted(set(names) - lookup.keys())}")
    if not {n for group in rule.groups for n in group} <= set(names):
      raise ValueError("Group contains unknown or ineligible targets")
    group_ids = {name: i for i, group in enumerate(rule.groups) for name in group}
    draws = {}
    samples = {}
    for name in names:
      owner = lookup[name]
      key = ("group", group_ids[name]) if name in group_ids else ("owner", name)
      size = 3 if parameter in {"cog_translation", "joint_translation", "joint_rotation", "inertia"} else None
      if key not in draws:
        draws[key] = rule.noise.sample(rng, size)
      sample = draws[key]
      samples[name] = np.asarray(sample).tolist()
      if parameter == "mass":
        old = _positive(owner["mass"], "Nominal mass")
        new = _positive(old * sample if rule.noise.mode == "scale" else old + sample, "Mass")
        owner["mass"] = new
        if scale_inertia_with_mass:
          owner["inertia"] = {k: v * new / old for k, v in owner["inertia"].items()}
      elif parameter == "link_length":
        length_scale(data, owner, sample, rule.noise.mode)
      elif parameter == "inertia":
        _inertia(owner, sample)
      elif parameter == "cog_translation":
        owner["cog"] = (np.asarray(owner["cog"]) + sample).tolist()
      else:
        origin = owner["origin"]
        rotation = Rotation.from_quat(np.roll(origin["orientation"], -1))
        if parameter == "joint_translation":
          origin["position"] = (np.asarray(origin["position"]) + sample).tolist()
        elif parameter == "joint_rotation":
          # Right composition: rotation vector in nominal joint frame.
          origin["orientation"] = np.roll((rotation * Rotation.from_rotvec(sample)).as_quat(), 1).tolist()
        else:
          if owner["type"] not in {"revolute", "prismatic"}:
            raise ValueError("joint_offset supports revolute/prismatic joints only")
          axis = np.asarray(owner["axis"], dtype=float)
          axis /= np.linalg.norm(axis)
          if owner["type"] == "revolute":
            origin["orientation"] = np.roll((rotation * Rotation.from_rotvec(axis * sample)).as_quat(), 1).tolist()
          else:
            origin["position"] = (np.asarray(origin["position"]) + rotation.apply(axis * sample)).tolist()
    records.append({"parameter": parameter, "mode": rule.noise.mode, "samples": samples})
  return records
