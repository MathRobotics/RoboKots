import numpy as np
import pytest

from robokots.kots import Kots
from robokots.perturbation import PerturbationSpec, apply_perturbation
from robokots import NoiseSpec, ParameterPerturbation
from robokots.kots import StateType


def rule(parameter, **kwargs):
  names = kwargs.pop("names", None)
  groups = kwargs.pop("groups", ())
  return ParameterPerturbation(parameter=parameter, names=names, groups=groups,
                               noise=NoiseSpec(**kwargs))


def rigid_chain(backend="numpy", joint_type="revolute"):
  data = make_model()
  data["links"][1]["inertia"]["izz"] = 2.5
  data["links"][2]["type"] = "rigid"
  data["joints"][0].update(type=joint_type, axis=[0., 0., 1.])
  data["joints"][1]["origin"]["position"] = [1., 0., 0.]
  return Kots.from_json_data(data, order=4, backend=backend)


def make_model():
  return {
      "schema_version": "0.0.2",
      "links": [
          {"id": 0, "name": "world", "mass": 0.0},
          {"id": 1, "name": "body", "mass": 2.0, "cog": [0.1, 0.0, 0.0],
           "inertia": {"ixx": 1.0, "ixy": 0.1, "ixz": 0.0, "iyy": 2.0, "iyz": 0.0, "izz": 3.0}},
          {"id": 2, "name": "rod", "type": "soft", "mass": 3.0, "length": 1.5,
           "inertia": {"ixx": 1.0, "ixy": 0.0, "ixz": 0.0, "iyy": 1.0, "iyz": 0.0, "izz": 1.0}},
      ],
      "joints": [
          {"id": 0, "name": "world_body", "type": "fixed", "parent_link_id": 0, "child_link_id": 1,
           "origin": {"position": [0.0, 0.0, 0.0], "orientation": [1.0, 0.0, 0.0, 0.0]}},
          {"id": 1, "name": "body_rod", "type": "fixed", "parent_link_id": 1, "child_link_id": 2,
           "origin": {"position": [0.0, 0.0, 0.0], "orientation": [1.0, 0.0, 0.0, 0.0]}},
      ],
  }


def test_perturbation_is_reproducible_non_mutating_and_preserves_model_options():
  nominal = Kots.from_json_data(make_model(), order=4, backend="numpy")
  original = nominal.robot_.to_dict()
  spec = PerturbationSpec(
      seed=42,
      mass_relative_std=0.05,
      link_length_relative_std=0.1,
      link_translation_std=0.002,
      link_cog_translation_std=0.003,
  )
  actual, report = apply_perturbation(nominal, spec, return_report=True)
  repeated = apply_perturbation(nominal, spec)

  assert nominal.robot_.to_dict() == original
  assert actual.robot_.to_dict() == repeated.robot_.to_dict()
  assert actual.order() == 4
  assert actual._input_backend_ == "numpy"
  assert set(report.mass_scales) == {"body", "rod"}
  assert set(report.length_scales) == {"rod"}
  assert set(report.joint_origin_translations) == {"world_body", "body_rod"}
  with pytest.raises(TypeError):
    report.mass_scales["body"] = 1.0

  before = original["links"][1]
  after = next(link for link in actual.robot_.to_dict()["links"] if link["name"] == "body")
  scale = report.mass_scales["body"]
  assert after["mass"] == pytest.approx(before["mass"] * scale)
  assert after["inertia"]["iyy"] == pytest.approx(before["inertia"]["iyy"] * scale)


def test_scope_and_zero_mass_links_are_handled_safely():
  nominal = Kots.from_json_data(make_model())
  actual, report = apply_perturbation(
      nominal,
      PerturbationSpec(seed=4, mass_relative_std=0.2, link_names=("world", "body")),
      return_report=True,
  )
  assert set(report.mass_scales) == {"body"}
  assert actual.robot_.link("world").mass == 0.0
  assert actual.robot_.link("rod").mass == nominal.robot_.link("rod").mass


@pytest.mark.parametrize("kwargs", [
    {"mass_relative_std": -0.1},
    {"link_translation_std": float("nan")},
    {"seed": -1},
])
def test_invalid_specs_are_rejected(kwargs):
  with pytest.raises((TypeError, ValueError)):
    PerturbationSpec(**kwargs)


def test_unknown_scope_is_rejected():
  nominal = Kots.from_json_data(make_model())
  with pytest.raises(ValueError, match="Unknown link"):
    apply_perturbation(nominal, PerturbationSpec(link_names=("missing",)))


def test_perturbation_api_is_available_from_package():
  from robokots import PerturbationSpec as ExportedSpec, apply_perturbation as exported_apply
  assert ExportedSpec is PerturbationSpec
  assert exported_apply is apply_perturbation


def test_rust_model_input_is_recompiled_from_the_perturbed_model():
  rigid_model = make_model()
  rigid_model["links"] = rigid_model["links"][:2]
  rigid_model["joints"] = rigid_model["joints"][:1]
  nominal = Kots.from_json_data(rigid_model, backend="rust")
  actual = apply_perturbation(nominal, PerturbationSpec(seed=9, mass_relative_std=0.1))
  assert actual._input_backend_ == "rust"
  assert actual.robot_.link("body").mass != nominal.robot_.link("body").mass


def test_rigid_length_changes_forward_kinematics_and_jacobian():
  nominal = rigid_chain()
  spec = PerturbationSpec(rules=(rule("link_length", mode="scale", std=0, mean=1.2, names=("body",)),))
  actual = apply_perturbation(nominal, spec)
  motion = np.array([[[.3, .2, .1, .0]], [[-.4, .1, .0, .2]]])
  state = StateType("link", "rod", "pos", "world")
  for k in (nominal, actual):
    k.import_motion_array(motion)
    k.kinematics(backend="numpy")
  np.testing.assert_allclose(actual.state_info(state), 1.2 * nominal.state_info(state), atol=1e-12)
  np.testing.assert_allclose(actual.jacobian(state), 1.2 * nominal.jacobian(state), atol=1e-12)
  np.testing.assert_allclose(actual.robot_.link("body").cog, nominal.robot_.link("body").cog)
  assert actual.robot_.link("body").mass == nominal.robot_.link("body").mass


@pytest.mark.parametrize("kind,offset", [("revolute", .13), ("prismatic", .024)])
@pytest.mark.parametrize("backend", ["numpy", "rust"])
def test_zero_offset_matches_shifted_coordinate_dynamics(kind, offset, backend):
  nominal = rigid_chain(backend, kind)
  # Non-identity origin exercises local versus parent-frame composition.
  data = nominal.robot_.to_dict()
  data["joints"][0]["origin"]["orientation"] = [np.sqrt(.5), np.sqrt(.5), 0., 0.]
  nominal = Kots.from_json_data(data, backend=backend)
  actual = apply_perturbation(nominal, PerturbationSpec(rules=(
      rule("joint_offset", std=0, mean=offset, names=("world_body",)),)))
  q, v, a = np.array([.4]), np.array([.3]), np.array([.2])
  gravity = [.3, -.4, -9.81]
  np.testing.assert_allclose(actual.inverse_dynamics(q, v, a, gravity=gravity),
                             nominal.inverse_dynamics(q + offset, v, a, gravity=gravity), atol=1e-11)
  for model, coordinate in ((actual, q), (nominal, q + offset)):
    model.import_motion_array(np.stack([coordinate, v, a], axis=-1))
    model.kinematics(backend="numpy")
  state = StateType("link", "rod", "pos", "world")
  np.testing.assert_allclose(actual.state_info(state), nominal.state_info(state), atol=1e-12)


def test_groups_branching_and_json_replay():
  import json
  data = rigid_chain().robot_.to_dict()
  data["links"].append(dict(data["links"][2], id=3, name="branch"))
  data["joints"].append(dict(data["joints"][1], id=2, name="branch_joint", child_link_id=3,
                             origin={"position": [0., 2., 0.], "orientation": [1., 0., 0., 0.]}))
  nominal = Kots.from_json_data(data)
  actual, report = apply_perturbation(nominal, PerturbationSpec(seed=42, rules=(
      rule("mass", distribution="uniform", mode="scale", low=.8, high=1.2,
           names=("rod", "branch"), groups=(("rod", "branch"),)),
      rule("link_length", std=0, mean=.1, names=("body",)),
  )), return_report=True)
  assert actual.robot_.link("rod").mass == actual.robot_.link("branch").mass
  np.testing.assert_allclose(actual.robot_.joint("body_rod").origin.pos(), [1.1, 0, 0])
  np.testing.assert_allclose(actual.robot_.joint("branch_joint").origin.pos(), [0, 2.1, 0])
  snapshot = json.loads(json.dumps(report.to_dict()))
  replay = Kots.from_json_data(snapshot["model_data"])
  assert replay.robot_.to_dict() == actual.robot_.to_dict()
  assert nominal.robot_.to_dict() == data


def test_inertia_is_realizable_after_large_anisotropic_samples():
  from robokots._perturbation_rules import _inertia
  nominal = rigid_chain()
  spec = PerturbationSpec(seed=8, rules=(rule("inertia", distribution="loguniform", mode="scale", low=.01, high=100),))
  actual = apply_perturbation(nominal, spec)
  for link in actual.robot_.links[1:]:
    xx, yy, zz, xy, xz, yz = link.inertia
    eigenvalues = np.linalg.eigvalsh([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    assert eigenvalues.min() > 0
    assert eigenvalues[-1] <= eigenvalues[:2].sum() + 1e-10
  bad = make_model()["links"][1]
  bad["inertia"].update(ixx=1., iyy=1., izz=10., ixy=0.)
  with pytest.raises(ValueError, match="inconsistent inertia"):
    _inertia(bad, np.ones(3))


def test_rotation_is_proper_and_source_native_model_stays_lazy():
  nominal = rigid_chain("rust")
  assert nominal._python_robot_ is None
  actual = apply_perturbation(nominal, PerturbationSpec(seed=8, rules=(
      rule("joint_rotation", std=.2, names=("world_body",)),)))
  assert nominal._python_robot_ is None
  rotation = actual.robot_.joint("world_body").origin.rot()
  np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
  assert np.linalg.det(rotation) == pytest.approx(1)


@pytest.mark.parametrize("distribution,options", [
    ("normal", dict(mean=1., std=.05)), ("uniform", dict(low=.8, high=1.2)),
    ("lognormal", dict(std=.05)), ("loguniform", dict(low=.8, high=1.2))])
def test_distributions_reproducible_and_do_not_touch_global_rng(distribution, options):
  nominal = rigid_chain()
  spec = PerturbationSpec(seed=6, rules=(rule("mass", distribution=distribution, mode="scale", **options),))
  np.random.seed(14)
  before = np.random.get_state()
  a = apply_perturbation(nominal, spec)
  b = apply_perturbation(nominal, spec)
  assert a.robot_.to_dict() == b.robot_.to_dict()
  after = np.random.get_state()
  np.testing.assert_array_equal(before[1], after[1])
  assert before[2:] == after[2:]


@pytest.mark.parametrize("operation", [
    rule("mass", std=0, mean=-100, names=("body",)),
    rule("link_length", std=0, mean=-2, names=("body",)),
    rule("link_length", std=0, names=("rod",)),
    rule("joint_offset", std=.1, names=("body_rod",)),
    rule("mass", distribution="lognormal", mode="scale", std=0, mean=1000),
])
def test_invalid_samples_and_unsupported_targets_leave_source_unchanged(operation):
  nominal = rigid_chain()
  before = nominal.robot_.to_dict()
  with pytest.raises(ValueError):
    apply_perturbation(nominal, PerturbationSpec(rules=(operation,)))
  assert nominal.robot_.to_dict() == before


def test_urdf_length_shortcut_and_native_dynamics_agree():
  from pathlib import Path
  path = Path(__file__).parent / "test_model/branched_fixed.urdf"
  nominal = Kots.from_urdf_file(str(path), backend="rust")
  actual, report = apply_perturbation(nominal, PerturbationSpec(
      seed=42, link_length_relative_std=.02, mass_relative_std=.05,
      rules=(rule("joint_rotation", std=.01), rule("cog_translation", std=.001),
             rule("inertia", distribution="lognormal", mode="scale", std=.1))), return_report=True)
  assert report.length_scales
  assert any(not np.allclose(j.origin.pos(), nominal.robot_.joint(j.name).origin.pos())
             for j in actual.robot_.joints)
  python = Kots.from_json_data(report.model_data, backend="numpy")
  rng = np.random.default_rng(12)
  q, v, a = rng.normal(size=(3, 2, actual.dof()))
  gravity = [.1, .2, -9.81]
  np.testing.assert_allclose(actual.inverse_dynamics(q, v, a, gravity=gravity),
                             python.inverse_dynamics(q, v, a, gravity=gravity), atol=1e-10)


@pytest.mark.parametrize("kwargs", [
    dict(std=-1), dict(std=complex(1)), dict(std=float("inf")),
    dict(distribution="uniform", low=2, high=1),
    dict(distribution="uniform", mode="scale", low=0, high=1),
    dict(distribution="lognormal", std=.1),
    dict(distribution="normal", std=.1, low=0),
])
def test_invalid_distribution_definitions(kwargs):
  with pytest.raises(ValueError):
    NoiseSpec(**kwargs)


def test_group_validation_and_jax_model_constants():
  with pytest.raises(ValueError, match="disjoint"):
    rule("mass", std=.1, groups=(("body", "rod"), ("rod",)))
  with pytest.raises(ValueError, match="unknown or ineligible"):
    apply_perturbation(rigid_chain(), PerturbationSpec(rules=(rule("mass", std=.1, groups=(("absent",),)),)))
  data = rigid_chain().robot_.to_dict()
  nominal = Kots.from_json_data(data, lib="jax")
  actual = apply_perturbation(nominal, PerturbationSpec(seed=4, link_length_relative_std=.03))
  assert actual.lib_ == "jax"
  assert not np.allclose(actual.robot_.joint("body_rod").origin.pos(), [1, 0, 0])
