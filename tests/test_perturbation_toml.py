from pathlib import Path
import copy
import tomllib

import pytest

from robokots import NoiseSpec, ParameterPerturbation, PerturbationSpec, apply_perturbation
from robokots.kots import Kots


CONFIG = '''
seed = 42
link_length_relative_std = 0.01
scale_inertia_with_mass = false

[[rules]]
parameter = "mass"
names = ["base", "rod1"]
groups = [["base", "rod1"]]
[rules.noise]
distribution = "uniform"
mode = "scale"
low = 0.8
high = 1.2

[[rules]]
parameter = "cog_translation"
names = ["rod1"]
[rules.noise]
std = 0.001
'''


def test_file_text_and_dict_match_python_spec_and_result(tmp_path):
  path = tmp_path / "摂動.toml"
  path.write_text(CONFIG, encoding="utf-8")
  expected = PerturbationSpec(seed=42, link_length_relative_std=.01,
      scale_inertia_with_mass=False, rules=(
          ParameterPerturbation(parameter="mass", names=("base", "rod1"),
              groups=(("base", "rod1"),),
              noise=NoiseSpec(distribution="uniform", mode="scale", low=.8, high=1.2)),
          ParameterPerturbation(parameter="cog_translation", names=("rod1",),
              noise=NoiseSpec(std=.001)),
      ))
  data = tomllib.loads(CONFIG)
  before = copy.deepcopy(data)
  specs = [PerturbationSpec.from_toml(CONFIG), PerturbationSpec.from_dict(data),
           PerturbationSpec.from_toml_file(path), PerturbationSpec.from_toml_file(str(path))]
  assert data == before
  nominal = Kots.from_json_file(str(Path(__file__).parent / "test_model/soft_rod.json"))
  reference = apply_perturbation(nominal, expected).robot_.to_dict()
  for spec in specs:
    assert spec == expected
    assert apply_perturbation(nominal, spec).robot_.to_dict() == reference


@pytest.mark.parametrize("text,location", [
    ('mass_relatve_std = 0.1', "spec: unknown key"),
    ('seed = true', "spec:"),
    ('rules = "mass"', "spec.rules:"),
    ('rules = [1]', "spec.rules[0]:"),
    ('[[rules]]\nparameter = "mass"', "spec.rules[0].noise:"),
    ('[[rules]]\nparameter = "mass"\n[rules.noise]\nstd = -1', "spec.rules[0].noise:"),
    ('[[rules]]\nparameter = "mass"\n[rules.noise]\nstd = 0.1\nnames = ["base"]', "spec.rules[0].noise: unknown key"),
    ('link_names = {base = true}', "spec.link_names:"),
    ('[[rules]]\nparameter = "mass"\ngroups = ["base"]\n[rules.noise]\nstd = 0.1', "spec.rules[0].groups:"),
])
def test_invalid_configuration_has_location(text, location):
  with pytest.raises(ValueError) as error:
    PerturbationSpec.from_toml(text)
  assert location in str(error.value)


def test_defaults_syntax_and_missing_file(tmp_path):
  assert PerturbationSpec.from_toml("") == PerturbationSpec()
  with pytest.raises(tomllib.TOMLDecodeError):
    PerturbationSpec.from_toml('seed = [')
  with pytest.raises(FileNotFoundError):
    PerturbationSpec.from_toml_file(tmp_path / "missing.toml")
