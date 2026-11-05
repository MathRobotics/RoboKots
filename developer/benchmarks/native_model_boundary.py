"""Measure Python-dict -> native model separately from model serialization."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import robokots._rust_core as extension
from robokots.kots import Kots
from robokots.outward.rust.model import _model_data_from_robot
from .core_state_layout import measure
from .kernel_layout import error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    model = root / "tests/test_model/branched_fixed.urdf"
    k = Kots.from_urdf_file(str(model), order=4)
    data = _model_data_from_robot(k.robot_)
    cls = extension.RustCompiledRobot
    operations = {
        "compile_dict": lambda: cls.from_model_data(data),
        "serialize_and_compile": lambda: cls.from_model_data(_model_data_from_robot(k.robot_)),
    }
    timings = {name: measure(fn, 10, 50, 50) for name, fn in operations.items()}
    q, v, a = np.random.default_rng(812).normal(scale=.2, size=(3, k.dof()))
    output = cls.from_model_data(data).rnea(q, v, a, np.array([.2, -.3, -9.81]))
    report = {
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "numpy": np.__version__,
                        "extension_sha256": hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest()},
        "workload": {"model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                     "warmup": 10, "samples": 50, "calls": 50, "seed": 812,
                     "dtype": "float64", "gravity": [.2, -.3, -9.81]},
        "timings": timings, "rnea_output": output.tolist(),
    }
    if args.compare:
        before = json.loads(args.compare.read_text())
        assert report["workload"] == before["workload"]
        np.testing.assert_allclose(output, before["rnea_output"], atol=1e-12, rtol=1e-12)
        report["accuracy_vs_before"] = error(output, before["rnea_output"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
