import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from developer.benchmarks.jacobian_resources import render


def test_resource_benchmark_isolated_workers(tmp_path):
    output = tmp_path / "resources"
    subprocess.run([
        sys.executable, "-m", "developer.benchmarks.jacobian_resources",
        "--dof", "1", "--derivatives", "0", "--rows", "7",
        "--methods", "numpy_full", "reverse_jit", "--repeats", "2",
        "--warmup", "0", "--check-values", "--output", str(output),
    ], cwd=Path(__file__).resolve().parents[1], check=True, timeout=120,
        env=dict(os.environ, JAX_PLATFORMS="cpu"), capture_output=True, text=True)
    data = json.loads(output.with_suffix(".json").read_text())
    assert not data["failures"]
    assert len(data["results"]) == 2
    for result in data["results"]:
        assert len(result["samples_ms"]) == 2
        assert result["median_ms"] == np.median(result["samples_ms"])
        assert result["peak_mib"] >= result["baseline_peak_mib"] > 0
        assert result["post_first_peak_growth_mib"] >= 0
        assert len(result["errors"]) == 2
        assert max(e["max_abs"] for e in result["errors"]) < 1e-10
        assert max(e["max_abs"] for e in result["value_errors"]) < 1e-10
        assert result["shape"] == [1, 3]
    assert "Completed: 2 / 2; failures: 0" in output.with_suffix(".md").read_text()


def test_resource_report_explains_memory_and_failures():
    metadata = dict(utc="test", platform="test", cpu="cpu", dof=7, seed=1, expected=1)
    text = render(metadata, [], [{"error": "timeout"}])
    assert "high-water mark" in text
    assert "failures: 1" in text
    assert "timeout" in text
