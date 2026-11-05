"""Contracts and independent agreement checks for the table benchmark."""
import jax
import numpy as np
import pytest

from developer.benchmarks.jacobian_accuracy_table import (
    ROWS, evaluate, make_model, report, selections,
)
from robokots.kots import Kots


def test_table_derivative_orders():
    kots = Kots.from_json_data(make_model(2), order=7)
    for k in range(5):
        states = selections(kots, k)
        assert len(states) == 8
        assert [s.time_order for s in states] == [k + 2] * 5 + [k + 3] * 3
        assert states[0].data_type == ("vel", "acc", "jerk", "snap", "crackle")[k]


@pytest.mark.parametrize("k,zero", [(0, True), (4, False)])
def test_table_accuracy(k, zero):
    jax.config.update("jax_enable_x64", True)
    motion = np.zeros((2, 7)) if zero else np.random.default_rng(17).normal(scale=0.2, size=(2, 7))
    results = evaluate(make_model(2), motion, k, [0.3, -0.4, -9.81])
    assert [r["row"] for r in results] == [row[0] for row in ROWS]
    for item in results:
        assert len(item["errors"]) == 6
        for pair, metrics in item["errors"].items():
            assert metrics["max_abs"] < (1e-4 if "numerical" in pair else 1e-9)


def test_report_keeps_all_five_columns_and_marks_incomplete():
    metadata = dict(utc="test", platform="test", python="test", numpy="test", jax="test",
                    devices="cpu", dof=7, gravity=[0, 0, 0], seed=1, samples=1, max_derivative=4)
    text = report(metadata, [])
    assert "k = 0 | k = 1 | k = 2 | k = 3 | k = 4" in text
    assert "Completed cases: 0 / 40" in text
    assert "pending" in text
