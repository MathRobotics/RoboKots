from copy import deepcopy

import pytest

from developer.benchmarks.compare_time_ad_formulations import compare, render


def runs():
    metadata = dict(platform="test", dof=7, seed=1, gravity=[0, 0, -9.81], model={},
                    motions=[[1]], rows=[7], derivatives=[0], repeats=5, warmup=1,
                    check_values=True, utc="previous")
    result = dict(row=7, k=0, environment={"jax": "test"}, first_ms=1000., median_ms=1.,
                  peak_mib=200., value=[1.], other_value=[2.], jacobian=[[3.]], other_jacobian=[[4.]])
    fk = dict(metadata=metadata, results=[dict(result, method="time_forward_jit")], failures=[
        dict(row=7, k=0, method="time_reverse_jit", error="timed out after 600 seconds")])
    new = dict(metadata=dict(metadata, utc="new"), results=[dict(result, method="id_time_forward_jit")], failures=[])
    return fk, new


def test_comparison_preserves_sources_and_marks_missing_results():
    fk, new = runs()
    original = deepcopy((fk, new))
    result = compare(fk, new)
    assert (fk, new) == original
    assert result["fk_utc"] == "previous" and result["id_utc"] == "new"
    assert len(result["direct_errors_id_vs_fk"]) == 1
    assert result["direct_errors_id_vs_fk"][0]["errors"]["jacobian"]["max_abs"] == 0
    text = render(result)
    assert "previous run" in text and "new run" in text
    assert "timeout" in text and "not completed" in text
    assert "Derivative coefficients + AD" in text
    assert "not a separate mathematical differentiation method" in text


@pytest.mark.parametrize("key", ["motions", "gravity", "repeats"])
def test_comparison_rejects_mismatched_conditions(key):
    fk, new = runs()
    new["metadata"][key] = "different"
    with pytest.raises(ValueError, match=key):
        compare(fk, new)


def test_comparison_rejects_mismatched_environment():
    fk, new = runs()
    new["results"][0]["environment"] = {"jax": "different"}
    with pytest.raises(ValueError, match="environment"):
        compare(fk, new)
