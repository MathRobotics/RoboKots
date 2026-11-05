import pytest

from robokots.state_io.jsonl import iter_jsonl_rows, make_jsonl_row


@pytest.mark.parametrize("axis", ["times", "steps"])
@pytest.mark.parametrize("count", [0, 1, 3])
def test_axis_length_mismatch_raises(axis, count):
    with pytest.raises(ValueError):
        list(iter_jsonl_rows(({} for _ in range(2)), **{axis: iter(range(count))}))


def test_times_and_steps_are_both_preserved():
    rows = list(iter_jsonl_rows(iter([{"value": [1]}, {"value": [2]}]),
                               times=iter([.1, .2]), steps=iter([8, 9]), meta={"run": "test"}))
    assert rows == [dict(value=[1], t=.1, step=8, schema_version=1, run="test"),
                    dict(value=[2], t=.2, step=9, schema_version=1, run="test")]


def test_three_way_length_mismatch():
    with pytest.raises(ValueError):
        list(iter_jsonl_rows([{}, {}], times=[0, 1], steps=[0]))
    assert list(iter_jsonl_rows([], times=[], steps=[])) == []


@pytest.mark.parametrize("key", ["t", "step", "schema_version"])
@pytest.mark.parametrize("source", ["state", "meta"])
def test_reserved_keys_cannot_be_overwritten(key, source):
    kwargs = {source: {key: 999}}
    if source == "meta":
        kwargs["state"] = {}
    with pytest.raises(ValueError, match="reserved"):
        make_jsonl_row(**kwargs)


def test_metadata_and_payload_collisions_are_rejected():
    with pytest.raises(ValueError, match="collision"):
        make_jsonl_row({"value": [1]}, meta={"value": "metadata"})


@pytest.mark.parametrize("source", ["state", "meta"])
def test_stringified_key_collisions_are_rejected(source):
    kwargs = {source: {1: "first", "1": "second"}}
    if source == "meta":
        kwargs["state"] = {}
    with pytest.raises(ValueError, match="collision"):
        make_jsonl_row(**kwargs)
