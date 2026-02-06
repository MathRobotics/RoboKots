from __future__ import annotations

from typing import Any, Iterable, Optional, Iterator
import json
import numpy as np


# ============================================================
# JSONL helpers (flat schema rows)
# ============================================================
def _to_jsonable(value: Any) -> Any:
    """
    Convert numpy/scalar containers into JSON-serializable Python types.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return value


def make_jsonl_row(
    state: dict,
    *,
    t: Optional[float] = None,
    step: Optional[int] = None,
    meta: Optional[dict] = None,
    schema_version: int = 1,
) -> dict:
    """
    Build one flat JSONL row from a state dict.

    Required by convention:
      - schema_version
      - t or step (if available)
    """
    row: dict[str, Any] = {}
    if t is not None:
        row["t"] = float(t)
    if step is not None:
        row["step"] = int(step)
    row["schema_version"] = int(schema_version)
    if meta:
        row.update(_to_jsonable(meta))
    for k, v in state.items():
        row[str(k)] = _to_jsonable(v)
    return row


def iter_jsonl_rows(
    states: Iterable[dict],
    *,
    times: Optional[Iterable[float]] = None,
    steps: Optional[Iterable[int]] = None,
    meta: Optional[dict] = None,
    schema_version: int = 1,
) -> Iterator[dict]:
    """
    Yield JSONL rows for a sequence of state dicts.

    If times/steps are provided, they are zipped in order.
    """
    if times is not None:
        for st, t in zip(states, times):
            yield make_jsonl_row(st, t=t, meta=meta, schema_version=schema_version)
        return
    if steps is not None:
        for st, s in zip(states, steps):
            yield make_jsonl_row(st, step=s, meta=meta, schema_version=schema_version)
        return
    for st in states:
        yield make_jsonl_row(st, meta=meta, schema_version=schema_version)


def write_jsonl(path: str, rows: Iterable[dict], *, ensure_ascii: bool = False) -> None:
    """
    Write JSON Lines file from an iterable of rows.
    """
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=ensure_ascii)
            f.write("\n")
