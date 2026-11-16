"""EvalCase payload serialization — item_id validation / metadata fault tolerance and symmetry.

item_id is the case's unique id (the anchor that writes the score back): missing / None must fail explicitly,
and ``str(None)`` must not silently produce the string ``"None"`` and pollute downstream reconciliation.
to_payload always writes a non-Optional field (metadata), symmetric with from_payload's
default ``{}``, so an empty metadata does not drop the field.
"""

from typing import Any

import pytest
from benchmark.queue.eval_case import EvalCase


def _full_payload() -> dict[str, Any]:
    return {
        "item_id": "case-1",
        "input": {"query": "q"},
        "output": {"response_text": "r"},
        "expected_output": {"reference": "ref"},
        "source_trace_id": "trace-1",
        "metadata": {"turns": 2},
        "case_id": "ds-item-1",
    }


def test_from_payload_roundtrip() -> None:
    """A complete payload deserialized and serialized again, with the fields matching one to one."""
    case = EvalCase.from_payload(_full_payload())
    assert case.item_id == "case-1"
    assert case.metadata == {"turns": 2}
    payload = case.to_payload()
    assert payload["item_id"] == "case-1"
    assert payload["metadata"] == {"turns": 2}
    assert payload["source_trace_id"] == "trace-1"
    assert payload["case_id"] == "ds-item-1"


def test_from_payload_missing_item_id_raises() -> None:
    """A payload with no item_id is a protocol error — fail explicitly, and do not silently use an empty string."""
    with pytest.raises(ValueError, match="item_id"):
        EvalCase.from_payload({"input": {}})


def test_from_payload_none_item_id_raises() -> None:
    """item_id=None must not become the string 'None' through str() (that would pollute the reconciliation anchor)."""
    with pytest.raises(ValueError, match="item_id"):
        EvalCase.from_payload({"item_id": None})


def test_from_payload_numeric_item_id_kept() -> None:
    """A numeric item_id (a sync-model task_id sequence number) is kept as str, and is not blocked by mistake."""
    case = EvalCase.from_payload({"item_id": 42})
    assert case.item_id == "42"


def test_from_payload_metadata_none_or_dirty_to_empty_dict() -> None:
    """metadata=None / a non-dict dirty value → {} (the annotation is dict[str, Any], and it does not leak)."""
    assert EvalCase.from_payload({"item_id": "i"}).metadata == {}
    assert EvalCase.from_payload({"item_id": "i", "metadata": None}).metadata == {}
    assert EvalCase.from_payload({"item_id": "i", "metadata": [1, 2]}).metadata == {}


def test_to_payload_always_includes_metadata() -> None:
    """metadata is not Optional (default_factory=dict) → an empty one is still written, symmetric with the default."""
    case = EvalCase(item_id="i1")
    payload = case.to_payload()
    assert payload["metadata"] == {}
    # An omitted Optional field is still left out (written only when present)
    assert "source_trace_id" not in payload
    assert "case_id" not in payload
    assert "expected_output" not in payload


def test_empty_metadata_roundtrip_stable() -> None:
    """An empty metadata round-trips stably: {} → the payload contains {} → from_payload restores {}."""
    case = EvalCase(item_id="i1", metadata={})
    assert EvalCase.from_payload(case.to_payload()).metadata == {}
