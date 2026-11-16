"""EvalCase — unified data abstraction for one evaluation case (design doc §2.2).

A single evaluation case travels in the BullMQ job payload and is not staged in PG.
The processor deserializes ``job.data`` into an ``EvalCase`` and then runs the evaluation.
"""

from dataclasses import dataclass, field
from typing import Any, Self


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One evaluation case (travels in the BullMQ job payload; not staged in PG).

    Attributes:
        item_id: unique case id (sync model = item_id, async model = task_id).
        input: user input / query.
        output: the agent's actual output (the reply to judge).
        expected_output: ground truth / reference answer.
        source_trace_id: related Langfuse trace ID (anchor for writing the score back; optional).
        metadata: business metadata (react_rounds / turns, and so on).
        case_id: Langfuse dataset item ID (optional, used to rebuild the detail page).
    """

    item_id: str
    input: Any = None
    output: Any = None
    expected_output: Any = None
    source_trace_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    case_id: str | None = None

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> Self:
        """Deserialize BullMQ job.data into an EvalCase.

        Job data shape: ``{eval_type, run_id, item_id, input, output,
        expected_output, source_trace_id, metadata, case_id}``.
        The processor reads ``eval_type`` / ``run_id`` itself; they are not part of EvalCase.

        :raises ValueError: ``item_id`` is missing / None / empty — item_id is the
            reconcile anchor for writing the score back. Silently emitting ``"None"`` / ``""``
            would pollute downstream reconcile, so this must fail explicitly.
        """
        item_id = data.get("item_id")
        if item_id is None or item_id == "":
            msg = f"item_id is required in payload (got {item_id!r})"
            raise ValueError(msg)
        raw_metadata = data.get("metadata")
        return cls(
            item_id=str(item_id),
            input=data.get("input"),
            output=data.get("output"),
            expected_output=data.get("expected_output"),
            source_trace_id=data.get("source_trace_id"),
            # isinstance tightens this: "metadata": None / a non-dict dirty value → {}
            # (more explicit than ``or {}``, and it does not let a non-dict truthy value through)
            metadata=raw_metadata if isinstance(raw_metadata, dict) else {},
            case_id=data.get("case_id"),
        )

    def to_payload(self) -> dict[str, Any]:
        """Serialize to BullMQ job data (without eval_type / run_id; the caller appends those).

        Field types follow the same rules: optional fields (expected_output /
        source_trace_id / case_id) are written only when not None; ``metadata`` is not
        optional (default_factory=dict) → always written, symmetric with the default ``{}``
        in from_payload, so an empty metadata does not drop the field.
        """
        payload: dict[str, Any] = {
            "item_id": self.item_id,
            "input": self.input,
            "output": self.output,
            "metadata": self.metadata,
        }
        if self.expected_output is not None:
            payload["expected_output"] = self.expected_output
        if self.source_trace_id is not None:
            payload["source_trace_id"] = self.source_trace_id
        if self.case_id is not None:
            payload["case_id"] = self.case_id
        return payload


__all__ = ["EvalCase"]
