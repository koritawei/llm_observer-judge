"""Stubs for mq_consumer tests — a concrete subclass of :class:`BaseTaskMessage` plus a matching converter.

After the framework dropped the default protocol, :class:`BaseTaskMessage` has no business fields, and ``evaluator_name`` /
``to_evaluator_request`` are abstract contracts. Each concrete schema belongs to a source (such as luoshu). To unit-test the
generic orchestration of handler / service / enrichment **without depending on any concrete source**, this provides a minimal
usable concrete subclass :class:`StubTaskMessage` and a matching :class:`StubConverter`:

- :class:`StubTaskMessage` — carries the evaluation-input fields directly (``input`` / ``output`` /
  ``metadata`` / ``trace_id``, and so on). ``evaluator_name`` uses the shared field on the base (passed
  at construction), and ``source`` is taken from ``metadata["source"]`` (so the enrichment layer can dispatch by source).
- :class:`StubConverter` — claims the source :data:`STUB_SOURCE` (routing is topic → source →
  converter, and no longer discriminates by content). ``convert`` restores the message with ``model_validate``. Service tests
  that take a codec round-trip (``TolerantEvalCodec(STUB_SOURCE)`` → ``convert_for_source``)
  register it via ``register_converter``.
"""

from typing import Any

from benchmark.evaluators.base import EvaluatorRequest
from benchmark.ingest.mq.protocol import BaseTaskMessage
from pydantic import ConfigDict, Field

#: Source id claimed by the stub converter (the source key of the consumers config in the service tests, and the codec it is bound to).
STUB_SOURCE = "stub"


class StubTaskMessage(BaseTaskMessage):
    """A generic test message: it models the evaluation input directly, for orchestration tests that do not depend on a concrete source."""

    model_config = ConfigDict(extra="ignore")

    input: Any = None
    output: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None
    project_id: str | None = None
    agent_name: str | None = None

    @property
    def source(self) -> str | None:
        return self.metadata.get("source")

    def to_evaluator_request(self) -> EvaluatorRequest:
        metadata = dict(self.metadata)
        if self.trace_id and "trace_id" not in metadata:
            metadata["trace_id"] = self.trace_id
        return EvaluatorRequest(
            input=self.input,
            output=self.output,
            metadata=metadata,
            trace_id=self.trace_id,
            agent_name=self.agent_name,
            project_id=self.project_id,
        )


class StubConverter:
    """Claims the source :data:`STUB_SOURCE` and restores a raw dict into a :class:`StubTaskMessage`."""

    source = STUB_SOURCE

    def convert(self, raw: dict[str, Any]) -> StubTaskMessage:
        return StubTaskMessage.model_validate(raw)


__all__ = ["STUB_SOURCE", "StubConverter", "StubTaskMessage"]
