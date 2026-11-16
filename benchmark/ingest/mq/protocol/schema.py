"""**Abstract base class** for an MQ message — no business fields, only the contract the framework depends on.

The framework layer (codec / handler / runtime / enrichment) depends only on :class:`BaseTaskMessage`
and knows no concrete source or protocol. Each source defines **its own** concrete
schema in ``consumers/<source>/schema.py`` (modeling its raw JSON directly) and implements this contract:

- ``evaluator_name``: which evaluator to route to (used by the handler).
- ``source``: source id; the enrichment layer dispatches by it (default ``None`` = no enrichment).
- ``to_evaluator_request()``: project to the domain input :class:`EvaluatorRequest`.
- ``unevaluatable_reason()``: **evaluability gate** after enrichment (default is allow).

Framework plumbing (built into the base class; subclasses need not override):

- ``decode_error`` / ``rejected()``: placeholder and validation gate for a bad message (not JSON / unsupported protocol).

The base class is a pydantic ``BaseModel`` (so subclasses can be serialized by ``message_queue``'s ``JsonCodec``);
``extra="ignore"`` lets a subclass tolerate unknown fields, and a bad message degrades to a skippable placeholder instead of raising at decode and blocking the queue.
"""

from pydantic import BaseModel, ConfigDict, PrivateAttr

from benchmark.evaluators.base import EvaluatorRequest


class BaseTaskMessage(BaseModel):
    """Abstract contract for an eval-task message — no business fields; concrete fields are defined by each source subclass."""

    model_config = ConfigDict(extra="ignore")

#: Logic name of the evaluator this message routes to (used by the handler): a shared field, default empty string (= bad route, handler skips);
#: each source subclass overrides the default (for example luoshu → "luoshu").
    evaluator_name: str = ""

#: Non-None means the raw body is not valid JSON / an unsupported protocol (decode or dispatch failed).
#: Marked by codec / convert_for_source and read by the handler validation gate. Private attribute; not serialized.
    _decode_error: str | None = PrivateAttr(default=None)

#: Result status of enrichment, written by each source enricher. The generic layer only defines the field and **does not
#: prescribe the values** (concrete statuses belong to the source). Why it exists: enrichment failure has several causes (missing id /
#: no instance config / query error / truly absent), and the old implementation could only infer from "the enrichment field is None",
#: so the four causes collapsed into one log line and could not be told apart.
    enrich_status: str | None = None

    @property
    def decode_error(self) -> str | None:
        """Failure reason for a bad message (illegal structure / unsupported protocol); None for a normal message."""

        return self._decode_error

    @classmethod
    def rejected(cls, reason: str) -> "BaseTaskMessage":
        """Build a bad-message placeholder: an empty instance plus the failure reason, for the handler validation gate to skip."""

        msg = cls()
        msg._decode_error = reason
        return msg

    @property
    def source(self) -> str | None:
        """Source id (implemented by the subclass); the enrichment layer dispatches an enricher by it. Default None = no enrichment."""

        return None

    @property
    def requires_trace_resolution(self) -> bool:
        """Whether the business ``trace_id`` must be resolved to a Langfuse primary key before evaluation (the handler passes this through to evaluate).

        Default True (keep the generic behavior). If the ``trace_id`` written by the source is itself the Langfuse primary key and the eval data
        rides on the message (for example Luoshu side-by-side: candidate content is sent with the request), a subclass overrides this to False and skips before_eval.
        """

        return True

    def to_evaluator_request(self) -> EvaluatorRequest:
        """Project to the domain input :class:`EvaluatorRequest` (implemented by the subclass)."""

        raise NotImplementedError

    def unevaluatable_reason(self) -> str | None:
        """Reason the message still does not meet the eval precondition after enrichment; ``None`` = evaluable.

        Default ``None`` (no gate; existing source behavior is unchanged). **The rule "this message can never be evaluated" belongs to
        the source** (only the source knows which data it projected into the eval input), so a subclass overrides it and the handler
        asks this one question before ``evaluate`` — if it is not ``None``, warn and skip (already consumed; commit the offset).

        Why here instead of letting the evaluator raise: incomplete data is a **permanent** defect and a retry will never succeed. If it
        enters evaluation, it becomes "evaluator error → classified by ``EvaluatorError.code`` → a wrong label retries forever",
        and every retry reruns enrich (another external fetch). Stopping it here **skips once** and does not need
        the evaluator to cooperate.

        Split with :meth:`enrich_retry_reason`: this method handles only **permanent** defects (skip);
        transient failures (should retry) go there.
        """

        return None

    def enrich_retry_reason(self) -> str | None:
        """Reason enrichment is a **transient** failure and should be retried; ``None`` = no retry needed.

        Default ``None`` (existing source behavior is unchanged). Symmetric with :meth:`unevaluatable_reason`, and also declared by
        the source — only the source knows which of its enrichment failures are recoverable.

        **Why it exists**: an enrichment failure may be permanent (the data really is absent) or transient
        (timeout / 5xx / network blip). In the old implementation both looked like "the enrichment field is None", so
        ``unevaluatable_reason`` treated both as a permanent defect and **dropped them forever** — one blip of an external service silently dropped
        the message, even though a retry would have succeeded.

        A subclass that declares a retry must itself ensure "a retry might actually succeed": a failure caused by process-level cache (for example credentials
        are not configured so the instance map is empty) is **not** transient — a retry in the same process will never succeed and
        causes head-of-line blocking.
        """

        return None


__all__ = ["BaseTaskMessage"]
