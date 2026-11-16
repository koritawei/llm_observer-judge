"""Message enrichment framework — after conversion and before evaluation, attach external data to a message by source.

Protocol conversion (protocol/codec) only does **structural mapping** and is **synchronous**; some sources need to
**asynchronously** query an external system (such as Langfuse) after they have ``project_id`` + ``trace_id``. This layer
provides a "sourced, registrable, async" extension point symmetric with protocol conversion:

- :class:`Enricher` — enricher contract: ``source`` declares which source it claims (matching
  ``BaseTaskMessage.source``), and ``enrich`` asynchronously returns the enriched message.
- :func:`register_enricher` / :func:`enrich` — registry + dispatch. The handler calls
  :func:`enrich` before evaluation; no matching source or an enrichment failure **returns the message unchanged** (never blocks the eval flow).

Per-source enrichment logic lives under ``app/mq_consumer/consumers/<source>/`` and self-registers on import.
This module is only the generic framework and contains no source-specific logic.
"""

from typing import Protocol, runtime_checkable

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.ingest.mq.protocol import BaseTaskMessage

logger = get_logger("agent_eval.mq.enrichment")


class EnrichmentUnavailableError(Exception):
    """Enrichment is a transient failure and must be retried (the handler raises → the worker retries the message).

    Distinct from "permanent defect → skip and commit the offset": the old implementation collapsed both into "enrichment field is
    None", so an external timeout/5xx also dropped the message permanently.

    An enricher must not raise this directly: the :func:`enrich` dispatch layer swallows every exception by contract (enrichment is
    an enhancement and must not block the generic flow). The source should declare the status in ``enrich_status``,
    and the handler raises this exception after reading ``enrich_retry_reason()``.
    """


@runtime_checkable
class Enricher(Protocol):
    """Enricher for one source: declare ``source``, and ``enrich`` asynchronously returns the enriched message."""

    @property
    def source(self) -> str:
        """Source id claimed by this enricher (matches ``BaseTaskMessage.source``)."""
        ...

    async def enrich(self, message: BaseTaskMessage) -> BaseTaskMessage:
        """Fetch enrichment data as needed and return the enriched message (on failure, degrade to returning the original)."""
        ...


#: source → enricher; filled by each source package via register_enricher on import.
_ENRICHERS: dict[str, "Enricher"] = {}


def register_enricher(enricher: "Enricher") -> None:
    """Register a source enricher (keyed by ``source``; a later registration overwrites the same name)."""

    _ENRICHERS[enricher.source] = enricher


async def enrich(message: BaseTaskMessage) -> BaseTaskMessage:
    """Dispatch by ``message.source`` to the matching enricher; no source, no enricher, or an exception all return the original.

    This is the unified extension point "after conversion, before evaluation": enrichment failure never blocks evaluation — enrichment is an **enhancement**,
    not a prerequisite of evaluation.
    """

    source = message.source
    if not source:
        return message
    enricher = _ENRICHERS.get(source)
    if enricher is None:
        return message
    try:
        return await enricher.enrich(message)
    except Exception:
        logger.warning(
            "mq.enrich.failed source=%s trace_id=%s",
            source,
            getattr(message, "trace_id", None),
            exc_info=True,
        )
        return message


async def aclose_enrichers() -> None:
    """On shutdown, close enrichers that hold resources (such as an httpx client); enrichers without ``aclose`` are skipped.

    Lifecycle teardown symmetric with :func:`enrich`: called by the MQ service after stopping consumption and workers, to release
    process-wide reused connections held by each source enricher and avoid leaks. A close error on one enricher only warns and does not affect the others.
    """

    for source, enricher in _ENRICHERS.items():
        aclose = getattr(enricher, "aclose", None)
        if aclose is None:
            continue
        try:
            await aclose()
        except Exception:  # noqa: BLE001 - shutdown cleanup is best-effort
            logger.warning("mq.enrich.aclose_failed source=%s", source, exc_info=True)


__all__ = [
    "Enricher",
    "EnrichmentUnavailableError",
    "aclose_enrichers",
    "enrich",
    "register_enricher",
]
