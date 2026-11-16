"""Protocol-conversion dispatch framework — dispatch a raw message to the converter for its source and normalize it to a :class:`BaseTaskMessage` subclass.

Routing is **topic → source → converter**: each source (``consumers/<source>/``) registers one converter
(keyed in the registry by its ``source``), and the ``runtime`` layer binds each source to one topic plus a dedicated codec;
a message is parsed by the converter of the source for the topic it arrived on, and is **no longer classified by a content field such as ``eventType``**
(topic names differ across environments, but one topic carries only one source).

This module provides **converter contract + registry + dispatch by source**:

- :class:`ProtocolConverter` — converter contract: ``source`` declares the claimed source id, and ``convert``
  maps a raw dict to a concrete :class:`BaseTaskMessage` subclass (missing required fields should raise).
- :func:`register_converter` / :data:`_CONVERTERS` — registry keyed by source;
  :func:`convert_for_source` looks up the converter by source: **unregistered source / conversion error → reject**
  (returns ``BaseTaskMessage.rejected`` with a reason, skipped by the handler validation gate).

Adding an upstream only requires implementing :class:`ProtocolConverter` under ``consumers/<source>`` and calling
:func:`register_converter`, then binding that source to a topic in ``conf``; this framework and the
handler / eval service need no changes.
"""

from typing import Any, Protocol, runtime_checkable

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.ingest.mq.protocol.schema import BaseTaskMessage

logger = get_logger("agent_eval.mq.protocol")


@runtime_checkable
class ProtocolConverter(Protocol):
    """Converter for one upstream source: declare ``source`` and normalize that source's raw structure with ``convert``."""

    #: Claimed source id (matches consumers/<source> and the source key bound in conf).
    source: str

    def convert(self, raw: dict[str, Any]) -> BaseTaskMessage:
        """Map a raw dict to a unified :class:`BaseTaskMessage` subclass (missing required fields should raise)."""
        ...


#: source → converter; filled by each source package (consumers/<source>) via register_converter on import.
_CONVERTERS: dict[str, ProtocolConverter] = {}


def register_converter(converter: ProtocolConverter) -> None:
    """Register a source converter (keyed by ``converter.source``; a later registration overwrites the same name)."""

    _CONVERTERS[converter.source] = converter


def registered_sources() -> list[str]:
    """Registered source ids (sorted, for diagnostics/logs)."""

    return sorted(_CONVERTERS)


def convert_for_source(source: str, raw: dict[str, Any]) -> BaseTaskMessage:
    """Look up the converter by ``source`` and normalize the raw dict to a concrete subclass (the framework only knows the abstract parent).

    An unregistered source (unknown source configured / that source package not imported) or a ``convert`` error (for example a required field
    is missing) → **reject** (return ``BaseTaskMessage.rejected`` with a reason, skipped by the handler validation gate) —
    a structure claimed by no source, or one that fails validation, is a bad route, not a silent normalization.
    """

    converter = _CONVERTERS.get(source)
    if converter is None:
        return BaseTaskMessage.rejected(
            f"unsupported source (no converter registered): source={source!r}"
        )
    try:
        return converter.convert(raw)
    except Exception as exc:
        logger.warning("mq.protocol.convert_failed source=%s", source, exc_info=True)
        return BaseTaskMessage.rejected(f"converter error: source={source!r}: {exc}")


__all__ = [
    "ProtocolConverter",
    "convert_for_source",
    "register_converter",
    "registered_sources",
]
