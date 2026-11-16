"""Inbound message protocol layer — message shape + tolerant decode + protocol conversion.

The three modules each have one job:

- :mod:`~benchmark.ingest.mq.protocol.schema` — :class:`BaseTaskMessage` (abstract contract, no business
  fields; the framework depends only on it); concrete fields are defined per source in ``consumers/<source>/schema.py``.
- :mod:`~benchmark.ingest.mq.protocol.codec` — :class:`TolerantEvalCodec`: tolerant decode +
  inbound protocol conversion (bytes → dict → ``convert_for_source(source)``); each consumer is bound to its source.
- :mod:`~benchmark.ingest.mq.protocol.dispatch` — :class:`ProtocolConverter` contract +
  :func:`register_converter` + :func:`convert_for_source`: generic framework that routes by **topic → source → converter**
  and dispatches by source (concrete converters live under consumers/<source> and self-register).

External callers import from this package facade; the implementation is split across the three modules above.
"""

from benchmark.ingest.mq.protocol.codec import TolerantEvalCodec
from benchmark.ingest.mq.protocol.dispatch import (
    ProtocolConverter,
    convert_for_source,
    register_converter,
    registered_sources,
)
from benchmark.ingest.mq.protocol.schema import BaseTaskMessage

__all__ = [
    "BaseTaskMessage",
    "ProtocolConverter",
    "TolerantEvalCodec",
    "convert_for_source",
    "register_converter",
    "registered_sources",
]
