"""Tolerant JSON codec + protocol conversion — raw bytes → a concrete :class:`BaseTaskMessage` subclass.

Two jobs:

1. **Tolerant decode**: a bad message that is not JSON is marked "pending reject" (``BaseTaskMessage.rejected``)
   and handed to the handler validation gate. ``decode`` on the shared-library :class:`~message_queue.codec.JsonCodec`
   raises directly on non-JSON bytes; :meth:`~message_queue.consumer.MQConsumer.run` catches that, logs
   **ERROR**, and ``continue``s, so the policy is outside agent-eval's control. This codec turns "decode failed" from an exception
   into a **marked placeholder message**, so the validate/skip policy stays inside agent-eval.
2. **Protocol conversion**: a valid JSON object is passed to :func:`benchmark.ingest.mq.protocol.convert_for_source`
   and normalized to a concrete :class:`BaseTaskMessage` subclass for the ``source`` bound to this codec; an unregistered source
   or a conversion error is rejected (``BaseTaskMessage.rejected``, skipped by the handler validation gate).

Takes effect only on the agent-eval side (injected via ``register_consumer(codec=...)``); the shared library is unchanged.
"""

import json
from typing import Any

from benchmark.libs.mq.codec import JsonCodec

from benchmark.ingest.mq.protocol.dispatch import convert_for_source
from benchmark.ingest.mq.protocol.schema import BaseTaskMessage

#: Cap on raw-byte preview in logs/reasons, so a large message does not flood the log.
_PREVIEW_BYTES = 64


class TolerantEvalCodec:
    """Delegate encoding to :class:`JsonCodec`; when decoding a :class:`BaseTaskMessage` subclass, convert to dict first, then run the protocol layer.

    Non-:class:`BaseTaskMessage` target types delegate straight to :class:`JsonCodec` (other uses are left alone).
    Decode failure (not JSON / not an object / unclaimed / conversion error) → mark a placeholder pending reject.
    """

    def __init__(self, source: str) -> None:
        #: Source bound to this codec (the topic this consumer subscribes to); decode picks the converter from it.
        self._source = source
        self._inner = JsonCodec()

    def encode(self, value: Any) -> bytes:
        return self._inner.encode(value)

    def decode[T](self, data: bytes, target_type: type[T]) -> T:
        if not (isinstance(target_type, type) and issubclass(target_type, BaseTaskMessage)):
            return self._inner.decode(data, target_type)
        try:
            raw = json.loads(data)
            if not isinstance(raw, dict):
                return BaseTaskMessage.rejected(  # type: ignore[return-value]
                    f"非 JSON 对象结构 (type={type(raw).__name__}, "
                    f"preview={data[:_PREVIEW_BYTES]!r})"
                )
            return convert_for_source(self._source, raw)  # type: ignore[return-value]
        except Exception as exc:
            reason = (
                f"非 JSON 结构 (bytes={len(data)}, preview={data[:_PREVIEW_BYTES]!r}): {exc}"
            )
            return BaseTaskMessage.rejected(reason)  # type: ignore[return-value]


__all__ = ["TolerantEvalCodec"]
