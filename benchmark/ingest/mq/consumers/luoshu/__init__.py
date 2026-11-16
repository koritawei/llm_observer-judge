"""Luoshu source consumer logic — protocol conversion + Langfuse enrichment (self-registers on import).

Importing this package registers :class:`LuoshuConverter` (protocol conversion) and :class:`LuoshuLangfuseEnricher`
(Langfuse enrichment by project_id + trace_id) into the framework converter / enricher registries;
the handler and eval service do not need to know the concrete source.
"""

from benchmark.ingest.mq.consumers.luoshu.converter import LuoshuConverter
from benchmark.ingest.mq.consumers.luoshu.enrichment import LuoshuTraceEnricher
from benchmark.ingest.mq.consumers.luoshu.schema import SOURCE, LuoshuTaskMessage
from benchmark.ingest.mq.enrichment import register_enricher
from benchmark.ingest.mq.protocol import register_converter

register_converter(LuoshuConverter())
register_enricher(LuoshuTraceEnricher())

__all__ = ["SOURCE", "LuoshuConverter", "LuoshuTraceEnricher", "LuoshuTaskMessage"]
