"""Consumer logic split by source (different MQ / upstream) — each subpackage self-registers on import.

One subpackage per source (``consumers/<source>/``), containing that source's protocol converter
and optional enricher, and self-registering via ``register_converter`` /
``register_enricher`` in its ``__init__`` (aligned with the manifest pattern in ``app/evaluators/registry.py``).

**Adding an MQ source**: append one import here; the subpackage ``__init__`` completes registration; the framework
(protocol / enrichment / runtime / handler) and other sources need no changes.
"""

from benchmark.ingest.mq.consumers import luoshu  # noqa: F401 - self-register converter/enricher on import

__all__: list[str] = []
