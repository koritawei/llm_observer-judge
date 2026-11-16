"""Domain-layer business enums — owner routing.

``CasePriority`` and ``MatchType`` define the business meaning of owner-route matching.
They are domain concepts, not ORM columns, so they live in the domain layer rather than
``data_model.enums``.

    - ``CasePriority`` — case priority (P0/P1/P2), applied to ``OwnerMapping.priority``
    - ``MatchType`` — the three owner-route match kinds, applied to ``OwnerMapping.match_type``

Service layers (for example ``badcase_attribution.aone``) import along ``services → domain``.
ORM column enums stay in :mod:`benchmark.records.enums`.
"""

from enum import StrEnum


class CasePriority(StrEnum):
    """Case priority level.

    Applied to ``OwnerMapping.priority``. ``P0`` is urgent, ``P2`` is routine.
    """

    P0 = "P0"
    P1 = "P1"
    P2 = "P2"


class MatchType(StrEnum):
    """The three match kinds for owner routing.

    Applied to ``OwnerMapping.match_type``.
    """

    AGENT_CAUSE = "agent_cause"
    OWNER_GROUP = "owner_group"
    DEFAULT = "default"


class PipelineSource(StrEnum):
    """Source channel written to ``trace_pipeline_records.trace_meta_json.source``.

    Not an ORM scalar column. Each entry point injects it on
    ``EvaluatorRequest.metadata['pipeline_source']``, and ``after_eval`` writes the JSON.

    Values:
    - ``kafka`` / ``metaq`` / ``http`` / ``benchmark`` / ``sidecar`` — fixed channel names
    - the dataset entry writes the composite ``dataset-{dataset_name}-{run_name}``
      (see :func:`format_dataset_pipeline_source`; this enum's ``DATASET`` is only the prefix)
    """

    KAFKA = "kafka"
    METAQ = "metaq"  # MetaQ (luoshu RocketMQ) consume path
    HTTP = "http"
    DATASET = "dataset"
    BENCHMARK = "benchmark"
    SIDECAR = "sidecar"  # sidecar-trace-worker MetaQ notify + Lindorm/OSS read-back


#: Request metadata key carrying :class:`PipelineSource` (avoids clash with
#: MQ ``metadata['source']`` like luoshu).
PIPELINE_SOURCE_METADATA_KEY = "pipeline_source"


def format_dataset_pipeline_source(
    dataset_name: str | None,
    run_name: str | None = None,
) -> str:
    """Compose ``trace_meta_json.source`` for dataset-eval rows.

    Format: ``dataset-{dataset_name}-{run_name}`` (empty segments allowed).
    Example: ``dataset-vehicle-baseline-v1-eval-example-20260731``.
    """
    name = (dataset_name or "").strip()
    run = (run_name or "").strip()
    return f"{PipelineSource.DATASET.value}-{name}-{run}"


def is_dataset_pipeline_source(source: str | None) -> bool:
    """True only for composite ``dataset-…`` sources (no bare / ``dataset+``)."""
    if not source:
        return False
    s = source.strip().lower()
    prefix = f"{PipelineSource.DATASET.value}-"
    return s.startswith(prefix)


__all__ = [
    "CasePriority",
    "MatchType",
    "PipelineSource",
    "PIPELINE_SOURCE_METADATA_KEY",
    "format_dataset_pipeline_source",
    "is_dataset_pipeline_source",
]
