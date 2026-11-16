"""Benchmark streaming shape — Kafka realtime-stream trigger (migrated from ``app/judge``).

Consumes the ``langfuse.records`` topic, runs a per-record streaming LLM judge, then synchronous triage attribution,
then persists to ``trace_pipeline_records`` (no run state machine). The lifecycle is carried by
:class:`~benchmark.streaming.runner.StreamingBenchmarkRunner`;
dynamic topic / task management lives in ``reconciler`` + ``registry``.
"""
