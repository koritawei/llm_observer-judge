"""Judge Service — Kafka consumer for LLM-as-a-Judge evaluation.

Consumes ``langfuse.records`` Kafka topic via ``libs/message_queue`` ``MQConsumer``,
filters ``record_type=trace`` messages, and runs the B2→B6 evaluation pipeline.
"""
