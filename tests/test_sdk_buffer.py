from llmobserverjudge.sdk.buffer import BoundedRingBuffer


def test_drop_oldest() -> None:
    buf = BoundedRingBuffer(capacity=2)
    buf.push("a")
    buf.push("b")
    buf.push("c")
    assert len(buf) == 2
    assert buf.metrics.dropped_spans_total == 1
    assert buf.drain(10) == ["b", "c"]


def test_dag_unknown_dependency() -> None:
    from llmobserverjudge.eval_worker.dag import EvalDAGExecutor
    import pytest

    with pytest.raises(ValueError, match="unknown agent"):
        EvalDAGExecutor(
            [
                {"agent_id": "a", "depends_on": ["missing"]},
            ]
        )
