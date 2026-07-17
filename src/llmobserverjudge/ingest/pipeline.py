"""Sampling → Index → Write pipeline (SDD-IG-12)."""

from __future__ import annotations

from llmobserverjudge.ingest.nodes import (
    IndexNode,
    IngestNode,
    IngestState,
    SamplingConfig,
    SamplingNode,
    WriteNode,
)
from llmobserverjudge.shared.mq import MessageQueue
from llmobserverjudge.shared.storage import InMemorySpanStore, SpanStore


class IngestPipeline:
    def __init__(
        self,
        nodes: list[IngestNode] | None = None,
        *,
        mq: MessageQueue | None = None,
        store: SpanStore | None = None,
        sampling_config: SamplingConfig | None = None,
        force_sample: bool | None = None,
    ) -> None:
        if nodes is not None:
            self.nodes = nodes
            return
        if mq is None:
            raise ValueError("mq is required when nodes not provided")
        store = store or InMemorySpanStore()
        self.nodes = [
            SamplingNode(mq, config=sampling_config, force_sample=force_sample),
            IndexNode(),
            WriteNode(store),
        ]
        self.store = store

    async def run(self, state: IngestState) -> IngestState:
        for node in self.nodes:
            state = await node.run(state)
        return state
