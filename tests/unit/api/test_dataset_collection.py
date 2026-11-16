"""Unimplemented semantics of the corpus-collection API."""

import pytest
from benchmark.routes.benchmark_console import start_dataset_collection
from fastapi import HTTPException


@pytest.mark.asyncio
async def test_collection_start_does_not_fake_remote_execution() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await start_dataset_collection("ds-1")
    assert exc_info.value.status_code == 501
    assert "not implemented" in str(exc_info.value.detail)
