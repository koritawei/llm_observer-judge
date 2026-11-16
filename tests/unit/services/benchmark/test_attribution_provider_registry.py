"""Unit tests for AttributionProviderRegistry — register / route / type contract / factory."""

import typing
from unittest.mock import MagicMock

import pytest
from benchmark.runs.common.external_client import ExternalJobStatusResponse
AttributionProvider = None
AttributionProviderRegistry = None


@pytest.mark.asyncio
async def test_register_and_get_provider() -> None:
    """register → get_provider round trip."""
    registry = AttributionProviderRegistry()
    provider = MagicMock()
    registry.register("external", provider)
    assert registry.get_provider("external") is provider


def test_get_provider_unknown_returns_none() -> None:
    """An unregistered provider name → None."""
    registry = AttributionProviderRegistry()
    assert registry.get_provider("nonexistent") is None


@pytest.mark.asyncio
async def test_from_external_client_registers_external() -> None:
    """from_aone_client(client) → register the "aone" provider."""
    mock_client = MagicMock()
    registry = AttributionProviderRegistry.from_external_client(mock_client)
    provider = registry.get_provider("external")
    assert provider is not None
    # AoneAttributionProvider should wrap the client
    assert hasattr(provider, "query_job_status")


def test_from_external_client_with_none() -> None:
    """from_aone_client(None) → register no provider (warn only)."""
    registry = AttributionProviderRegistry.from_external_client(None)
    assert registry.get_provider("external") is None


def test_register_multiple_providers() -> None:
    """Registering several providers does not interfere."""
    registry = AttributionProviderRegistry()
    p1 = MagicMock()
    p2 = MagicMock()
    registry.register("external", p1)
    registry.register("other", p2)
    assert registry.get_provider("external") is p1
    assert registry.get_provider("other") is p2


def test_query_job_status_returns_concrete_type() -> None:
    """Protocol.query_job_status returns a concrete AoneJobStatusResponse.

    Guard against regression: returning object would weaken type checking (the caller would still have to cast the whole row by hand).
    """
    hints = typing.get_type_hints(AttributionProvider.query_job_status)
    assert hints["return"] is ExternalJobStatusResponse
