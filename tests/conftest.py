import asyncio
import hashlib
import hmac
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ.setdefault("EVALBENCH_SECRET_DATABASE", "eval")
os.environ.setdefault("EVALBENCH_SECRET_PLUGIN_TRUST", "test-trust")
os.environ.setdefault("EVALBENCH_SECRET_CALLBACK", "test-callback")
os.environ.setdefault("EVALBENCH_ENV", "local")

from benchmark.kernel.runtime import open_kernel

CONF = Path(__file__).resolve().parents[1] / "conf"
ROOT = CONF.parent


@pytest.fixture
async def kernel():
    instance = open_kernel(CONF)
    await instance.setup()
    await instance.leases.redis.flushdb()
    await instance.store.truncate()
    try:
        yield instance
    finally:
        await instance.close()


@pytest.fixture
async def live(kernel):
    kernel.start_workers()
    yield kernel


async def wait_terminal(kernel, run_id: str, timeout: float = 20):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = await kernel.store.get_run(run_id)
        if last and last["status"] in {"completed", "failed", "stopped"}:
            return last
        await asyncio.sleep(0.05)
    raise AssertionError(last)


@pytest.fixture(autouse=True)
def _runtime_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_EVAL_MODEL_API_KEY", "sk-test-dummy")
    monkeypatch.setenv("AGENT_EVAL_JUDGE_AMAP_AGENT_MODEL_API_KEY", "test-placeholder-not-used")
    monkeypatch.setenv("AGENT_EVAL_JUDGE_AMAP_SEARCH_AGENT_MODEL_API_KEY", "test-placeholder-not-used")
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def reset_provider_registry() -> Iterator[None]:
    from benchmark.libs.agenthost.app.providers import ProviderRegistry

    ProviderRegistry._discovered = False
    yield
    ProviderRegistry._discovered = False


@pytest.fixture(autouse=True)
def _clear_domain_parsed_cache() -> None:
    from benchmark.domain._cache import clear_parsed_cache

    clear_parsed_cache()


def pytest_sessionfinish(session, exitstatus) -> None:
    del session, exitstatus
    import gc
    from concurrent.futures import ThreadPoolExecutor

    for obj in gc.get_objects():
        if isinstance(obj, ThreadPoolExecutor):
            obj.shutdown(wait=False, cancel_futures=True)


def case(item_id="1", output="hello", expected="hello"):
    return {
        "item_id": item_id,
        "input": {"text": expected},
        "output": {"text": output},
        "expected": {"text": expected},
    }
