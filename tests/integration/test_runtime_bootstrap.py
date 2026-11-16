"""AppRuntime bootstrap integration test — verify symmetric prefix assembly from conf/app.<env>.yaml.

It takes the same ``build_app_runtime`` + ``bootstrap()`` main path as the main lifespan,
does not bypass ``_init_agent``, and does not paper over gaps with MagicMock.
"""

from collections.abc import AsyncIterator
from pathlib import Path

import benchmark.evaluators.registry
import business.agents.registry  # noqa: F401 — eager-import attribution @agent_component
import pytest
import pytest_asyncio
from benchmark.libs.agenthost.agents.config.schemas import AgentConfigSchema
from benchmark.libs.agenthost.app import AppRuntime
from benchmark.libs.agenthost.app.providers import ProviderRegistry
from business.evaluators.example.agent import ExampleEvaluatorAgent
from benchmark.paths import app_config_path
from benchmark.service.bootstrap import AgentEvalRuntime
from benchmark.services.evaluation import build_app_runtime
from benchmark.trace.tools import (  # noqa: F401 — @tool_component + tool adapter
    TraceToolProvider,
)
from business.agents.example.attribution.agent import AttributionAgent
SERVICE_ROOT = Path(__file__).resolve().parents[2]


def test_build_app_runtime_uses_agent_eval_runtime() -> None:
    """Model retry must be injected by default at the runtime boundary and must not depend on each evaluator wrapping it by hand."""

    runtime = build_app_runtime(app_config_path("local"), SERVICE_ROOT)

    assert isinstance(runtime, AgentEvalRuntime)


def test_build_app_runtime_loads_trace_tool_extension() -> None:
    """tools.extensions.trace must be registered before load_yaml, otherwise startup raises ValueError immediately."""

    from benchmark.trace.tools import TraceToolExt

    for env in ("local",):
        runtime = build_app_runtime(app_config_path(env), SERVICE_ROOT)
        ext = runtime.app_config.get_section_extension("tools", "trace", TraceToolExt)
        assert isinstance(ext, TraceToolExt), env
        assert ext.max_observation_chars == 65536, env
        assert ext.timeout_seconds == 30, env
        assert ext.max_retries == 3, env
        assert ext.resolve_lookback_days == 14, env


@pytest_asyncio.fixture()
async def runtime(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AppRuntime]:
    monkeypatch.setenv("AGENT_EVAL_MODEL_API_KEY", "test-placeholder-not-used")
    monkeypatch.setenv(
        "AGENT_EVAL_JUDGE_AMAP_AGENT_MODEL_API_KEY",
        "test-placeholder-not-used",
    )
    # Mirror main.py lifespan: adapter must exist before bootstrap discover.
    from benchmark.libs.agenthost.tools.component_metadata import _build_tool_component_provider

    ProviderRegistry.register_metadata_adapter("tool", _build_tool_component_provider)
    rt = build_app_runtime(app_config_path("local"), SERVICE_ROOT)
    await rt.bootstrap()
    try:
        yield rt
    finally:
        await rt.shutdown()


async def test_both_agents_loaded_with_symmetric_prefix_keys(runtime: AppRuntime) -> None:
    """judge_example + attribution_example are really assembled into agent_manager via _init_agent."""

    manager = runtime.agent_manager
    assert set(manager.list()) >= {"judge_example", "attribution_example"}

    judge = manager.get("judge_example")
    attribution = manager.get("attribution_example")
    assert isinstance(judge, ExampleEvaluatorAgent)
    assert isinstance(attribution, AttributionAgent)

    assert runtime.get_llm("judge_example") is not None
    assert runtime.get_llm("attribution_example") is not None


async def test_trace_tool_provider_registered_in_provider_registry(
    runtime: AppRuntime,
) -> None:
    """@tool_component(trace) must enter ProviderRegistry, otherwise YAML tools.providers raises LookupError."""

    assert runtime.bootstrapped
    assert ProviderRegistry.has("tool", "trace")
    # ComponentProvider only needs to be in the registry; an instance entering tool_manager also depends on
    # AGENT_EVAL_LANGFUSE_INSTANCES (empty means from_config returns None).


async def test_yaml_uses_symmetric_prefix_keys_and_templates(runtime: AppRuntime) -> None:
    """yaml agents/prompts fields match the conf/agents/{judge,attribution}_example directories."""

    agents_section = runtime.app_config.get_section("agents", AgentConfigSchema)
    assert "judge_example" in agents_section
    assert "attribution_example" in agents_section

    prompt_mgr = runtime.prompt_manager
    default_loader = getattr(prompt_mgr, "_default", None)
    base_dirs = getattr(default_loader, "base_dirs", ()) if default_loader is not None else ()
    rendered = [str(p) for p in base_dirs]
    assert any("judge_example/prompts" in p for p in rendered)
    assert any("attribution_example/prompts" in p for p in rendered)
    assert any("attribution_example/prompts" in p for p in rendered)
