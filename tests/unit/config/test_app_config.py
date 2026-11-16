"""Config-contract tests for conf/app.<env>.yaml (standard agentgraph sections).

The per-evaluator conf/<name>/evaluator.yaml is gone; evaluators are now declared as
``agents.<name>`` blocks in ``conf/app.<env>.yaml``, picking their LLM via
``default_model`` (→ ``models.<key>``) and prompt via ``prompt`` (→ ``prompts.template_dir``).
"""

import pytest
import yaml
from benchmark.paths import app_config_path, default_conf_dir

_ENVS = ["daily", "pre", "prod"]


def _load(env: str) -> dict:
    return yaml.safe_load(app_config_path(env).read_text(encoding="utf-8"))


def test_app_config_path_points_at_env_file():
    assert app_config_path("daily").name == "app.daily.yaml"
    assert app_config_path("pre") == default_conf_dir() / "app.pre.yaml"


@pytest.mark.parametrize("env", _ENVS)
def test_env_yaml_has_standard_sections(env):
    cfg = _load(env)
    assert cfg["discovery"]["scan_namespaces"] == ["app", "tools"]
    assert cfg["observability"]["app_name"] == "agent-eval"
    # prod defaults to WARNING (production does not flood INFO); daily/pre stay at INFO so issues are easier to trace
    expected_level = "WARNING" if env == "prod" else "INFO"
    assert cfg["observability"]["log_level"] == expected_level
    # agents section: judge_example and attribution_example evaluators reference their models + prompts
    judge_example = cfg["agents"]["judge_example"]
    assert judge_example["type"] == "judge_example"
    assert judge_example["default_model"] in cfg["models"]
    assert isinstance(judge_example["prompt"], str) and judge_example["prompt"]
    attribution_example = cfg["agents"]["attribution_example"]
    assert attribution_example["type"] == "attribution_example"
    assert attribution_example["default_model"] in cfg["models"]
    assert isinstance(attribution_example["badcase_prompt_name"], str)


@pytest.mark.parametrize("env", _ENVS)
def test_env_yaml_model_and_prompt_sections(env):
    cfg = _load(env)
    judge_example = cfg["agents"]["judge_example"]
    model = cfg["models"][judge_example["default_model"]]
    assert model["provider"] == "openai"
    assert model["api_key_env"]  # secret resolved from env, never committed
    assert cfg["prompts"]["template_dir"]
    attribution_example = cfg["agents"]["attribution_example"]
    attr_model = cfg["models"][attribution_example["default_model"]]
    assert attr_model["provider"] == "openai"


@pytest.mark.parametrize("env", _ENVS)
def test_env_yaml_uses_gpt55_openai_for_amap_agent_judge(env):
    cfg = _load(env)

    judge = cfg["agents"]["judge_amap_agent"]
    assert judge["type"] == "judge_amap_agent"
    assert judge["default_model"] == "judge_amap_agent"
    assert judge["core_completion_checkpoint_prompt"] == "core_completion_checkpoints"
    assert judge["core_completion_prompt"] == "core_completion"
    assert judge["expression_prompt"] == "expression"
    assert judge["weights"] == {"core_completion": 0.5, "expression": 0.5}
    assert "completion_checkpoint_prompt" not in judge
    assert "completion_prompt" not in judge
    assert "tool_call_checkpoint_prompt" not in judge
    assert "tool_call_prompt" not in judge
    assert cfg["models"][judge["default_model"]] == {
        "provider": "openai",
        "model_name": "gpt-5.5-0424-global",
        "base_url": "https://ai-llm-gateway.amap.com/open_api/v1",
        "api_key_env": "AGENT_EVAL_JUDGE_AMAP_AGENT_MODEL_API_KEY",
        "temperature": 1.0,
        "max_tokens": 4096,
        "timeout": 120,
    }


@pytest.mark.parametrize("env", _ENVS)
def test_env_yaml_uses_gpt55_openai_for_amap_agent_attribution(env):
    cfg = _load(env)

    attribution = cfg["agents"]["attribution_amap_agent"]
    assert attribution["type"] == "attribution_amap_agent"
    assert attribution["model"] == "attribution_amap_agent"
    assert cfg["models"][attribution["model"]] == {
        "provider": "openai",
        "model_name": "gpt-5.5-0424-global",
        "base_url": "https://ai-llm-gateway.amap.com/open_api/v1",
        "api_key_env": "AGENT_EVAL_JUDGE_AMAP_AGENT_MODEL_API_KEY",
        "temperature": 1.0,
        "max_tokens": 4096,
        "timeout": 300,
    }


@pytest.mark.parametrize("env", _ENVS)
def test_env_yaml_wires_competitor_pairwise_evaluator(env):
    cfg = _load(env)
    pairwise = cfg["agents"]["judge_competitor_pairwise"]

    assert pairwise["type"] == "judge_competitor_pairwise"
    assert pairwise["default_model"] == "judge_competitor_pairwise"
    assert pairwise["default_model"] in cfg["models"]
    assert pairwise["prompt"] == "pairwise"
    assert pairwise["focus_candidate"] == "gaode"
    assert pairwise["max_parallel_llm_calls"] >= 2
    assert any(
        path.endswith("judge_competitor_pairwise/prompts")
        for path in cfg["prompts"]["template_dir"]
    )

    model = cfg["models"][pairwise["default_model"]]
    assert model["provider"] == "openai"
    assert model["model_name"] == "qwen3.7-max"
    # Dedicated api_key (for cost allocation): must not be shared with the amap_agent primary judge (5223c08e)
    assert model["api_key_env"] == "AGENT_EVAL_COMPETITOR_PAIRWISE_API_KEY"
    assert "extra_params" not in model


@pytest.mark.parametrize("env", _ENVS)
def test_env_yaml_wires_memory_pairwise_evaluator(env):
    cfg = _load(env)
    pairwise = cfg["agents"]["judge_memory_pairwise"]

    assert pairwise["type"] == "judge_memory_pairwise"
    assert pairwise["default_model"] == "judge_memory_pairwise"
    assert pairwise["default_model"] in cfg["models"]
    assert pairwise["prompt"] == "memory_pairwise"
    assert pairwise["focus_candidate"] == "with_memory"

    model = cfg["models"][pairwise["default_model"]]
    assert model["provider"] == "anthropic"
    assert model["model_name"] == "claude-sonnet-4-6"
    # Dedicated api_key (for cost allocation): must be independent, and must not be shared with amap_agent/competitor.
    assert model["api_key_env"] == "AGENT_EVAL_JUDGE_MEMORY_PAIRWISE_MODEL_API_KEY"
    assert model["api_key_env"] != "AGENT_EVAL_JUDGE_AMAP_AGENT_MODEL_API_KEY"


@pytest.mark.parametrize("env", _ENVS)
def test_env_yaml_wires_memory_pairwise_attribution(env):
    """Pairwise attribution for judge_memory_pairwise: models / prompts / agents must be symmetric across the three environments."""
    cfg = _load(env)
    attribution = cfg["agents"]["attribution_memory_pairwise"]

    assert attribution["type"] == "attribution_memory_pairwise"
    assert attribution["default_model"] == "attribution_memory_pairwise"
    assert attribution["default_model"] in cfg["models"]
    assert attribution["attribution_prompt_name"] == "memory_attribution"
    assert 0.0 <= attribution["confidence_threshold"] <= 1.0
    assert any(
        path.endswith("attribution_memory_pairwise/prompts")
        for path in cfg["prompts"]["template_dir"]
    )

    model = cfg["models"][attribution["default_model"]]
    assert model["provider"] == "anthropic"
    assert model["model_name"] == "claude-sonnet-4-6"
    # The same dedicated key as the judge (same cost-allocation definition), not shared with example/amap_agent.
    assert model["api_key_env"] == "AGENT_EVAL_JUDGE_MEMORY_PAIRWISE_MODEL_API_KEY"


class TestStreamingEnabledBoolCoercion:
    """Lenient bool normalization of the streaming switch (regression lock for the 2026-09-28 crash-loop incident).

    Live BENCHMARK_STREAMING_ENABLED was mistyped as 'fasle'. pydantic bool parsing
    failed, Settings() raised ValidationError at import, and the service restarted every 5s. The normalization
    layer must fall typos back to a semantic value, and no misspelling may block process startup again.
    """

    def test_fasle_typo_coerced_to_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from benchmark.settings.setting import Settings

        monkeypatch.setenv("BENCHMARK_STREAMING_ENABLED", "fasle")
        s = Settings()
        assert s.benchmark_streaming_enabled is False

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("flase", False),
            ("fals", False),
            ("ture", True),
            ("yes", True),
            ("no", False),
            ("TRUE", True),
            ("False", False),
            ("1", True),
            ("0", False),
        ],
    )
    def test_bool_env_var_coercion_matrix(
        self,
        monkeypatch: pytest.MonkeyPatch,
        raw: str,
        expected: bool,
    ) -> None:
        from benchmark.settings.setting import Settings

        monkeypatch.setenv("BENCHMARK_STREAMING_ENABLED", raw)
        s = Settings()
        assert s.benchmark_streaming_enabled is expected

    def test_shadow_typo_coerced_to_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from benchmark.settings.setting import Settings

        monkeypatch.setenv("BENCHMARK_STREAMING_SHADOW", "flase")
        s = Settings()
        assert s.benchmark_streaming_shadow is False
