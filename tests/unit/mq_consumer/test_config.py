"""EvalMQConfig parsing + startup decision + backend projection unit tests (several sources, several topics)."""

from pathlib import Path

from benchmark.libs.mq import BackendType, Env

from benchmark.ingest.mq.runtime import (
    ConsumerTopicConfig,
    EvalMQConfig,
    load_mq_config,
    resolve_namesrv_domain,
    should_start_consumer,
    to_backend_configuration,
)

_YAML = """\
custom:
  mq_consumer:
    enabled: true
    worker_concurrency: 6
    worker_queue_size: 300
    retry_max_attempts: 3
    retry_initial_backoff_ms: 200
    consumers:
      luoshu:
        topic: "luoshu_agent_eval_pre"
        consumer_group: "CID-luoshu_agent_eval_pre"
      direct:
        topic: "direct_agent_eval_pre"
"""


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "app.daily.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_defaults() -> None:
    cfg = EvalMQConfig()
    assert cfg.enabled is True
    assert cfg.consumers == {}  # Empty by default; the real subscription comes from conf/app.<env>.yaml
    assert cfg.active_consumers() == []
    assert cfg.manager_group() == "CID-agent-eval"  # With no subscription, fall back to the service-level default
    assert cfg.worker_concurrency == 4
    assert cfg.worker_queue_size == 200
    assert cfg.retry_max_attempts == 0  # 0 = retry forever until success
    assert cfg.retry_initial_backoff_ms == 500
    assert cfg.retry_max_backoff_ms == 30000


def test_consumer_topic_config_group_derivation() -> None:
    # An empty consumer_group → derive CID-<topic> from the topic; an explicit config wins.
    assert ConsumerTopicConfig(topic="t").resolved_group() == "CID-t"
    assert ConsumerTopicConfig(topic="t", consumer_group="CID-x").resolved_group() == "CID-x"


def test_load_mq_config_from_yaml(tmp_path: Path) -> None:
    cfg = load_mq_config(_write_yaml(tmp_path, _YAML))
    assert set(cfg.consumers) == {"luoshu", "direct"}
    assert cfg.consumers["luoshu"].topic == "luoshu_agent_eval_pre"
    assert cfg.consumers["luoshu"].resolved_group() == "CID-luoshu_agent_eval_pre"
    # direct with no consumer_group configured → derive it from the topic.
    assert cfg.consumers["direct"].resolved_group() == "CID-direct_agent_eval_pre"
    # active_consumers is sorted by source name: direct < luoshu.
    assert [name for name, _ in cfg.active_consumers()] == ["direct", "luoshu"]
    assert cfg.worker_concurrency == 6
    assert cfg.worker_queue_size == 300
    assert cfg.retry_max_attempts == 3
    assert cfg.retry_initial_backoff_ms == 200


def test_active_consumers_skips_disabled() -> None:
    cfg = EvalMQConfig(
        consumers={
            "luoshu": ConsumerTopicConfig(topic="a"),
            "off": ConsumerTopicConfig(topic="b", enabled=False),
        }
    )
    assert [name for name, _ in cfg.active_consumers()] == ["luoshu"]


def test_load_mq_config_env_overrides(tmp_path: Path) -> None:
    cfg = load_mq_config(
        _write_yaml(tmp_path, _YAML),
        namesrv_env="http://custom-nsaddr",
        enabled_override=False,
    )
    assert cfg.enabled is False
    assert cfg.namesrv_domain == "http://custom-nsaddr"


def test_load_mq_config_missing_section_returns_defaults(tmp_path: Path) -> None:
    cfg = load_mq_config(_write_yaml(tmp_path, "custom: {}\n"))
    assert cfg == EvalMQConfig()


def test_resolve_namesrv_domain() -> None:
    explicit = EvalMQConfig(namesrv_domain="http://explicit")
    assert resolve_namesrv_domain(explicit, "daily") == "http://explicit"
    default = EvalMQConfig(namesrv_domain="")
    assert "nsaddr?nofix=1" in resolve_namesrv_domain(default, "daily")
    assert "nsaddr-sh?nofix=1" in resolve_namesrv_domain(default, "prod")


def test_to_backend_configuration_maps_env_and_extra() -> None:
    cfg = EvalMQConfig(
        consumers={"luoshu": ConsumerTopicConfig(topic="t", consumer_group="CID-x")},
        handler_timeout_s=90,
        max_buffer_size=500,
    )
    conf = to_backend_configuration(cfg, "daily")
    assert conf.env == Env.DAILY
    assert conf.backend == BackendType.METAQ
    # producer_group (only a backend placeholder in consumer-only mode) takes the group of the first enabled subscription.
    assert conf.extra["producer_group"] == "CID-x"
    assert conf.extra["ack_timeout"] == "90.0"
    assert conf.extra["max_buffer_size"] == "500"
    assert conf.extra["orderly_consume"] == "false"


def test_to_backend_configuration_unknown_env_is_local() -> None:
    conf = to_backend_configuration(EvalMQConfig(), "development")
    assert conf.env == Env.LOCAL  # → QueueManager uses InMemoryBackend


def test_should_start_consumer_disabled() -> None:
    cfg = EvalMQConfig(enabled=False, consumers={"luoshu": ConsumerTopicConfig(topic="t")})
    assert should_start_consumer(cfg) is False


def test_should_start_consumer_enabled_with_consumers() -> None:
    cfg = EvalMQConfig(enabled=True, consumers={"luoshu": ConsumerTopicConfig(topic="t")})
    assert should_start_consumer(cfg) is True


def test_should_start_consumer_enabled_but_no_consumers() -> None:
    # Enabled but with no enabled subscription configured → do not start (avoid an idle worker pool).
    assert should_start_consumer(EvalMQConfig(enabled=True)) is False
