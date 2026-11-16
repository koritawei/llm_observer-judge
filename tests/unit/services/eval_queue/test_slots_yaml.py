"""Unit tests for the queue_factory slots YAML."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmark.settings.node_identity import NodeIdentity
from benchmark.queue.queue_factory import (
    apply_slots_to_settings,
    load_queue_prefix_from_yaml,
    load_slots_from_yaml,
)

_CONF = Path(__file__).resolve().parents[4] / "conf"
_GENERIC_IDENTITY = NodeIdentity(hostname="ci-host", host_ips=frozenset(), idc="ci")
_NODE_SLOT_YAML = """
custom:
  bullmq:
    queue_prefix: pre
    slots:
      sync:
        worker_concurrency: 3
        slots_per_run_per_node: 3
      external:
        worker_concurrency: 3
        slots_per_run_per_node: 3
        cluster_aone_soft_cap: 6
        callback_timeout_ms: 600000
        attr_worker_concurrency: 3
        attr_slots_per_run_per_node: 3
        attr_cluster_soft_cap: 6
    node_slots:
      - match:
          hostname: pinned-host.pre.na610
          host_ip: 10.1.2.3
          idc: na610
        queue_prefix: pre-pinned
        slots:
          sync:
            worker_concurrency: 8
            slots_per_run_per_node: 4
          external:
            worker_concurrency: 8
            slots_per_run_per_node: 4
            cluster_aone_soft_cap: 8
            attr_worker_concurrency: 8
            attr_slots_per_run_per_node: 4
            attr_cluster_soft_cap: 8
"""


def _blank_settings() -> SimpleNamespace:
    return SimpleNamespace(
        eval_queue_worker_concurrency=5,
        sync_slots_per_run_per_node=2,
        sync_slot_busy_delay_ms=3000,
        sync_slot_heal_interval_s=60,
        sync_retry_max_duration_ms=7_200_000,
        max_slot_probes_per_run=2,
        external_worker_concurrency=2,
        external_slots_per_run_per_node=2,
        external_cluster_notification_soft_cap=4,
        external_slot_busy_delay_ms=3000,
        external_slot_heal_interval_s=60,
        external_retry_max_duration_ms=43_200_000,
        external_run_max_duration_ms=43_200_000,
        external_callback_concurrency=20,
        external_local_attr_concurrency=1,
        attr_worker_concurrency=2,
        attr_slots_per_run_per_node=2,
        attr_cluster_soft_cap=4,
        attr_callback_timeout_ms=86_400_000,
        attr_callback_url="",
        external_running_ttl_ms=86_400_000,
        external_callback_timeout_ms=86_400_000,
    )


@pytest.mark.parametrize(
    "row",
    [
        # Matches custom.bullmq.slots in conf/app.<env>.yaml;
        # the daily judge slot limit is 1 (serial within one run); pre is 3 ways per run;
        # prod sync=40×2; external/attr: 200 runs × 4 ways within one run (worker/cluster=800).
        ("daily", "daily", 3_600_000, 7_200_000, 3, 2, 1, 4, 1, 2, 4, 2),
        ("pre", "pre", 7_200_000, 7_200_000, 3, 3, 3, 6, 3, 3, 6, 3),
        ("prod", "prod", 7_200_000, 7_200_000, 40, 800, 4, 800, 2, 4, 800, 800),
    ],
)
def test_yaml_bullmq_matrix_matches_design(row: tuple) -> None:
    (
        env,
        prefix,
        sync_retry,
        ext_retry,
        sync_conc,
        ext_conc,
        ext_slots,
        ext_soft_cap,
        sync_slots,
        attr_slots,
        attr_soft_cap,
        attr_conc,
    ) = row
    yaml_path = _CONF / f"app.{env}.yaml"
    assert yaml_path.is_file()

    assert load_queue_prefix_from_yaml(yaml_path, identity=_GENERIC_IDENTITY) == prefix
    slots = load_slots_from_yaml(yaml_path, identity=_GENERIC_IDENTITY)
    assert slots, f"custom.bullmq.slots missing in {yaml_path}"

    settings = _blank_settings()
    apply_slots_to_settings(settings, slots)

    assert settings.eval_queue_worker_concurrency == sync_conc
    assert settings.sync_retry_max_duration_ms == sync_retry
    assert settings.external_retry_max_duration_ms == ext_retry
    assert settings.external_run_max_duration_ms == 86_400_000
    assert settings.sync_slots_per_run_per_node == sync_slots
    assert settings.external_slots_per_run_per_node == ext_slots
    assert settings.external_cluster_notification_soft_cap == ext_soft_cap
    assert settings.max_slot_probes_per_run == 2
    assert settings.external_worker_concurrency == ext_conc
    assert settings.attr_slots_per_run_per_node == attr_slots
    assert settings.attr_cluster_soft_cap == attr_soft_cap
    assert settings.attr_worker_concurrency == attr_conc
    if env == "prod":
        assert settings.external_cluster_notification_soft_cap > 0
    assert settings.attr_callback_timeout_ms == 600_000
    assert settings.external_callback_timeout_ms == 600_000
    assert settings.attr_callback_url == ("http://agent-eval/v1/benchmarks/callback")


def test_apply_slots_maps_external_worker_concurrency() -> None:
    settings = _blank_settings()
    apply_slots_to_settings(
        settings,
        {
            "external": {
                "worker_concurrency": 4,
                "retry_max_duration_ms": 3_600_000,
            },
        },
    )
    assert settings.external_worker_concurrency == 4
    assert settings.external_retry_max_duration_ms == 3_600_000
    # A field that does not appear keeps its default
    assert settings.eval_queue_worker_concurrency == 5
    assert settings.external_run_max_duration_ms == 43_200_000


def test_load_yaml_missing_file_falls_back_to_defaults(tmp_path: Path) -> None:
    """The config file is missing (the local-development case) → warning + an empty fallback, and do not raise."""
    missing = tmp_path / "app.none.yaml"
    assert load_queue_prefix_from_yaml(missing) == ""
    assert load_slots_from_yaml(missing) == {}


def test_load_yaml_bad_format_raises_runtime_error(tmp_path: Path) -> None:
    """Malformed YAML → RuntimeError with the path (fail-fast; a silent fallback to the default concurrency is forbidden)."""
    bad = tmp_path / "app.bad.yaml"
    bad.write_text("custom: [unclosed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="app.bad.yaml"):
        load_slots_from_yaml(bad)


def test_load_yaml_unreadable_raises_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Insufficient read permission (OSError) → RuntimeError with the path."""

    def _deny_open(*args: object, **kwargs: object) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "open", _deny_open)
    with pytest.raises(RuntimeError, match="读取配置文件失败"):
        load_queue_prefix_from_yaml(tmp_path / "app.daily.yaml")


@pytest.mark.parametrize(
    ("bad_value", "yaml_key"),
    [
        ("abc", "worker_concurrency"),
        (None, "slots_per_run_per_node"),
        (True, "slot_busy_delay_ms"),
    ],
)
def test_apply_slots_rejects_bad_int_value(bad_value: object, yaml_key: str) -> None:
    """A bad value for a slots int field (string/None/bool) → ValueError with the field path."""
    settings = _blank_settings()
    with pytest.raises(ValueError, match=f"slots.sync.{yaml_key}"):
        apply_slots_to_settings(settings, {"sync": {yaml_key: bad_value}})


def test_apply_slots_rejects_bad_attr_callback_timeout() -> None:
    """A bad value on the attr_callback_timeout_ms fallback branch also fails fast."""
    settings = _blank_settings()
    with pytest.raises(ValueError, match="slots.external.callback_timeout_ms"):
        apply_slots_to_settings(settings, {"external": {"callback_timeout_ms": "soon"}})


def test_pre_yaml_pins_pre_agent_eval_for_target_node() -> None:
    """The named node (service011171024115 / 11.171.24.115) exclusively owns the pre_agent-eval prefix;

    every other node still uses the generic pre; slots are not overridden, and the generic slots are kept.
    """
    yaml_path = _CONF / "app.pre.yaml"
    pinned = NodeIdentity(
        hostname="agent-eval-service011171024115.pre.na620",
        host_ips=frozenset({"11.171.24.115"}),
        idc="na620",
    )
    assert load_queue_prefix_from_yaml(yaml_path, identity=pinned) == "pre_agent-eval"
    slots = load_slots_from_yaml(yaml_path, identity=pinned)
    settings = _blank_settings()
    apply_slots_to_settings(settings, slots)
    # node slots are not written → keep the generic slots
    assert settings.eval_queue_worker_concurrency == 3
    assert settings.external_worker_concurrency == 3
    assert settings.external_cluster_notification_soft_cap == 6

    # Other nodes (including the old pinned-host address): the generic prefix + the generic slots
    other = NodeIdentity(
        hostname="agent-eval-service033084102185.pre.na610",
        host_ips=frozenset({"33.84.102.185"}),
        idc="na610",
    )
    assert load_queue_prefix_from_yaml(yaml_path, identity=other) == "pre"
    other_slots = load_slots_from_yaml(yaml_path, identity=other)
    other_settings = _blank_settings()
    apply_slots_to_settings(other_settings, other_slots)
    assert other_settings.eval_queue_worker_concurrency == 3


def test_node_slots_applied_when_hostname_ip_idc_match(tmp_path: Path) -> None:
    """All three node_slots fields match → override the queue prefix and merge the slots, and an unwritten field keeps the generic value."""
    yaml_path = tmp_path / "app.pre.yaml"
    yaml_path.write_text(_NODE_SLOT_YAML, encoding="utf-8")
    identity = NodeIdentity(
        hostname="pinned-host.pre.na610",
        host_ips=frozenset({"10.1.2.3"}),
        idc="na610",
    )
    assert load_queue_prefix_from_yaml(yaml_path, identity=identity) == "pre-pinned"
    slots = load_slots_from_yaml(yaml_path, identity=identity)
    settings = _blank_settings()
    apply_slots_to_settings(settings, slots)
    assert settings.eval_queue_worker_concurrency == 8
    assert settings.sync_slots_per_run_per_node == 4
    assert settings.external_worker_concurrency == 8
    assert settings.external_slots_per_run_per_node == 4
    assert settings.external_cluster_notification_soft_cap == 8
    assert settings.attr_worker_concurrency == 8
    assert settings.attr_slots_per_run_per_node == 4
    assert settings.attr_cluster_soft_cap == 8
    assert settings.external_callback_timeout_ms == 600_000


@pytest.mark.parametrize(
    "identity",
    [
        NodeIdentity(
            hostname="other-host.pre.na610",
            host_ips=frozenset({"10.1.2.3"}),
            idc="na610",
        ),
        NodeIdentity(
            hostname="pinned-host.pre.na610",
            host_ips=frozenset({"10.0.0.1"}),
            idc="na610",
        ),
        NodeIdentity(
            hostname="pinned-host.pre.na610",
            host_ips=frozenset({"10.1.2.3"}),
            idc="ea119",
        ),
    ],
)
def test_node_slots_skipped_when_any_match_field_differs(
    identity: NodeIdentity,
    tmp_path: Path,
) -> None:
    """Any one of hostname / IP / site does not match → still use the generic prefix + slots."""
    yaml_path = tmp_path / "app.pre.yaml"
    yaml_path.write_text(_NODE_SLOT_YAML, encoding="utf-8")
    assert load_queue_prefix_from_yaml(yaml_path, identity=identity) == "pre"
    slots = load_slots_from_yaml(yaml_path, identity=identity)
    settings = _blank_settings()
    apply_slots_to_settings(settings, slots)
    assert settings.eval_queue_worker_concurrency == 3
    assert settings.sync_slots_per_run_per_node == 3
    assert settings.external_worker_concurrency == 3
    assert settings.external_slots_per_run_per_node == 3
    assert settings.external_cluster_notification_soft_cap == 6
    assert settings.attr_worker_concurrency == 3
