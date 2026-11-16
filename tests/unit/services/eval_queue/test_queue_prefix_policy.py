"""Unit tests for the BullMQ queue_prefix ownership policy (driven by envSign / SIGMA)."""

from pathlib import Path

import pytest
from benchmark.paths import default_conf_dir
from benchmark.queue.queue_prefix_policy import (
    RESERVED_QUEUE_PREFIXES,
    QueuePrefixPolicyError,
    QueuePrefixResolveInput,
    RuntimeLaneDetectInput,
    assert_committed_standard_yaml_prefixes,
    build_bullmq_logic_prefix,
    detect_runtime_lane,
    has_platform_lane_signal,
    is_reserved_queue_prefix,
    is_temporary_env_sign,
    resolve_queue_prefix,
    uses_ambiguous_standard_profile_fallback,
)

_CONF = Path(__file__).resolve().parents[4] / "conf"
_TEMP_SIGN = "aone-20221028-project-agent-eval-demo"


def _resolve(
    *,
    deploy_profile: str,
    yaml_prefix: str,
    env_prefix: str,
    env_sign: str = "",
    sigma_app_stage: str = "",
    deploy_kind: str = "",
) -> str:
    return resolve_queue_prefix(
        QueuePrefixResolveInput(
            deploy_profile=deploy_profile,
            yaml_prefix=yaml_prefix,
            env_prefix=env_prefix,
            env_sign=env_sign,
            sigma_app_stage=sigma_app_stage,
            deploy_kind=deploy_kind,
        )
    )


def test_committed_standard_yaml_prefixes_locked() -> None:
    assert_committed_standard_yaml_prefixes(_CONF)
    assert_committed_standard_yaml_prefixes(default_conf_dir())


def test_temporary_env_sign_length_rule() -> None:
    assert not is_temporary_env_sign("staging")
    assert not is_temporary_env_sign("production")
    assert is_temporary_env_sign(_TEMP_SIGN)
    assert len(_TEMP_SIGN) > 20


@pytest.mark.parametrize(
    ("env_sign", "expected"),
    [
        ("testing", "daily"),
        ("staging", "pre"),
        ("gray", "pre"),
        ("production", "prod"),
        ("STAGING", "pre"),
    ],
)
def test_standard_env_sign_forces_fixed_prefix(env_sign: str, expected: str) -> None:
    assert (
        _resolve(
            deploy_profile="pre",
            yaml_prefix=expected,
            env_prefix="dev-alice",
            env_sign=env_sign,
        )
        == expected
    )


def test_standard_preserves_node_slots_derived_yaml_prefix() -> None:
    assert (
        _resolve(
            deploy_profile="pre",
            yaml_prefix="pre_agent-eval",
            env_prefix="dev-should-not-win",
            env_sign="staging",
        )
        == "pre_agent-eval"
    )


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("DAILY", "daily"),
        ("PRE_PUBLISH", "pre"),
        ("GRAY", "pre"),
        ("SMALLFLOW", "pre"),
        ("PUBLISH", "prod"),
    ],
)
def test_sigma_stage_forces_fixed_prefix(stage: str, expected: str) -> None:
    assert (
        _resolve(
            deploy_profile="daily",
            yaml_prefix=expected,
            env_prefix="feat-x",
            env_sign="",
            sigma_app_stage=stage,
        )
        == expected
    )


def test_standard_overrides_business_prefix_change() -> None:
    """Standard pre-release/production: a business change to EVAL_QUEUE_PREFIX is overwritten back to the fixed value."""
    assert (
        _resolve(
            deploy_profile="prod",
            yaml_prefix="prod",
            env_prefix="dev-pollute-prod",
            env_sign="production",
        )
        == "prod"
    )
    assert (
        _resolve(
            deploy_profile="pre",
            yaml_prefix="pre",
            env_prefix="feat-search",
            env_sign="staging",
        )
        == "pre"
    )


def test_temporary_env_requires_custom_prefix() -> None:
    with pytest.raises(QueuePrefixPolicyError, match="禁止使用标准默认前缀"):
        _resolve(
            deploy_profile="pre",
            yaml_prefix="pre",
            env_prefix="",
            env_sign=_TEMP_SIGN,
        )


def test_temporary_env_rejects_reserved_env_prefix() -> None:
    with pytest.raises(QueuePrefixPolicyError, match="禁止使用保留前缀"):
        _resolve(
            deploy_profile="pre",
            yaml_prefix="pre",
            env_prefix="pre",
            env_sign=_TEMP_SIGN,
        )


def test_temporary_env_accepts_custom_prefix() -> None:
    assert (
        _resolve(
            deploy_profile="pre",
            yaml_prefix="pre",
            env_prefix="dev-alice",
            env_sign=_TEMP_SIGN,
        )
        == "dev-alice"
    )


def test_fallback_without_platform_signal_uses_agent_eval_env() -> None:
    assert detect_runtime_lane(RuntimeLaneDetectInput(deploy_profile="prod")) == (
        "standard",
        "prod",
    )
    assert (
        _resolve(
            deploy_profile="prod",
            yaml_prefix="prod",
            env_prefix="dev-x",
        )
        == "prod"
    )
    assert detect_runtime_lane(RuntimeLaneDetectInput(deploy_profile="feature-x")) == (
        "temporary",
        None,
    )


def test_ambiguous_standard_profile_fallback_detection() -> None:
    assert uses_ambiguous_standard_profile_fallback(RuntimeLaneDetectInput(deploy_profile="daily"))
    assert uses_ambiguous_standard_profile_fallback(RuntimeLaneDetectInput(deploy_profile="prod"))
    assert not uses_ambiguous_standard_profile_fallback(
        RuntimeLaneDetectInput(deploy_profile="daily", env_sign="testing")
    )
    assert not uses_ambiguous_standard_profile_fallback(
        RuntimeLaneDetectInput(deploy_profile="daily", sigma_app_stage="DAILY")
    )
    assert not uses_ambiguous_standard_profile_fallback(
        RuntimeLaneDetectInput(deploy_profile="daily", deploy_kind="local")
    )
    assert not uses_ambiguous_standard_profile_fallback(
        RuntimeLaneDetectInput(deploy_profile="feature-x")
    )
    assert not has_platform_lane_signal(env_sign="", sigma_app_stage="")
    assert has_platform_lane_signal(env_sign="staging", sigma_app_stage="")
    assert has_platform_lane_signal(env_sign="", sigma_app_stage="PUBLISH")


def test_assert_committed_only_current_profile_and_missing_warn(tmp_path: Path) -> None:
    (tmp_path / "app.pre.yaml").write_text(
        "custom:\n  bullmq:\n    queue_prefix: pre\n",
        encoding="utf-8",
    )
    warnings: list[str] = []
    # Check only the current profile; a missing daily/prod is not an error
    assert_committed_standard_yaml_prefixes(
        tmp_path,
        profiles=["pre"],
        missing="warn",
        warn=warnings.append,
    )
    assert warnings == []

    warnings.clear()
    assert_committed_standard_yaml_prefixes(
        tmp_path,
        profiles=["daily"],
        missing="warn",
        warn=warnings.append,
    )
    assert len(warnings) == 1
    assert "app.daily.yaml" in warnings[0]


def test_personal_deploy_kind_forces_temporary() -> None:
    with pytest.raises(QueuePrefixPolicyError, match="禁止使用标准默认前缀"):
        _resolve(
            deploy_profile="pre",
            yaml_prefix="pre",
            env_prefix="",
            env_sign="staging",
            deploy_kind="personal",
        )


def test_assert_committed_detects_tampered_prefix(tmp_path: Path) -> None:
    path = tmp_path / "app.pre.yaml"
    path.write_text(
        "custom:\n  bullmq:\n    queue_prefix: dev-pollute\n",
        encoding="utf-8",
    )
    (tmp_path / "app.daily.yaml").write_text(
        "custom:\n  bullmq:\n    queue_prefix: daily\n",
        encoding="utf-8",
    )
    (tmp_path / "app.prod.yaml").write_text(
        "custom:\n  bullmq:\n    queue_prefix: prod\n",
        encoding="utf-8",
    )
    with pytest.raises(QueuePrefixPolicyError, match="必须为 'pre'"):
        assert_committed_standard_yaml_prefixes(tmp_path)


def test_keyspace_generation_does_not_change_lane_business_rules() -> None:
    """The generation is joined only onto the Redis logical prefix; a bare-lane resolve / reserved-word restriction stays as it was."""
    assert RESERVED_QUEUE_PREFIXES == frozenset({"daily", "pre", "prod"})
    assert is_reserved_queue_prefix("pre")
    assert is_reserved_queue_prefix("prod")
    assert is_reserved_queue_prefix("daily")
    assert not is_reserved_queue_prefix("dev-alice")
    # The logical prefix includes the generation, and is not a bare reserved word
    assert not is_reserved_queue_prefix(build_bullmq_logic_prefix("pre"))

    bare_pre = _resolve(
        deploy_profile="pre",
        yaml_prefix="pre",
        env_prefix="feat-should-lose",
        env_sign="staging",
    )
    bare_iso = _resolve(
        deploy_profile="pre",
        yaml_prefix="pre",
        env_prefix="dev-alice",
        env_sign=_TEMP_SIGN,
    )
    assert bare_pre == "pre"
    assert bare_iso == "dev-alice"
    assert build_bullmq_logic_prefix(bare_pre) == "pre:bullmq:v2"
    assert build_bullmq_logic_prefix(bare_iso) == "dev-alice:bullmq:v2"
    with pytest.raises(QueuePrefixPolicyError, match="禁止使用保留前缀"):
        _resolve(
            deploy_profile="pre",
            yaml_prefix="pre",
            env_prefix="prod",
            env_sign=_TEMP_SIGN,
        )
