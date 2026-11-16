"""Pure-function unit tests for odps_export (SQL rendering / CSV / object key / T-1 vs T)."""

import csv
import io
import sys
from datetime import datetime
from types import ModuleType, SimpleNamespace
from typing import Any

import benchmark.reporting.framework.odps_export as odps_export_module
import pytest
from benchmark.reporting.framework.odps_export import (
    DEFAULT_PROJECT_ID,
    DEFAULT_SQL_TEMPLATE,
    OdpsExportEmptyError,
    OdpsExportError,
    build_overview_probe_sqls,
    default_ds,
    default_odps_ds,
    default_oss_ds,
    diagnose_probe_results,
    execute_query,
    render_sql,
    resolve_object_key,
    rows_to_csv_bytes,
    shift_ds,
)


class TestDefaultDates:
    def test_odps_is_t_minus_1(self) -> None:
        now = datetime(2026, 9, 9, 6, 0, 0)
        assert default_odps_ds(now) == "20260908"
        assert default_ds(now) == "20260908"

    def test_oss_is_t(self) -> None:
        assert default_oss_ds(datetime(2026, 9, 9, 6, 0, 0)) == "20260909"

    def test_month_boundary(self) -> None:
        now = datetime(2026, 9, 1, 0, 30, 0)
        assert default_odps_ds(now) == "20260831"
        assert default_oss_ds(now) == "20260901"

    def test_shift_ds(self) -> None:
        assert shift_ds("20260908", days=1) == "20260909"
        assert shift_ds("20260909", days=-1) == "20260908"


class TestRenderSql:
    def test_ds_and_limit_injected(self) -> None:
        template = "SELECT * FROM t WHERE ds = '{ds}' ORDER BY RAND() LIMIT {limit}"
        sql = render_sql(template, "20260908", 200)
        assert "ds = '20260908'" in sql
        assert "LIMIT 200" in sql
        assert "{ds}" not in sql and "{limit}" not in sql

    def test_default_template_injects_ds_and_limit(self) -> None:
        # The built-in template contains {ds}/{limit}/{project_id}, so a --odps-ds / --oss-ds backfill can inject the partition
        sql = render_sql(
            DEFAULT_SQL_TEMPLATE,
            "20260908",
            200,
            project_id=DEFAULT_PROJECT_ID,
        )
        assert "ds = '20260908'" in sql
        assert "LIMIT   200" in sql or "LIMIT 200" in sql
        assert "{ds}" not in sql and "{limit}" not in sql
        assert "{project_id}" not in sql
        assert (
            f"GET_JSON_OBJECT(content_json, '$.record.project_id') = '{DEFAULT_PROJECT_ID}'"
        ) in sql
        assert "GET_JSON_OBJECT(content_json, '$.record.project_id') AS project" in sql
        assert f"'{DEFAULT_PROJECT_ID}' AS project" not in sql
        assert "GETDATE()" not in sql
        assert "search_agent.stream.overview" in sql
        assert "autonavi_ai.dwd_gd_ai_chat_langfuse_trace_di" in sql
        # JOIN-filter first, then sample with ORDER BY RAND(), so the final row count is guaranteed
        assert sql.index("FROM    joined") < sql.index("ORDER BY RAND()")

    def test_project_id_injected_when_placeholder_present(self) -> None:
        template = "SELECT '{project_id}' AS project WHERE ds = '{ds}' LIMIT {limit}"
        sql = render_sql(template, "20260908", 10, project_id="proj_abc-1")
        assert "'proj_abc-1' AS project" in sql
        assert "{project_id}" not in sql

    def test_project_id_optional_when_placeholder_absent(self) -> None:
        template = "SELECT * FROM t WHERE ds = '{ds}' LIMIT {limit}"
        sql = render_sql(template, "20260908", 10)
        assert "{project_id}" not in sql
        assert "ds = '20260908'" in sql

    def test_project_id_required_when_placeholder_present(self) -> None:
        template = "SELECT '{project_id}' AS project WHERE ds = '{ds}' LIMIT {limit}"
        with pytest.raises(ValueError, match=r"\{project_id\}"):
            render_sql(template, "20260908", 10)

    @pytest.mark.parametrize("bad", ["'; DROP TABLE x; --", "a b", "proj'id", ""])
    def test_reject_bad_project_id(self, bad: str) -> None:
        template = "SELECT '{project_id}' AS project WHERE ds = '{ds}' LIMIT {limit}"
        with pytest.raises(ValueError, match="project_id"):
            render_sql(template, "20260908", 10, project_id=bad)

    def test_self_contained_template_passthrough(self) -> None:
        template = "SELECT 1 FROM t WHERE ds = TO_CHAR(DATEADD(GETDATE(), -1, 'dd'), 'yyyymmdd')"
        assert render_sql(template, "20260908", 200) == template

    @pytest.mark.parametrize(
        "bad",
        ["2026097", "2026-09-07", "202609070", "abcd0201", "'; DROP TABLE x; --"],
    )
    def test_reject_bad_ds(self, bad: str) -> None:
        with pytest.raises(ValueError, match="YYYYMMDD"):
            render_sql(DEFAULT_SQL_TEMPLATE, bad, 200)

    @pytest.mark.parametrize("bad_limit", [0, -1, 10001])
    def test_reject_bad_limit(self, bad_limit: int) -> None:
        with pytest.raises(ValueError, match="limit"):
            render_sql(DEFAULT_SQL_TEMPLATE, "20260908", bad_limit)

    @pytest.mark.parametrize(
        "template",
        [
            "SELECT 1 WHERE ds = '{ds}'",  # Only {ds}
            "SELECT 1 LIMIT {limit}",  # Only {limit}
        ],
    )
    def test_single_placeholder_rejected(self, template: str) -> None:
        # Placeholders must appear in pairs: a half replacement would run against the wrong partition or scan the whole table
        with pytest.raises(ValueError, match="占位符"):
            render_sql(template, "20260908", 10)


class TestRowsToCsvBytes:
    def test_header_bom_and_full_quoting(self) -> None:
        payload = rows_to_csv_bytes(
            ["query", "trace_id"],
            [["你好,高德", "t1"], ['含"引号"', "t2"]],
        )
        assert payload.startswith(b"\xef\xbb\xbf")
        text = payload.decode("utf-8-sig")
        for line in text.splitlines():
            assert line.startswith('"'), line
        rows = list(csv.reader(io.StringIO(text)))
        assert rows[0] == ["query", "trace_id"]
        assert rows[1] == ["你好,高德", "t1"]
        assert rows[2] == ['含"引号"', "t2"]

    def test_none_and_non_str_values(self) -> None:
        payload = rows_to_csv_bytes(["a", "b"], [[None, 106.5]])
        rows = list(csv.reader(io.StringIO(payload.decode("utf-8-sig"))))
        assert rows[1] == ["", "106.5"]

    def test_embedded_newline_roundtrip(self) -> None:
        payload = rows_to_csv_bytes(["r"], [["line1\nline2"]])
        rows = list(csv.reader(io.StringIO(payload.decode("utf-8-sig"))))
        assert rows[1] == ["line1\nline2"]


class TestResolveObjectKey:
    def test_uses_oss_ds_not_odps(self) -> None:
        # OSS uses T=20260909, even when ODPS is T-1=20260908
        assert resolve_object_key("20260909", "agent-eval/search-agent-overview") == (
            "agent-eval/search-agent-overview/20260909/benchmark_final_results.csv"
        )

    def test_prefix_slashes_trimmed(self) -> None:
        assert resolve_object_key("20260909", "/a/b/") == "a/b/20260909/benchmark_final_results.csv"

    def test_object_key_override(self) -> None:
        assert resolve_object_key("20260909", "x", object_key="/custom/key.csv") == "custom/key.csv"

    def test_custom_object_name(self) -> None:
        assert (
            resolve_object_key(
                "20260909",
                "pref",
                object_name="daily.csv",
            )
            == "pref/20260909/daily.csv"
        )

    def test_date_macro_expands_without_extra_oss_ds(self) -> None:
        assert resolve_object_key(
            "20261103",
            "search-eval/benchmark_result/${yyyymmdd}",
        ) == ("search-eval/benchmark_result/20261103/benchmark_final_results.csv")
        assert resolve_object_key(
            "20261103",
            "search-eval/benchmark_result/${yyyymmdd-1}",
        ) == ("search-eval/benchmark_result/20261102/benchmark_final_results.csv")

    def test_already_dated_prefix_not_double_appended(self) -> None:
        assert resolve_object_key(
            "20261103",
            "search-eval/benchmark_result/20261102",
        ) == ("search-eval/benchmark_result/20261102/benchmark_final_results.csv")


class _FakeColumn:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeRecord:
    def __init__(self, values: list[Any]) -> None:
        self.values = values


class _FakeReader:
    schema = SimpleNamespace(columns=[_FakeColumn("query"), _FakeColumn("trace_id")])

    def __iter__(self) -> Any:
        return iter([_FakeRecord(["q1", "t1"])])

    def __enter__(self) -> "_FakeReader":
        return self

    def __exit__(self, *args: Any) -> None:
        return None


class _FakeInstance:
    def __init__(self) -> None:
        self.open_reader_kwargs: dict[str, Any] = {}
        self.id = "fake-instance-1"
        self.status = "Terminated"
        self._waited = False

    def wait_for_success(self) -> None:
        self._waited = True

    def get_logview_address(self) -> str:
        return "https://logview.example/fake"

    def open_reader(self, *args: Any, **kwargs: Any) -> _FakeReader:
        # Aligned with production execute_query(tunnel=True, limit=False); record the arguments for assertions.
        _ = args
        self.open_reader_kwargs = dict(kwargs)
        return _FakeReader()


def _fake_sdk_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    provider_error: Exception | None = None,
) -> dict[str, Any]:
    """Inject fake credential/extension/odps modules (the intranet SDK is not in the test environment)."""
    calls: dict[str, Any] = {}
    provider = object()

    credential_mod = ModuleType("credential")
    credential_mod.names = SimpleNamespace(  # type: ignore[attr-defined]
        of_odps_account_notification_app=lambda app: f"aurn::odps-app::{app}",
    )

    def _get_provider(urn: str) -> object:
        calls["urn"] = urn
        if provider_error is not None:
            raise provider_error
        return provider

    extension_mod = ModuleType("extension")
    extension_mod.factory = SimpleNamespace(  # type: ignore[attr-defined]
        get_acs_credential_provider=_get_provider,
    )

    class _Account:
        def __init__(self, p: object) -> None:
            calls["account_provider"] = p

    class _ODPS:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls["odps_init"] = {"args": args, "kwargs": kwargs}

        def execute_sql(self, sql: str) -> _FakeInstance:
            calls["sql"] = sql
            instance = _FakeInstance()
            calls["instance"] = instance
            return instance

    odps_mod = ModuleType("odps")
    odps_mod.ODPS = _ODPS  # type: ignore[attr-defined]
    odps_accounts_mod = ModuleType("odps.accounts")
    odps_accounts_mod.CredentialProviderAccount = _Account  # type: ignore[attr-defined]
    odps_mod.accounts = odps_accounts_mod  # type: ignore[attr-defined]

    for name, mod in (
        ("credential", credential_mod),
        ("extension", extension_mod),
        ("odps", odps_mod),
        ("odps.accounts", odps_accounts_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


def _patch_settings(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> None:
    base = {
        "odps_notification_app_name": "odps-app-x",
        "odps_endpoint": "http://odps-endpoint/api",
        "odps_project": "autonavi_ai",
    }
    base.update(overrides)
    monkeypatch.setattr(odps_export_module, "settings", SimpleNamespace(**base))


class TestExecuteQuery:
    """Normandy credential wiring for execute_query (the SDK is injected as a fake module, and the call chain is what is checked)."""

    def test_uses_normandy_credential_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _fake_sdk_modules(monkeypatch)
        _patch_settings(monkeypatch)

        columns, rows = execute_query("SELECT 1")

        assert columns == ["query", "trace_id"]
        assert rows == [["q1", "t1"]]
        # The URN is generated from the aone app name; the provider is injected into ODPS through CredentialProviderAccount
        assert calls["urn"] == "aurn::odps-app::odps-app-x"
        assert calls["account_provider"] is not None
        # ODPS no longer receives positional AK/SK arguments, only account/project/endpoint
        init = calls["odps_init"]
        assert init["args"] == ()
        assert init["kwargs"]["project"] == "autonavi_ai"
        assert init["kwargs"]["endpoint"] == "http://odps-endpoint/api"
        assert calls["sql"] == "SELECT 1"
        instance = calls["instance"]
        assert isinstance(instance, _FakeInstance)
        assert instance.open_reader_kwargs == {
            "tunnel": True,
            "limit": False,
        }
        assert instance._waited is True

    def test_missing_project_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_sdk_modules(monkeypatch)
        _patch_settings(monkeypatch, odps_project="")

        with pytest.raises(OdpsExportError, match="AGENT_EVAL_ODPS_PROJECT"):
            execute_query("SELECT 1")

    def test_empty_reader_raises_with_diagnostics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _fake_sdk_modules(monkeypatch)
        _patch_settings(monkeypatch)

        class _EmptyReader:
            schema = SimpleNamespace(columns=[_FakeColumn("query")])
            count = 0

            def __iter__(self) -> Any:
                return iter([])

            def __enter__(self) -> "_EmptyReader":
                return self

            def __exit__(self, *args: Any) -> None:
                return None

        class _EmptyODPS:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                calls["odps_init"] = {"args": args, "kwargs": kwargs}

            def execute_sql(self, sql: str) -> _FakeInstance:
                calls["sql"] = sql
                instance = _FakeInstance()

                def _open(*a: Any, **kw: Any) -> _EmptyReader:
                    _ = a
                    instance.open_reader_kwargs = dict(kw)
                    return _EmptyReader()

                instance.open_reader = _open  # type: ignore[method-assign]
                calls["instance"] = instance
                return instance

        import benchmark.libs.warehouse as odps_mod  # noqa: PLC0415

        odps_mod.ODPS = _EmptyODPS  # type: ignore[misc, assignment]

        with pytest.raises(OdpsExportEmptyError, match="reader_mode="):
            execute_query("SELECT 1")

    def test_missing_notification_app_name_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_sdk_modules(monkeypatch)
        _patch_settings(monkeypatch, odps_notification_app_name="")

        with pytest.raises(OdpsExportError, match="AGENT_EVAL_ODPS_NOTIFICATION_APP_NAME"):
            execute_query("SELECT 1")

    def test_provider_failure_hint_resource_binding(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_sdk_modules(monkeypatch, provider_error=RuntimeError("no available alias"))
        _patch_settings(monkeypatch)

        with pytest.raises(OdpsExportError, match="Normandy ODPS 凭证获取失败"):
            execute_query("SELECT 1")

    def test_sdk_not_installed_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Setting sys.modules to None makes the import raise ImportError deterministically
        for name in ("credential", "extension"):
            monkeypatch.setitem(sys.modules, name, None)
        _patch_settings(monkeypatch)

        with pytest.raises(OdpsExportError, match="未安装"):
            execute_query("SELECT 1")


class TestBuildOverviewProbeSqls:
    def test_visibility_stages_precede_filter_stages(self) -> None:
        probes = dict(build_overview_probe_sqls("20260908"))
        # p0 row/column visibility must come before the p1 filter count: when the content column's Label reads NULL,
        # p1=0 cannot tell "really no data" from "the column is not visible"
        assert list(probes) == [
            "p0_raw_partition_rows",
            "p0_raw_content_visible",
            "p0_dwd_rootjson_visible",
            "p1_raw_filter_count",
            "p1b_name_distribution",
            "p2_join_count",
            "p3_join_sample5",
        ]
        # p1b name distribution: when p1=0, tell apart no traffic on the chain / a rename / field-level redaction rewriting the name value
        assert "GROUP BY" in probes["p1b_name_distribution"]
        assert "$.record.name" in probes["p1b_name_distribution"]
        # The p0 section must not extract JSON (when the column is invisible, GET_JSON_OBJECT is all NULL and cannot discriminate)
        assert "GET_JSON_OBJECT" not in probes["p0_raw_partition_rows"]
        assert "GET_JSON_OBJECT" not in probes["p0_raw_content_visible"]
        assert "GET_JSON_OBJECT" not in probes["p0_dwd_rootjson_visible"]
        assert "LENGTH(content)" in probes["p0_raw_content_visible"]
        assert "root_input" in probes["p0_dwd_rootjson_visible"]
        # Every injection uses the real partition, with no placeholder left behind
        for sql in probes.values():
            assert "20260908" in sql
            assert "{ds}" not in sql

    def test_reject_bad_ds(self) -> None:
        with pytest.raises(ValueError, match="YYYYMMDD"):
            build_overview_probe_sqls("2026-09-08")


class TestDiagnoseProbeResults:
    def test_missing_p0_degrades_to_manual(self) -> None:
        text = diagnose_probe_results({})
        assert "人工" in text

    def test_raw_partition_empty(self) -> None:
        text = diagnose_probe_results({"p0_raw_partition_rows": 0})
        assert "分区无数据" in text

    def test_content_column_invisible(self) -> None:
        # The scenario confirmed here: the partition has volume and content is all NULL → Label Security / redaction
        text = diagnose_probe_results(
            {"p0_raw_partition_rows": 100, "p0_raw_content_visible": 0},
        )
        assert "content" in text
        assert "不可见" in text

    def test_link_no_traffic(self) -> None:
        text = diagnose_probe_results(
            {
                "p0_raw_partition_rows": 100,
                "p0_raw_content_visible": 100,
                "p1_raw_filter_count": 0,
            },
        )
        assert "search_agent.stream.overview" in text
        assert "p1b_name_distribution" in text

    def test_join_miss(self) -> None:
        text = diagnose_probe_results(
            {
                "p0_raw_partition_rows": 100,
                "p0_raw_content_visible": 100,
                "p1_raw_filter_count": 10,
                "p2_join_count": 0,
            },
        )
        assert "JOIN" in text

    def test_dwd_column_invisible(self) -> None:
        text = diagnose_probe_results(
            {
                "p0_raw_partition_rows": 100,
                "p0_raw_content_visible": 100,
                "p0_dwd_rootjson_visible": 0,
                "p1_raw_filter_count": 10,
                "p2_join_count": 10,
            },
        )
        assert "root_input" in text

    def test_all_stages_ok_fallback(self) -> None:
        text = diagnose_probe_results(
            {
                "p0_raw_partition_rows": 100,
                "p0_raw_content_visible": 100,
                "p0_dwd_rootjson_visible": 100,
                "p1_raw_filter_count": 10,
                "p2_join_count": 10,
            },
        )
        assert "logview" in text
