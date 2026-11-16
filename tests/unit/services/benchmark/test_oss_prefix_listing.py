"""OSS prefix listing: drop date macros + the most recent N date partitions (real OSS dates)."""

from datetime import UTC, datetime
from types import SimpleNamespace

from benchmark.runs import oss_prefix_listing as mod


def test_strip_date_macros_normalizes_path() -> None:
    assert mod.strip_date_macros("agent-eval/x/${yyyymmdd-1}") == "agent-eval/x"
    assert mod.strip_date_macros("oss://agent-eval-oss/a/${yyyymmdd}/b") == "a/b"
    assert mod.strip_date_macros("/a//${yyyymmdd+3}//") == "a"
    assert mod.strip_date_macros("") == ""


class _FakeBucket:
    """Minimal stand-in for an oss2 Bucket: list_objects returns an object shaped like ListObjectsResult.

    oss2's common-prefix attribute is ``prefix_list`` (not ``prefixes``),
    and this builds field names in the real SDK shape so it does not regress.
    """

    def __init__(
        self,
        *,
        prefix_list: list[str] | None = None,
        objects: list[object] | None = None,
    ) -> None:
        self._prefix_list = prefix_list or []
        self._objects = objects or []
        self.calls: list[dict[str, object]] = []

    def list_objects(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(
            prefix_list=self._prefix_list,
            object_list=self._objects,
        )


def test_date_entries_sorted_desc_and_limited(monkeypatch) -> None:
    bucket = _FakeBucket(
        prefix_list=[
            "base/20261018/",
            "base/20261021/",
            "base/not-a-date/",
            "base/20261020/",
            "base/20261019/",
            "base/20261017/",
            "base/20261016/",
        ],
    )
    monkeypatch.setattr(mod, "_build_bucket", lambda: (bucket, "bkt"))

    entries = mod._list_sync("base/${yyyymmdd-1}", 5)

    assert [item.date for item in entries] == [
        "20261021",
        "20261020",
        "20261019",
        "20261018",
        "20261017",
    ]
    # Show only the prefix (no bucket / filename)
    assert entries[0].path == "base/20261021"
    assert bucket.calls[0]["delimiter"] == "/"
    assert bucket.calls[0]["prefix"] == "base/"


def test_object_entries_sorted_by_last_modified(monkeypatch) -> None:
    objects = [
        SimpleNamespace(key="base/a.csv", last_modified=datetime(2026, 10, 20, tzinfo=UTC)),
        SimpleNamespace(key="base/b.csv", last_modified=datetime(2026, 10, 22, tzinfo=UTC)),
        SimpleNamespace(key="base/c.csv", last_modified=datetime(2026, 10, 21, tzinfo=UTC)),
    ]
    bucket = _FakeBucket(objects=objects)
    monkeypatch.setattr(mod, "_build_bucket", lambda: (bucket, "bkt"))

    entries = mod._list_sync("base", 2)

    assert [item.path for item in entries] == ["base/b.csv", "base/c.csv"]
    assert entries[0].date == ""
    assert entries[0].modified_at == "2026-10-22T00:00:00+00:00"
