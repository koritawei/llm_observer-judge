"""OSS blob retry and cache behaviour."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from benchmark.ingest.sidecar.oss_blob import OssBlob, _should_retry_oss


def test_should_retry_server_and_auth_errors() -> None:
    assert _should_retry_oss(SimpleNamespace(status=500))
    assert _should_retry_oss(SimpleNamespace(status=401))
    assert _should_retry_oss(SimpleNamespace(status=403))
    assert _should_retry_oss(SimpleNamespace(status=None))
    assert not _should_retry_oss(SimpleNamespace(status=404))
    assert not _should_retry_oss(SimpleNamespace(status=400))


def test_retry_oss_does_not_retry_not_found() -> None:
    class _OssError(Exception):
        def __init__(self, status: int) -> None:
            self.status = status

    blob = OssBlob(
        endpoint="https://oss.example",
        access_key_id="ak",
        access_key_secret="sk",
    )
    bucket = MagicMock()
    bucket.get_object.side_effect = _OssError(404)
    with (
        patch.object(blob, "_oss_bucket", return_value=bucket),
        patch("oss2.exceptions.OssError", _OssError),
        pytest.raises(_OssError),
    ):
        blob._retry_oss("bkt", lambda client: client.get_object("missing"))
    assert bucket.get_object.call_count == 1
