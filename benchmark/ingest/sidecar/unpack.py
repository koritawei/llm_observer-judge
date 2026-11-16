"""zstd unpack of sidecar-trace observation blobs."""

import json
from typing import Any

import zstandard as zstd


class UnpackError(ValueError):
    """Packed observations are not a zstd JSON array."""


def unpack_observations(packed: bytes | memoryview | bytearray) -> list[Any]:
    """Decompress zstd-packed JSON array of observation objects."""
    try:
        data = json.loads(zstd.ZstdDecompressor().decompress(bytes(packed)).decode("utf-8"))
    except (zstd.ZstdError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise UnpackError("observations blob is not a zstd JSON array") from exc
    if not isinstance(data, list):
        raise UnpackError("observations json is not a list")
    return data


__all__ = ["UnpackError", "unpack_observations"]
