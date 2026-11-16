# -*- coding: utf-8 -*-
"""Evolution object hashes.

Bytes are ``evo.v1|{kind}|{canonical json}``. Measurements become decimal
strings. Binary floats, non-NFC text, and format characters are rejected.
"""

from __future__ import annotations

import math
import unicodedata
from decimal import Decimal
from typing import Any

from benchmark.kernel.confloader import canonical, sha256_text


def evolution_hash(object_kind: str, payload: dict[str, Any]) -> str:
    if not object_kind or any(char in object_kind for char in "| \n"):
        raise ValueError("object kind is invalid")
    body = canonical(_ready(payload))
    return sha256_text(f"evo.v1|{object_kind}|{body}")


def _ready(value: Any) -> Any:
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, str):
        return _text(value)
    if isinstance(value, Decimal):
        return _decimal(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError("measurement_invalid")
        return _decimal(Decimal(str(value)))
    if isinstance(value, dict):
        return {_text(str(key)): _ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_ready(item) for item in value]
    raise ValueError("unsupported_hash_value")


def _text(value: str) -> str:
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError("non_nfc_string")
    if any(unicodedata.category(char) == "Cf" for char in value):
        raise ValueError("format_character")
    return value


def _decimal(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("measurement_invalid")
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"
