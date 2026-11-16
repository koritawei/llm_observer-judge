"""Unit tests for LLM JSON parsing helpers."""

from typing import Literal

import pytest
from benchmark.utils.llm_json import (
    extract_json_object,
    parse_and_validate_llm_json,
    strip_json_fence,
)
from pydantic import BaseModel


class _BinaryVerdict(BaseModel):
    score: Literal[0, 1]
    reason: str


def test_strip_json_fence_plain_json_unchanged() -> None:
    assert strip_json_fence('{"a": 1}') == '{"a": 1}'


def test_strip_json_fence_removes_markdown_block() -> None:
    content = '```json\n{"a": 1}\n```'
    assert strip_json_fence(content) == '{"a": 1}'


def test_strip_json_fence_strips_outer_whitespace() -> None:
    content = '  ```\n{"a": 1}\n```  '
    assert strip_json_fence(content) == '{"a": 1}'


def test_extract_json_object_strips_preamble() -> None:
    assert extract_json_object('如下：{"a": 1}\n完') == '{"a": 1}'
    assert extract_json_object("no braces") == "no braces"


def test_parse_and_validate_llm_json_repairs_unescaped_quotes() -> None:
    result = parse_and_validate_llm_json(
        '{"score":0,"reason":"回复中提到"2018年左右"的说法"}', _BinaryVerdict
    )

    expected = _BinaryVerdict(score=0, reason='回复中提到"2018年左右"的说法')
    assert result == expected


def test_parse_and_validate_llm_json_does_not_repair_valid_schema_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_repair(*args: object, **kwargs: object) -> str:
        raise AssertionError("schema errors must not invoke JSON repair")

    monkeypatch.setattr("benchmark.utils.llm_json.repair_json", unexpected_repair)

    assert (
        parse_and_validate_llm_json(
            '{"score":2,"reason":"invalid score"}',
            _BinaryVerdict,
        )
        is None
    )


def test_parse_and_validate_llm_json_returns_none_when_unrecoverable() -> None:
    assert parse_and_validate_llm_json("not JSON", _BinaryVerdict) is None


@pytest.mark.parametrize(
    "content",
    [
        '{"score":0,"reason":"',
        '{"score":',
    ],
)
def test_parse_and_validate_llm_json_rejects_truncated_output_without_repair(
    content: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_repair(*args: object, **kwargs: object) -> str:
        raise AssertionError("truncated model output must not invoke JSON repair")

    monkeypatch.setattr("benchmark.utils.llm_json.repair_json", unexpected_repair)

    assert parse_and_validate_llm_json(content, _BinaryVerdict) is None
