"""Unit tests for read_label_config."""

from benchmark.labeling.project_resolver import read_label_config


def test_read_default_template() -> None:
    content = read_label_config("conf/label_studio/badcase_review.xml")
    assert "<View>" in content
    assert "judge_score_correct" in content
    assert "attribution_correct" in content


def test_read_nonexistent_raises() -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        read_label_config("conf/label_studio/nonexistent.xml")
