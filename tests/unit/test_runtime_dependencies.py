"""Keep optional report storage requirements unambiguous for installers."""

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement


@pytest.mark.parametrize("package", ["storage", "oss2", "normandy-credential-sdk"])
def test_report_storage_dependencies_have_one_constraint(package: str) -> None:
    project_file = Path(__file__).resolve().parents[2] / "pyproject.toml"
    project = tomllib.loads(project_file.read_text())
    requirements = [Requirement(value) for value in project["project"]["dependencies"]]
    matches = [requirement for requirement in requirements if requirement.name == package]

    assert len(matches) == 1
    if package == "oss2":
        assert "2.19.0" in matches[0].specifier
        assert "2.18.0" not in matches[0].specifier
    if package == "normandy-credential-sdk":
        assert "extension" in matches[0].extras

    lock = tomllib.loads(project_file.with_name("uv.lock").read_text())
    locked_project = next(item for item in lock["package"] if item["name"] == "agent-eval")
    locked_requirements = locked_project["metadata"]["requires-dist"]
    assert len([item for item in locked_requirements if item["name"] == package]) == 1
