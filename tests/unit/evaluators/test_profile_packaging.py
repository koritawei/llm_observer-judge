"""Packaging gate: a non-.py resource read at runtime must be covered by SERVICE_DATA of the deploy build.

Plugins locate a Profile with ``Path(__file__).with_name("profile.yaml")``. The source tree and
pytest can always read it; only the deployed state (Bazel runfiles) raises
``FileNotFoundError`` at startup when the declaration is missing. Here the BUILD glob declaration is cross-checked
against the resources actually on disk, so a missing declaration shows up in unit tests instead of a pre-env that will not start.
"""

import re
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[3]
BUILD_FILE = SERVICE_ROOT / "BUILD.docker-service.bazel"


def _service_data_patterns() -> list[str]:
    """Extract the path patterns declared in ``SERVICE_DATA = glob([...])``."""

    block = re.search(
        r"SERVICE_DATA = glob\(\[(.*?)\]\)",
        BUILD_FILE.read_text(encoding="utf-8"),
        re.DOTALL,
    )
    assert block is not None, f"SERVICE_DATA glob block not found in {BUILD_FILE.name}"
    patterns = re.findall(r'"([^"]+)"', block.group(1))
    assert patterns, "SERVICE_DATA declares no patterns"
    return patterns


def _packaged_files() -> set[Path]:
    """The set of files SERVICE_DATA actually packs into runfiles (relative to the service root)."""

    packaged: set[Path] = set()
    for pattern in _service_data_patterns():
        file_pattern = f"{pattern}/*" if pattern.endswith("/**") else pattern
        packaged |= {
            path.relative_to(SERVICE_ROOT)
            for path in SERVICE_ROOT.glob(file_pattern)
            if path.is_file()
        }
    return packaged


def test_evaluator_profiles_are_packaged() -> None:
    profiles = {
        path.relative_to(SERVICE_ROOT)
        for path in (SERVICE_ROOT / "business" / "evaluators").glob("*/profile.yaml")
    }
    assert profiles, "no evaluator profile.yaml found — check path assumptions"
    missing = sorted(str(path) for path in profiles - _packaged_files())
    assert not missing, f"profile.yaml not covered by SERVICE_DATA: {missing}"


def test_attribution_conf_resources_are_packaged() -> None:
    """The conf/ shipped inside the attribution plugin package (config + prompts) is also read from disk at runtime."""

    conf_files = {
        path.relative_to(SERVICE_ROOT)
        for path in (SERVICE_ROOT / "business" / "agents").glob("*/conf/**/*")
        if path.is_file()
    }
    missing = sorted(str(path) for path in conf_files - _packaged_files())
    assert not missing, f"attribution conf not covered by SERVICE_DATA: {missing}"
