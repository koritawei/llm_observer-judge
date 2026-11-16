"""domain.agent_names — agent-name normalization with no service dependency."""

import subprocess
import sys
from pathlib import Path

from benchmark.domain.agent_names import canonical_agent_name
from benchmark.domain.eval_count_keys import eval_count_name_aliases

_AGENT_EVAL_ROOT = Path(__file__).resolve().parents[3]


def test_canonical_agent_name_unifies_benchmark_suffix() -> None:
    assert canonical_agent_name("example_benchmark") == "example"
    assert canonical_agent_name("amap_agent_benchmark") == "amap_agent"
    assert canonical_agent_name("search_eval") == "search_ai_card"
    assert canonical_agent_name("ieval_external", "amap_chat") == "ieval_external"


def test_eval_count_aliases_include_benchmark_stem() -> None:
    aliases = eval_count_name_aliases("example_benchmark")
    assert "example" in aliases
    assert "example_benchmark" in aliases


def test_eval_count_keys_import_does_not_load_services() -> None:
    """A clean interpreter: importing eval_count_keys must not pull in benchmark.services / run_service."""
    root = _AGENT_EVAL_ROOT.as_posix()
    script = "\n".join(
        [
            "import sys",
            f"sys.path.insert(0, {root!r})",
            "assert 'benchmark.services' not in sys.modules",
            "import benchmark.domain.eval_count_keys as keys",
            "assert 'benchmark.services' not in sys.modules",
            "assert 'benchmark.runs.run_service' not in sys.modules",
            "assert 'benchmark.domain.agent_names' in sys.modules",
            "assert keys.eval_count_name_aliases('example_benchmark')",
        ]
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
