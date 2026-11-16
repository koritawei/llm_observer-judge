"""Single-case probe: only look at aoned latency and the result. It does not enter the Pipeline and does not write back.

Usage (from this package directory)::

    uv run python probe.py --task case.json

case.json fields match TaskPack. A real call needs::

    CODE_ATTRIBUTION_ENABLED=true
    AONED_BASE_URL=https://<sandbox>/open-api/v1
    AONED_TOKEN=ck_...
    AONED_PROJECT_PATH=/home/admin/workspace/amap-ai-agent
"""

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmark.attribution.code.models import ResultPack, TaskPack
from benchmark.attribution.code.runner import CodeAttributionModule

_LINE_REF = re.compile(r":\d+\b")
_PATCH = re.compile(r"```|diff --git")


def load_task(path: Path) -> TaskPack:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("task json must be an object")
    fields = {key: str(data.get(key) or "") for key in TaskPack.__dataclass_fields__}
    required = (
        "case_id",
        "query",
        "symptom_layer",
        "symptom_category",
        "symptom_root_cause",
        "symptom_chain",
        "symptom_suggestion",
        "trace_summary",
    )
    missing = [key for key in required if not fields[key]]
    if missing:
        raise ValueError("missing " + ",".join(missing))
    return TaskPack(**fields)


def assess(result: ResultPack) -> list[str]:
    issues: list[str] = []
    if result.status != "ok":
        issues.append(f"status={result.status} {result.error_message}".strip())
        return issues
    for name in ("module_root_cause", "evidence_chain", "fix_plan"):
        text = getattr(result, name)
        if _LINE_REF.search(text):
            issues.append(f"{name} 含文件行号")
        if _PATCH.search(text):
            issues.append(f"{name} 含补丁")
    return issues


@dataclass(frozen=True)
class ProbeStats:
    """Latency counters for one probe run."""

    wall_sec: float
    session_sec: float
    poll_count: int
    attempts: int


def format_report(result: ResultPack, stats: ProbeStats) -> str:
    issues = assess(result)
    payload = {
        "wall_sec": round(stats.wall_sec, 3),
        "session_sec": round(stats.session_sec, 3),
        "poll_count": stats.poll_count,
        "attempts": stats.attempts,
        "result": asdict(result),
        "issues": issues,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="单 case 代码归因探针，不回写宿主")
    parser.add_argument("--task", required=True, help="TaskPack JSON 路径")
    args = parser.parse_args(argv)
    task = load_task(Path(args.task))
    module = CodeAttributionModule()
    started = time.monotonic()
    result = module.run(task)
    wall = time.monotonic() - started
    timings = module.last_timings
    print(
        format_report(
            result,
            ProbeStats(
                wall_sec=wall,
                session_sec=timings.session_sec,
                poll_count=timings.poll_count,
                attempts=timings.attempts,
            ),
        )
    )
    return 0 if result.status == "ok" and not assess(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
