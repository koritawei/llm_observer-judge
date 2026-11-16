#!/usr/bin/env python3
"""Append the unified D0-D6 human-review sheet.

Run from the agent-eval service directory:
    .venv/bin/python -m scripts.add_human_eval_sheet report.xlsx
"""

import os
import sys
from pathlib import Path

if __name__ == "__main__" and len(sys.argv) != 2:
    print(__doc__, file=sys.stderr)
    sys.exit(2)

# Preserve direct script invocation without mutating sys.path.
if __name__ == "__main__" and not __package__:
    sys.argv[1] = str(Path(sys.argv[1]).resolve())
    os.chdir(Path(__file__).resolve().parents[1])
    os.execv(sys.executable, [sys.executable, "-m", "scripts.add_human_eval_sheet", *sys.argv[1:]])

from benchmark.reporting.cluster.render.case_eval_xlsx import (
    append_human_eval_sheet,  # noqa: E402 - CLI bootstrap
)


def add_human_eval_sheet(xlsx_path: str) -> None:
    """Preserve existing data and skip a workbook that already has the review sheet."""
    try:
        import openpyxl  # noqa: PLC0415 - optional Excel dependency
    except ImportError as exc:
        raise RuntimeError("openpyxl_not_installed") from exc
    path = Path(xlsx_path)
    workbook = openpyxl.load_workbook(path)
    try:
        if "人工评测" in workbook.sheetnames:
            print("  「人工评测」sheet 已存在，跳过。")
            return
        missing = append_human_eval_sheet(workbook, workbook["评测明细"])
        if missing:
            print("警告：缺少上下文列：" + "、".join(missing), file=sys.stderr)
        workbook.save(path)
        print("  已添加「人工评测」sheet。")
    finally:
        workbook.close()


if __name__ == "__main__":
    add_human_eval_sheet(sys.argv[1])
