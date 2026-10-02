#!/usr/bin/env python3
"""Render the review status document from the machine-readable backlog.

``docs/dev/review_backlog.json`` is the source: one entry per item of the
2026-10-02 repository review, with a status (``done`` / ``partial`` /
``not_done``), where it landed, the tests that hold it, and for anything
not done the reason and the next step. This script renders
``docs/dev/2026-10-02-review-status.md`` from it.

The point is that the status page cannot drift from the work: the page is
generated, and ``tests/test_review_backlog.py`` fails when a ``done`` item
names a file or test that does not exist, or when the page is stale.

Usage
-----
    python scripts/build_review_status.py           # regenerate
    python scripts/build_review_status.py --check   # drift gate
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
BACKLOG = REPO_ROOT / "docs" / "dev" / "review_backlog.json"
OUT = REPO_ROOT / "docs" / "dev" / "2026-10-02-review-status.md"

STATUSES = ("done", "partial", "not_done")


def _code_list(paths: List[str]) -> str:
    return "<br>".join(f"`{p}`" for p in paths) if paths else "—"


def render(backlog: Dict[str, Any]) -> str:
    items = backlog["items"]
    counts = {s: sum(1 for i in items if i["status"] == s) for s in STATUSES}
    lines = [
        "# 2026-10-02 仓库审查：逐项状态",
        "",
        "由 `python scripts/build_review_status.py` 从 "
        "`docs/dev/review_backlog.json` 生成，不要手改本文件。",
        "",
        f"对应 `{backlog['review']}`。审查基线是 {backlog['baseline']['review']}，"
        f"工作基于 {backlog['baseline']['work']}。",
        "",
        f"共 {len(items)} 项：已完成 **{counts['done']}**，部分完成 "
        f"**{counts['partial']}**，未做 **{counts['not_done']}**。"
        "“已完成”的每一项都列出守着它的测试；“部分完成”和“未做”写明缺什么、为什么、下一步。",
        "",
        "## 已完成",
        "",
        "| ID | 事项 | 落地位置 | 验收测试 | 备注 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in items:
        if item["status"] == "done":
            lines.append(
                f"| {item['id']} | {item['title']} | {_code_list(item['where'])} | "
                f"{_code_list(item['tests'])} | {item['note'] or ''} |"
            )
    lines += [
        "",
        "## 部分完成",
        "",
        "| ID | 事项 | 已有的 | 还缺什么 | 下一步 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in items:
        if item["status"] == "partial":
            have = _code_list(item["where"] + item["tests"])
            lines.append(
                f"| {item['id']} | {item['title']} | {have} | {item['note']} | "
                f"{item['next'] or '—'} |"
            )
    lines += [
        "",
        "## 未做，以及为什么",
        "",
        "| ID | 事项 | 理由 | 下一步 |",
        "| --- | --- | --- | --- |",
    ]
    for item in items:
        if item["status"] == "not_done":
            lines.append(
                f"| {item['id']} | {item['title']} | {item['note']} | "
                f"{item['next'] or '—'} |"
            )
    lines += ["", "## 已核实无需改动", ""]
    lines += [f"- {text}" for text in backlog.get("verified_no_change", [])]
    lines += ["", "## 做的过程中查出的问题", ""]
    lines += [f"- {text}" for text in backlog.get("findings", [])]
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    backlog = json.loads(BACKLOG.read_text(encoding="utf-8"))
    page = render(backlog)
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != page:
            print(
                "[review_status] STALE -- run python scripts/build_review_status.py",
                file=sys.stderr,
            )
            return 1
        print("[review_status] OK")
        return 0
    OUT.write_text(page, encoding="utf-8")
    print(f"[review_status] wrote {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
