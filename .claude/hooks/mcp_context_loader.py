#!/usr/bin/env python3
"""Project-local MCP context loader for Claude Code hooks.

Fires on SessionStart (startup/resume/compact) and PostCompact.
Dumps a concise digest of:
  - codebase-memory graph status
  - ADR content
  - prompt-memory recent context (via global get_full_context skill)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
GLOBAL_GET_FULL_CONTEXT = Path.home() / ".claude/skills/get_full_context/get_full_context.py"


def _run(cmd: list[str], timeout: int = 60) -> str:
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return f"[timeout after {timeout}s]"
    except Exception as exc:  # noqa: BLE001
        return f"[error: {exc}]"


def _section(title: str, body: str) -> None:
    print(f"\n## {title}\n")
    print(body if body else "_No data._")


def main() -> int:
    print("# MCP Context Digest")
    print(
        "_Auto-loaded on session startup, resume, compact, or after refresh.\n"
        "_Source: `.claude/hooks/mcp_context_loader.py`_"
    )

    # 1. Graph status
    status_raw = _run(["python3", "scripts/mcp_status.py"], timeout=30)
    try:
        status_json = json.loads(status_raw)
        status_body = (
            f"- **Project:** {status_json.get('project')}\n"
            f"- **Status:** {status_json.get('status')}\n"
            f"- **Nodes:** {status_json.get('nodes')}\n"
            f"- **Edges:** {status_json.get('edges')}\n"
            f"- **Branch:** {status_json.get('git', {}).get('branch')}\n"
            f"- **Head SHA:** {status_json.get('git', {}).get('head_sha')}"
        )
    except json.JSONDecodeError:
        status_body = f"```\n{status_raw}\n```"
    _section("Codebase Memory Status", status_body)

    # 2. ADR
    adr_raw = _run(["python3", "scripts/mcp_adr.py", "--get"], timeout=30)
    try:
        adr_json = json.loads(adr_raw)
        adr_body = adr_json.get("content", "_No ADR yet._")
    except json.JSONDecodeError:
        adr_body = f"```\n{adr_raw}\n```"
    _section("Architecture Decision Record", adr_body)

    # 3. Prompt-memory context
    if GLOBAL_GET_FULL_CONTEXT.exists():
        prompt_body = _run(["python3", str(GLOBAL_GET_FULL_CONTEXT)], timeout=60)
    else:
        prompt_body = f"_Global get_full_context skill not found at {GLOBAL_GET_FULL_CONTEXT}_"
    _section("Prompt Memory Context", prompt_body)

    return 0


if __name__ == "__main__":
    sys.exit(main())
