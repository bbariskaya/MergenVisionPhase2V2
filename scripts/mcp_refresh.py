#!/usr/bin/env python3
"""Codebase-memory graph refresh script.

Kullanım:
    python scripts/mcp_refresh.py
    python scripts/mcp_refresh.py --since HEAD~5
    python scripts/mcp_refresh.py --mode moderate

Değişen dosyalara göre graph'i hızlı veya moderate modda yeniler,
``.codebase-memory/graph.db.zst`` artifact'ını persistence=True ile yazar.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import subprocess
import sys
from pathlib import Path
from typing import Any

CODEBASE_MEMORY_MCP = os.environ.get(
    "CODEBASE_MEMORY_MCP", "/home/user/.local/bin/codebase-memory-mcp"
)
DEFAULT_PROJECT = "home-user-Workspace-MergenVisionPhase2v2"
REPO_ROOT = Path(__file__).resolve().parents[1]

# Paths that usually signal an architectural decision may need updating.
ADR_SENSITIVE_PATHS = (
    "backend/app/domain",
    "backend/app/application/ports",
    "backend/app/api/routes",
    "backend/app/infrastructure",
    "backend/app/worker",
)


class McpClient:
    """Minimal stdio MCP client (kept self-contained for script portability)."""

    def __init__(self, command: str) -> None:
        self._proc = subprocess.Popen(
            [command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._next_id = 1
        self._initialize()

    def _initialize(self) -> None:
        result = self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "mcp-refresh-cli", "version": "1.0.0"},
            },
            timeout=10,
        )
        if not result:
            raise RuntimeError("MCP initialize failed")
        self._notify("notifications/initialized", {})

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        self._proc.stdin.write(json.dumps(msg) + "\n")  # type: ignore[union-attr]
        self._proc.stdin.flush()  # type: ignore[union-attr]

    def _request(
        self, method: str, params: dict[str, Any], timeout: int = 60
    ) -> dict[str, Any] | None:
        msg_id = self._next_id
        self._next_id += 1
        msg = {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}
        self._proc.stdin.write(json.dumps(msg) + "\n")  # type: ignore[union-attr]
        self._proc.stdin.flush()  # type: ignore[union-attr]
        ready, _, _ = select.select([self._proc.stdout], [], [], timeout)
        if not ready:
            return None
        line = self._proc.stdout.readline()
        if not line:
            return None
        return json.loads(line)

    def call_tool(self, name: str, arguments: dict[str, Any], timeout: int = 300) -> Any:
        resp = self._request(
            "tools/call",
            {"name": name, "arguments": arguments},
            timeout=timeout,
        )
        if resp is None:
            raise TimeoutError(f"Tool {name} timed out")
        if "error" in resp:
            raise RuntimeError(f"Tool {name} error: {resp['error']}")
        result = resp.get("result", {})
        if "structuredContent" in result:
            return result["structuredContent"]
        content = result.get("content", [])
        if content and content[0].get("type") == "text":
            text = content[0].get("text", "")
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        return result

    def close(self) -> None:
        self._proc.terminate()
        try:
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._proc.kill()


def _git_changed_files(since: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", since],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _git_untracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _touches_adr_sensitive(changed: list[str]) -> bool:
    return any(any(changed_path.startswith(prefix) for prefix in ADR_SENSITIVE_PATHS) for changed_path in changed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Codebase-memory graph refresh")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Codebase-memory proje adı")
    parser.add_argument("--since", default="HEAD", help="Git ref: değişikliklerin karşılaştırılacağı referans")
    parser.add_argument("--mode", choices=["fast", "moderate", "full"], default=None, help="Index modu (default: dosya sayısına göre)")
    parser.add_argument("--no-persist", action="store_true", help="graph.db.zst artifact'ı oluşturma")
    args = parser.parse_args(argv)

    if not Path(CODEBASE_MEMORY_MCP).exists():
        print(f"Hata: codebase-memory-mcp bulunamadı: {CODEBASE_MEMORY_MCP}", file=sys.stderr)
        return 1

    changed = _git_changed_files(args.since)
    untracked = _git_untracked_files()
    all_changed = list(dict.fromkeys(changed + untracked))  # preserve order, dedup

    if not all_changed:
        print("Değişen dosya yok; refresh atlanıyor.")
        return 0

    mode = args.mode
    if mode is None:
        mode = "moderate" if len(all_changed) > 50 else "fast"

    print(f"Değişen dosya sayısı: {len(all_changed)} (untracked: {len(untracked)})")
    print(f"Seçilen index modu: {mode}")

    client = McpClient(CODEBASE_MEMORY_MCP)
    try:
        print("Index başlatılıyor...")
        result = client.call_tool(
            "index_repository",
            {
                "repo_path": str(REPO_ROOT),
                "name": args.project,
                "mode": mode,
                "persistence": not args.no_persist,
            },
            timeout=600,
        )
        print("Index tamamlandı.")
        if isinstance(result, dict):
            summary_keys = ["nodes", "edges", "files_indexed", "elapsed_seconds", "persistence_path"]
            for key in summary_keys:
                if key in result:
                    print(f"  {key}: {result[key]}")

        if _touches_adr_sensitive(all_changed):
            print("\n⚠️  Mimari katmanlarda değişiklik algılandı.")
            print("    Eğer bir karar değiştiyse `mcp__codebase-memory-mcp__manage_adr` ile ADR güncelleyin.")
            print("    Etkilenen alanlar:")
            for prefix in ADR_SENSITIVE_PATHS:
                hits = [p for p in all_changed if p.startswith(prefix)]
                if hits:
                    print(f"      - {prefix}: {len(hits)} dosya")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
