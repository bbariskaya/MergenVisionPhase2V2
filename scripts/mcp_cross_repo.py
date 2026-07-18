#!/usr/bin/env python3
"""Cross-repo intelligence helper.

Kullanım:
    python scripts/mcp_cross_repo.py --repo /home/user/MergenVisionDemo --name MergenVisionDemo
    python scripts/mcp_cross_repo.py --cross-repo-only

1. Hedef repoyu codebase-memory-mcp'ye indexler (full/moderate/fast).
2. Opsiyonel olarak cross-repo-intelligence modunu çalıştırarak
   CROSS_HTTP_CALLS / CROSS_ASYNC_CALLS / CROSS_CHANNEL edge'lerini oluşturur.
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
PHASE2_PROJECT = "home-user-Workspace-MergenVisionPhase2v2"


class McpClient:
    """Minimal stdio MCP client."""

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
                "clientInfo": {"name": "mcp-cross-repo-cli", "version": "1.0.0"},
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

    def call_tool(self, name: str, arguments: dict[str, Any], timeout: int = 600) -> Any:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-repo intelligence helper")
    parser.add_argument("--repo", default="/home/user/MergenVisionDemo", help="Indexlenecek repo kökü")
    parser.add_argument("--name", default="MergenVisionDemo", help="Codebase-memory proje adı")
    parser.add_argument("--mode", choices=["fast", "moderate", "full"], default="moderate", help="Index modu")
    parser.add_argument("--no-persist", action="store_true", help="graph.db.zst artifact oluşturma")
    parser.add_argument("--cross-repo-only", action="store_true", help="Sadece cross-repo intelligence çalıştır")
    args = parser.parse_args(argv)

    if not Path(CODEBASE_MEMORY_MCP).exists():
        print(f"Hata: codebase-memory-mcp bulunamadı: {CODEBASE_MEMORY_MCP}", file=sys.stderr)
        return 1

    client = McpClient(CODEBASE_MEMORY_MCP)
    try:
        if not args.cross_repo_only:
            repo_path = Path(args.repo)
            if not repo_path.exists():
                print(f"Hata: repo bulunamadı: {repo_path}", file=sys.stderr)
                return 1
            print(f"Indexing {args.repo} as {args.name} (mode={args.mode})...")
            result = client.call_tool(
                "index_repository",
                {
                    "repo_path": str(repo_path),
                    "name": args.name,
                    "mode": args.mode,
                    "persistence": not args.no_persist,
                },
                timeout=1800,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False)[:2000])

        print("Running cross-repo intelligence...")
        cross = client.call_tool(
            "index_repository",
            {
                "repo_path": str(Path(__file__).resolve().parents[1]),
                "name": PHASE2_PROJECT,
                "mode": "cross-repo-intelligence",
                "target_projects": ["*"],
            },
            timeout=600,
        )
        print(json.dumps(cross, indent=2, ensure_ascii=False)[:2000])
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
