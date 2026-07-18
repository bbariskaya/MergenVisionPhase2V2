#!/usr/bin/env python3
"""Codebase-memory index status checker.

Kullanım:
    python scripts/mcp_status.py
    python scripts/mcp_status.py --project home-user-Workspace-MergenVisionPhase2v2
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
                "clientInfo": {"name": "mcp-status-cli", "version": "1.0.0"},
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

    def call_tool(self, name: str, arguments: dict[str, Any], timeout: int = 60) -> Any:
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
    parser = argparse.ArgumentParser(description="Codebase-memory index status")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Codebase-memory proje adı")
    args = parser.parse_args(argv)

    if not Path(CODEBASE_MEMORY_MCP).exists():
        print(f"Hata: codebase-memory-mcp bulunamadı: {CODEBASE_MEMORY_MCP}", file=sys.stderr)
        return 1

    client = McpClient(CODEBASE_MEMORY_MCP)
    try:
        result = client.call_tool("index_status", {"project": args.project}, timeout=30)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
