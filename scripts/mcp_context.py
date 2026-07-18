#!/usr/bin/env python3
"""Akıllı context retrieval orchestrator.

Kullanım:
    python scripts/mcp_context.py "bulk enrollment"
    python scripts/mcp_context.py "kim create_person'ı çağırıyor" --limit 10
    python scripts/mcp_context.py "enroll" --trace --snippet

Bu script codebase-memory-mcp sunucusuna stdio üzerinden bağlanır ve
search_graph + search_code + trace_path sonuçlarını birleştirir.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CODEBASE_MEMORY_MCP = os.environ.get(
    "CODEBASE_MEMORY_MCP", "/home/user/.local/bin/codebase-memory-mcp"
)
DEFAULT_PROJECT = "home-user-Workspace-MergenVisionPhase2v2"


class McpClient:
    """Minimal stdio MCP client for codebase-memory-mcp."""

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
                "clientInfo": {"name": "mcp-context-cli", "version": "1.0.0"},
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

    def call_tool(self, name: str, arguments: dict[str, Any], timeout: int = 120) -> Any:
        resp = self._request(
            "tools/call",
            {"name": name, "arguments": arguments},
            timeout=timeout,
        )
        if resp is None:
            raise TimeoutError(f"Tool {name} timed out")
        if "error" in resp:
            raise RuntimeError(f"Tool {name} error: {resp['error']}")
        # Prefer structuredContent when available.
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


@dataclass
class ContextItem:
    name: str = ""
    qualified_name: str = ""
    label: str = ""
    file_path: str = ""
    start_line: int = 0
    end_line: int = 0
    score: float = 0.0
    sources: set[str] = field(default_factory=set)
    note: str = ""

    @property
    def display_path(self) -> str:
        if self.start_line:
            return f"{self.file_path}:{self.start_line}"
        return self.file_path

    @property
    def source_priority(self) -> int:
        if "graph" in self.sources:
            return 0
        if "code" in self.sources:
            return 1
        return 2

    @property
    def sort_key(self) -> tuple[int, int, float, str]:
        # Definitions first, then by source (graph > code > trace), then score.
        is_definition = self.label in {"Function", "Method", "Class", "Interface", "Route"}
        return (
            0 if is_definition else 1,
            self.source_priority,
            -self.score,
            self.qualified_name,
        )


def _item_from_search_graph(node: dict[str, Any], source: str) -> ContextItem:
    # search_graph rank is negative (lower is better); invert so higher is better.
    rank = node.get("rank", 0.0)
    return ContextItem(
        name=node.get("name", ""),
        qualified_name=node.get("qualified_name", ""),
        label=node.get("label", ""),
        file_path=node.get("file_path", ""),
        start_line=node.get("start_line", 0),
        end_line=node.get("end_line", 0),
        score=-rank,
        sources={source},
    )


def _item_from_search_code(node: dict[str, Any], source: str) -> ContextItem:
    # search_code uses 'node' and 'file' keys instead of 'name'/'file_path'.
    return ContextItem(
        name=node.get("name") or node.get("node", ""),
        qualified_name=node.get("qualified_name", ""),
        label=node.get("label", ""),
        file_path=node.get("file_path") or node.get("file", ""),
        start_line=node.get("start_line", 0),
        end_line=node.get("end_line", 0),
        score=(node.get("in_degree", 0) + node.get("out_degree", 0)) * 0.1,
        sources={source},
    )


def _merge_items(items: list[ContextItem]) -> list[ContextItem]:
    by_qn: dict[str, ContextItem] = {}
    for item in items:
        key = item.qualified_name or f"{item.file_path}:{item.start_line}:{item.name}"
        if key in by_qn:
            existing = by_qn[key]
            existing.sources.update(item.sources)
            if abs(item.score) > abs(existing.score):
                existing.score = item.score
        else:
            by_qn[key] = item
    return sorted(by_qn.values(), key=lambda x: x.sort_key)


def _is_test_path(path: str) -> bool:
    return "/tests/" in path or path.startswith("tests/") or path.endswith("_test.py")


def _run_search_graph(client: McpClient, project: str, query: str, limit: int) -> list[ContextItem]:
    result = client.call_tool(
        "search_graph",
        {"project": project, "query": query, "limit": limit},
        timeout=120,
    )
    nodes = result.get("results", []) if isinstance(result, dict) else []
    return [_item_from_search_graph(n, "graph") for n in nodes]


def _run_search_code(client: McpClient, project: str, pattern: str, limit: int) -> list[ContextItem]:
    result = client.call_tool(
        "search_code",
        {"project": project, "pattern": pattern, "limit": limit, "mode": "compact"},
        timeout=120,
    )
    if not isinstance(result, dict):
        return []
    nodes = result.get("results", [])
    return [_item_from_search_code(n, "code") for n in nodes]


def _run_trace_path(
    client: McpClient, project: str, function_name: str, depth: int
) -> list[ContextItem]:
    result = client.call_tool(
        "trace_path",
        {
            "project": project,
            "function_name": function_name,
            "mode": "calls",
            "direction": "both",
            "depth": depth,
            "include_tests": False,
        },
        timeout=120,
    )
    items: list[ContextItem] = []
    if not isinstance(result, dict):
        return items
    # Result shape depends on mode/direction; collect nodes from paths.
    paths = result.get("paths", result.get("results", []))
    for path in paths:
        if not isinstance(path, list):
            path = [path]
        for node in path:
            if not isinstance(node, dict):
                continue
            items.append(
                ContextItem(
                    name=node.get("name", ""),
                    qualified_name=node.get("qualified_name", ""),
                    label=node.get("label", ""),
                    file_path=node.get("file_path", ""),
                    start_line=node.get("start_line", 0),
                    end_line=node.get("end_line", 0),
                    score=0.0,
                    sources={"trace"},
                    note=f"calls {function_name}",
                )
            )
    return items


def _fetch_snippet(client: McpClient, project: str, qualified_name: str) -> str:
    try:
        result = client.call_tool(
            "get_code_snippet",
            {"project": project, "qualified_name": qualified_name},
            timeout=60,
        )
        if isinstance(result, dict):
            # get_code_snippet returns the snippet under the 'source' key.
            source = result.get("source")
            if source:
                return str(source)
            # Fallback to raw content text if present.
            content = result.get("content", [])
            if content and content[0].get("type") == "text":
                return content[0].get("text", "")
        return str(result)[:500]
    except Exception as exc:  # noqa: BLE001
        return f"snippet error: {exc}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Akıllı codebase context retrieval")
    parser.add_argument("query", help="Doğal dil sorgusu")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Codebase-memory proje adı")
    parser.add_argument("--limit", type=int, default=15, help="Maksimum sonuç sayısı")
    parser.add_argument("--code-limit", type=int, default=15, help="search_code limiti")
    parser.add_argument("--trace", action="store_true", help="Top semboller için trace_path çalıştır")
    parser.add_argument("--trace-depth", type=int, default=2, help="trace derinliği")
    parser.add_argument("--snippet", action="store_true", help="İlk sonuç için kod snippet'ı getir")
    parser.add_argument("--json", action="store_true", help="JSON çıktı")
    args = parser.parse_args(argv)

    if not Path(CODEBASE_MEMORY_MCP).exists():
        print(f"Hata: codebase-memory-mcp bulunamadı: {CODEBASE_MEMORY_MCP}", file=sys.stderr)
        return 1

    client = McpClient(CODEBASE_MEMORY_MCP)
    try:
        all_items: list[ContextItem] = []

        # 1. Graph search
        graph_items = _run_search_graph(client, args.project, args.query, args.limit)
        all_items.extend(graph_items)

        # 2. Code search (use query as literal pattern)
        code_items = _run_search_code(client, args.project, args.query, args.code_limit)
        all_items.extend(code_items)

        # 3. Trace top symbols
        if args.trace:
            for item in graph_items[:5]:
                if item.name:
                    traced = _run_trace_path(
                        client, args.project, item.name, args.trace_depth
                    )
                    all_items.extend(traced)

        merged = _merge_items(all_items)
        # Keep non-test items first unless only tests remain.
        non_tests = [i for i in merged if not _is_test_path(i.file_path)]
        tests = [i for i in merged if _is_test_path(i.file_path)]
        merged = (non_tests + tests)[: args.limit]

        if args.json:
            print(
                json.dumps(
                    [
                        {
                            "name": i.name,
                            "qualified_name": i.qualified_name,
                            "label": i.label,
                            "file_path": i.file_path,
                            "start_line": i.start_line,
                            "end_line": i.end_line,
                            "score": i.score,
                            "sources": sorted(i.sources),
                            "note": i.note,
                        }
                        for i in merged
                    ],
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0

        print(f"## Context retrieval: `{args.query}`")
        print(f"Project: `{args.project}` | Toplam: {len(merged)} sonuç\n")
        print("| # | Tür | Sembol | Dosya | Kaynak | Not |")
        print("|---|-----|--------|-------|--------|-----|")
        for idx, item in enumerate(merged, 1):
            name = item.name or item.qualified_name.split(".")[-1] if item.qualified_name else "-"
            label = item.label or "-"
            path = item.display_path or "-"
            sources = ", ".join(sorted(item.sources))
            note = item.note or ""
            print(f"| {idx} | {label} | `{name}` | {path} | {sources} | {note} |")

        if args.snippet and merged:
            first = merged[0]
            print(f"\n### Snippet: `{first.name}`\n")
            snippet = _fetch_snippet(client, args.project, first.qualified_name or first.name)
            print("```python")
            print(snippet)
            print("```")

        return 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
