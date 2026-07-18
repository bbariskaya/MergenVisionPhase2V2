#!/usr/bin/env python3
"""Impact analysis script backed by codebase-memory-mcp.

Kullanım:
    python scripts/mcp_impact.py "enroll_batch"
    python scripts/mcp_impact.py "backend/app/api/routes/people.py"
    python scripts/mcp_impact.py "BulkEnrollmentService" --depth 4

Bir sembol/dosya üzerinde trace_path + complexity sorgusu yapar ve
markdown formatında etki raporu üretir.
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
                "clientInfo": {"name": "mcp-impact-cli", "version": "1.0.0"},
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

    def call_tool(self, name: str, arguments: dict[str, Any], timeout: int = 180) -> Any:
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


def _find_symbol(client: McpClient, project: str, target: str) -> dict[str, Any] | None:
    # If target looks like a file path, try search_code first.
    if "/" in target or target.endswith(".py"):
        result = client.call_tool(
            "search_code",
            {"project": project, "pattern": target, "limit": 5, "mode": "compact"},
            timeout=60,
        )
        if isinstance(result, dict):
            nodes = result.get("results", [])
            for node in nodes:
                if node.get("file") == target or node.get("file_path") == target:
                    return {
                        "name": node.get("node") or node.get("name"),
                        "qualified_name": node.get("qualified_name", ""),
                        "label": node.get("label", ""),
                        "file_path": node.get("file") or node.get("file_path", ""),
                        "start_line": node.get("start_line", 0),
                    }
    # Otherwise try graph name pattern search.
    result = client.call_tool(
        "search_graph",
        {"project": project, "name_pattern": f".*{target}.*", "limit": 10},
        timeout=60,
    )
    if not isinstance(result, dict):
        return None
    for node in result.get("results", []):
        if node.get("name") == target or node.get("qualified_name", "").endswith(f".{target}"):
            return node
    # Fallback to first result if any.
    if result.get("results"):
        return result["results"][0]
    return None


def _qualified_to_file_path(qualified_name: str) -> str:
    """Best-effort conversion from qualified_name to source file path."""
    parts = qualified_name.split(".")
    # Drop project name prefix (first segment) and the symbol suffix (last 1 or 2).
    if len(parts) < 2:
        return ""
    body = parts[1:]  # drop project name
    if len(body) > 1:
        body = body[:-1]  # drop function/method name
    if len(body) > 1 and body[-1][0].isupper():
        body = body[:-1]  # drop class name if present
    if not body:
        return ""
    return "/".join(body) + ".py"


def _collect_trace_nodes(trace_result: Any) -> list[dict[str, Any]]:
    if not isinstance(trace_result, dict):
        return []
    nodes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in ("callees", "callers"):
        for node in trace_result.get(key, []):
            if not isinstance(node, dict):
                continue
            qn = node.get("qualified_name", "")
            if not qn:
                continue
            if qn in seen:
                continue
            seen.add(qn)
            node = dict(node)
            if not node.get("file_path"):
                node["file_path"] = _qualified_to_file_path(qn)
            if not node.get("label"):
                node["label"] = "Function"
            nodes.append(node)
    return nodes


def _complexity_for_qualified_name(client: McpClient, project: str, qualified_name: str) -> dict[str, Any] | None:
    escaped = qualified_name.replace("'", "\\'")
    query = (
        "MATCH (f:Function|Method) WHERE f.qualified_name = '"
        + escaped
        + "' RETURN f.name, f.qualified_name, f.file_path, f.start_line, f.complexity, "
        "f.cognitive, f.transitive_loop_depth, f.linear_scan_in_loop, "
        "f.recursion_in_loop, f.is_entry_point, f.is_test, f.route_path "
        "LIMIT 1"
    )
    result = client.call_tool(
        "query_graph",
        {"project": project, "query": query, "max_rows": 5},
        timeout=120,
    )
    if not isinstance(result, dict):
        return None
    columns = result.get("columns", [])
    rows = result.get("rows", [])
    if not columns or not rows:
        return None
    row = rows[0]
    if not isinstance(row, list) or len(row) != len(columns):
        return None
    return dict(zip(columns, row))


def _complexity_for_names(client: McpClient, project: str, names: list[str]) -> dict[str, dict[str, Any]]:
    if not names:
        return {}
    # Escape single quotes in names for Cypher.
    escaped = [n.replace("'", "\\'") for n in names]
    names_literal = "[" + ", ".join(f"'{n}'" for n in escaped) + "]"
    query = (
        "MATCH (f:Function|Method) WHERE f.name IN "
        + names_literal
        + " RETURN f.name, f.qualified_name, f.file_path, f.start_line, f.complexity, "
        "f.cognitive, f.transitive_loop_depth, f.linear_scan_in_loop, "
        "f.recursion_in_loop, f.is_entry_point, f.is_test, f.route_path "
        "ORDER BY f.transitive_loop_depth DESC, f.complexity DESC"
    )
    result = client.call_tool(
        "query_graph",
        {"project": project, "query": query, "max_rows": 200},
        timeout=120,
    )
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(result, dict):
        return out
    columns = result.get("columns", [])
    rows = result.get("rows", [])
    if not columns:
        return out
    for row in rows:
        if not isinstance(row, list) or len(row) != len(columns):
            continue
        row_dict = dict(zip(columns, row))
        name = row_dict.get("f.name")
        if name:
            out[name] = row_dict
    return out


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def _int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return default


def _risk_score(complexity: dict[str, Any]) -> int:
    score = 0
    if _int(complexity.get("f.transitive_loop_depth"), 0) >= 3:
        score += 3
    elif _int(complexity.get("f.transitive_loop_depth"), 0) >= 2:
        score += 2
    if _int(complexity.get("f.complexity"), 0) >= 10:
        score += 3
    elif _int(complexity.get("f.complexity"), 0) >= 5:
        score += 2
    if _int(complexity.get("f.linear_scan_in_loop"), 0):
        score += 2
    if _normalize_bool(complexity.get("f.recursion_in_loop", False)):
        score += 2
    return score


def _risk_label(score: int) -> str:
    if score >= 7:
        return "🔴 HIGH"
    if score >= 4:
        return "🟡 MEDIUM"
    return "🟢 LOW"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Impact analysis via codebase-memory graph")
    parser.add_argument("target", help="Fonksiyon, sınıf veya dosya yolu")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Codebase-memory proje adı")
    parser.add_argument("--depth", type=int, default=3, help="trace_path derinliği")
    parser.add_argument("--json", action="store_true", help="JSON çıktı")
    args = parser.parse_args(argv)

    if not Path(CODEBASE_MEMORY_MCP).exists():
        print(f"Hata: codebase-memory-mcp bulunamadı: {CODEBASE_MEMORY_MCP}", file=sys.stderr)
        return 1

    client = McpClient(CODEBASE_MEMORY_MCP)
    try:
        symbol = _find_symbol(client, args.project, args.target)
        if not symbol:
            print(f"Hedef bulunamadı: {args.target}")
            return 1

        name = symbol.get("name", "")
        qualified = symbol.get("qualified_name", "")
        label = symbol.get("label", "")
        file_path = symbol.get("file_path", "")
        start_line = symbol.get("start_line", 0)

        trace_result = client.call_tool(
            "trace_path",
            {
                "project": args.project,
                "function_name": qualified,
                "mode": "calls",
                "direction": "both",
                "depth": args.depth,
                "include_tests": True,
            },
            timeout=180,
        )
        # If the name was ambiguous, retry with the first suggestion.
        if isinstance(trace_result, dict) and trace_result.get("status") == "ambiguous":
            suggestions = trace_result.get("suggestions", [])
            if suggestions:
                qualified = suggestions[0].get("qualified_name", qualified)
                trace_result = client.call_tool(
                    "trace_path",
                    {
                        "project": args.project,
                        "function_name": qualified,
                        "mode": "calls",
                        "direction": "both",
                        "depth": args.depth,
                        "include_tests": True,
                    },
                    timeout=180,
                )
        traced_nodes = _collect_trace_nodes(trace_result)
        # Complexity lookup for the target symbol uses its qualified name to avoid
        # collisions with same-named symbols elsewhere in the graph.
        target_complexity = _complexity_for_qualified_name(client, args.project, qualified) or {}

        # Complexity lookup for related symbols uses short names.
        related_names = [
            n.get("name", "") for n in traced_nodes
            if n.get("name") and n.get("name") != name
        ]
        complexity = _complexity_for_names(client, args.project, related_names)

        # Enrich trace nodes with graph metadata.
        for node in traced_nodes:
            c = complexity.get(node.get("name", ""), {})
            if c.get("f.file_path"):
                node["file_path"] = c["f.file_path"]
            if c.get("f.start_line"):
                node["start_line"] = c["f.start_line"]
            qn = node.get("qualified_name", "")
            is_route = (
                _normalize_bool(c.get("f.is_entry_point", False))
                or bool(c.get("f.route_path"))
                or ".routes." in qn
            )
            node["is_entry_point"] = is_route
            node["is_test"] = _normalize_bool(c.get("f.is_test", False))
            if node.get("is_entry_point"):
                node["label"] = "Route"
            elif node.get("is_test"):
                node["label"] = "Test"

        entry_points = [n for n in traced_nodes if n.get("is_entry_point") and not n.get("is_test")]
        tests = [n for n in traced_nodes if n.get("is_test")]
        others = [
            n for n in traced_nodes
            if not n.get("is_entry_point") and not n.get("is_test")
        ]

        # Enrich target symbol metadata from complexity query.
        if target_complexity.get("f.file_path"):
            file_path = target_complexity["f.file_path"]
        if target_complexity.get("f.start_line"):
            start_line = target_complexity["f.start_line"]
        target_risk_score = _risk_score(target_complexity)

        if args.json:
            print(
                json.dumps(
                    {
                        "target": {
                            "name": name,
                            "qualified_name": qualified,
                            "label": label,
                            "file_path": file_path,
                            "start_line": start_line,
                        },
                        "risk_score": target_risk_score,
                        "risk_label": _risk_label(target_risk_score),
                        "entry_points": entry_points,
                        "tests": tests,
                        "related_nodes": others,
                        "complexity": complexity,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0

        print(f"# Impact Analysis: `{name}`")
        print(f"- **Tür:** {label}")
        print(f"- **Konum:** {file_path}:{start_line}")
        print(f"- **Tam ad:** `{qualified}`")
        print(f"- **Risk:** {_risk_label(target_risk_score)} (skor: {target_risk_score})\n")

        if target_complexity:
            print("## Complexity metrikleri")
            print(f"- Cyclomatic complexity: {target_complexity.get('f.complexity', '-')}")
            print(f"- Cognitive: {target_complexity.get('f.cognitive', '-')}")
            print(f"- Transitive loop depth: {target_complexity.get('f.transitive_loop_depth', '-')}")
            print(f"- Linear scan in loop: {target_complexity.get('f.linear_scan_in_loop', '-')}")
            print(f"- Recursion in loop: {target_complexity.get('f.recursion_in_loop', False)}")
            print()

        def print_node_table(title: str, nodes: list[dict[str, Any]]) -> None:
            if not nodes:
                return
            print(f"## {title} ({len(nodes)})")
            print("| Sembol | Tür | Dosya | Karmaşıklık |")
            print("|--------|-----|-------|-------------|")
            for node in nodes[:30]:
                qn = node.get("qualified_name", "")
                short = qn.split(".")[-1] if qn else node.get("name", "")
                node_label = node.get("label", "")
                path = node.get("file_path", "")
                line = node.get("start_line", 0)
                loc = f"{path}:{line}" if path else "-"
                c = complexity.get(node.get("name", ""), {})
                comp = c.get("f.complexity", "-")
                print(f"| `{short}` | {node_label} | {loc} | {comp} |")
            print()

        print_node_table("Entry points", entry_points)
        print_node_table("Tests", tests)
        print_node_table("Related symbols", others)

        if not entry_points and not tests:
            print("> Not: Entry point veya test bulunamadı; hedef muhtemelen uç bir leaf fonksiyondur.")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
