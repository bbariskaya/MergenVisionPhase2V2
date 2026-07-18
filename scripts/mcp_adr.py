#!/usr/bin/env python3
"""ADR management helper for codebase-memory-mcp.

Kullanım:
    python scripts/mcp_adr.py --get
    python scripts/mcp_adr.py --init
    python scripts/mcp_adr.py --update --file docs/adr.md
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

DEFAULT_ADR = """## PURPOSE

MergenVision Phase2v2 is a face recognition platform backend. It ingests still images and video streams, maintains persistent person identities, and exposes REST APIs for enrollment, search, and video job management.

## STACK

- Python 3.12 + FastAPI + Pydantic v2
- SQLAlchemy 2.0 + Alembic (Postgres persistence)
- Qdrant (face embedding vector store)
- MinIO (object storage for source media)
- Native C++/CUDA runtime for image/video inference (optional lazy load)

## ARCHITECTURE

### Person aggregate root

`Person` is the aggregate root for identity management. A person owns zero or more `FaceIdentity` records, and each `FaceIdentity` owns zero or more `FaceSample` records. Samples carry the actual embedding and a pointer to the stored media object.

### Redirect / alias semantics

To keep historical video results stable while allowing identity merge/split, `FaceIdentity` supports `redirect_to_face_id`. Resolution always follows the redirect chain to the canonical identity before any read or enrollment operation.

### Bulk enrollment

The bulk enrollment service (`BulkEnrollmentService`) is implemented in Python first. It loops over photos one-by-one using the existing `ImageRecognitionEngine.detect_and_embed` path. A true GPU-hot-path batch enrollment will later be added in the native runtime (`infer_jpeg_batch`) and called from the service. The Python scaffold ensures the API contract, persistence flow, and tests are stable before the native optimization lands.

### Phase 1 → Phase 2 identity continuity

Both Phase 1 and Phase 2 share the same Qdrant collection and canonical identity resolution. A face enrolled in Phase 1 is recognized as a known person in Phase 2 video processing because both phases resolve to the same `FaceIdentity` and `Person` records.

### Smart context retrieval

A context orchestrator skill and `scripts/mcp_context.py` combine `search_graph`, `search_code`, and `trace_path` to answer natural-language code questions. `scripts/mcp_refresh.py` keeps the graph current after large changes, and `scripts/mcp_impact.py` maps the blast radius of a proposed change.

## PATTERNS

- Hexagonal / ports-and-adapters: domain, application ports, and infrastructure adapters are separated.
- Unit of Work: all writes go through `AbstractUnitOfWork` and are committed explicitly.
- Lazy native runtime initialization: `BulkEnrollmentService` uses a factory so the native image runtime is not loaded during app startup or health checks.
- Snapshot vs projection: video result records store immutable snapshot fields (`snapshot_face_id`, `snapshot_name`) alongside a `current_*` projection that follows redirects.

## TRADEOFFS

- Bulk enrollment is not yet batched on GPU; Python loop trades throughput for correctness and testability.
- Native runtime is optional at startup; tests can run without the compiled image runtime module.
- ADR and graph metadata are stored in codebase-memory-mcp, not prompt-memory, to avoid auto-writing memories.

## PHILOSOPHY

- Backend correctness before UI polish; backend before frontend changes.
- Stabilize contracts and persistence first, then swap in native GPU hot paths.
- Use the knowledge graph (`codebase-memory-mcp`) for exploration and impact analysis rather than blind grep.
"""


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
                "clientInfo": {"name": "mcp-adr-cli", "version": "1.0.0"},
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
    parser = argparse.ArgumentParser(description="ADR management for codebase-memory")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--get", action="store_true", help="Mevcut ADR'yi göster")
    group.add_argument("--init", action="store_true", help="Varsayılan ADR'yi oluştur")
    group.add_argument("--update", metavar="FILE", help="Belirtilen dosyayı ADR olarak kaydet")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Codebase-memory proje adı")
    args = parser.parse_args(argv)

    if not Path(CODEBASE_MEMORY_MCP).exists():
        print(f"Hata: codebase-memory-mcp bulunamadı: {CODEBASE_MEMORY_MCP}", file=sys.stderr)
        return 1

    if args.update:
        content = Path(args.update).read_text(encoding="utf-8")
    elif args.init:
        content = DEFAULT_ADR
    else:
        content = None

    client = McpClient(CODEBASE_MEMORY_MCP)
    try:
        if args.get:
            result = client.call_tool(
                "manage_adr",
                {"project": args.project, "mode": "get"},
                timeout=30,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0

        # init or update
        result = client.call_tool(
            "manage_adr",
            {"project": args.project, "mode": "update", "content": content},
            timeout=60,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
